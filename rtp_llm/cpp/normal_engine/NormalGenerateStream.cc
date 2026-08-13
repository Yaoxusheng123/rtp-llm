#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace rtp_llm {

namespace {
// 唤醒由 cv_ 的 notify 保证，这个时间片只决定 timeout_ms 的检查粒度。
constexpr int64_t OUTPUT_WAIT_SLICE_MS = 100;
}  // namespace

ErrorResult<GenerateOutputs> NormalGenerateStream::nextOutput() {
    // 谓词横跨三个条件：hasError()/getStatus() 由 mutex_ 保护并经 cv_ 通知，队列非空
    // 由队列自身的锁保护。必须统一在 mutex_ 上等待，否则谓词与睡眠不原子，通知会丢。
    // 不能用 autil 队列的 waitNotEmpty()：它拿锁后无条件 wait(1s)，且调用方的 isEmpty()
    // 与它是两次独立加锁，signal 落在两者之间即丢失，等待方要睡满 1 秒才恢复。
    {
        std::unique_lock<std::mutex> lock(*mutex_);
        auto                         ready = [this] {
            return hasError() || getStatus() == StreamState::FINISHED || !generate_outputs_queue_.isEmpty();
        };
        while (!ready()) {
            // 这里的超时只用于推进 timeout_ms 判定，唤醒本身由 cv_ 的 notify 保证。
            if (!cv_->wait_for(lock, std::chrono::milliseconds(OUTPUT_WAIT_SLICE_MS), ready)) {
                checkTimeoutWithoutLock();
            }
        }
    }
    if (hasError()) {
        return statusInfo();
    }
    if (generate_outputs_queue_.isEmpty()) {
        if (isFinished()) {
            return ErrorInfo(ErrorCode::FINISHED, "finished");
        } else {
            return ErrorInfo(ErrorCode::OUTPUT_QUEUE_IS_EMPTY, "output queue is empty");
        }
    }
    return generate_outputs_queue_.getAndPopFront();
}

bool NormalGenerateStream::hasOutput() {
    return !generate_outputs_queue_.isEmpty();
}

GenerateOutputs NormalGenerateStream::prepareGenerateOutput(const StreamUpdateInfo& update_info) {
    size_t          output_len = seqLength() - last_output_pos_;
    GenerateOutputs generate_results;
    generate_results.request_id = request_id_;

    for (int i = 0; i < nextBatchSize(); i++) {
        GenerateOutput generate_output;
        generate_output.aux_info.iter_count = iter_count_;
        generate_output.output_ids          = torch::empty({1, (int64_t)output_len}, torch::kInt32);

        // TODO(xinfei.sxf) optimize this copy : only copy last token
        complete_token_ids_->copyTokensTo(
            i, generate_output.output_ids.data_ptr<int32_t>(), last_output_pos_, output_len);
        if (returnLogits() && update_info.logits.defined()) {
            torch::Tensor logits_result;
            const auto&   select_tokens_id = generate_input_->generate_config->select_tokens_id;
            if (!select_tokens_id.empty()) {
                auto out_of_bound_token_id =
                    std::find_if_not(select_tokens_id.begin(), select_tokens_id.end(), [this](int select_token_id) {
                        return select_token_id >= 0 && select_token_id < vocabSize();
                    });
                if (out_of_bound_token_id == select_tokens_id.end()) {
                    auto select_indices =
                        torch::tensor(std::vector<int64_t>(select_tokens_id.begin(), select_tokens_id.end()),
                                      torch::kLong)
                            .to(update_info.logits.device());
                    logits_result = update_info.logits.index_select(1, select_indices);
                } else {
                    RTP_LLM_LOG_WARNING("select_token_id out of bound, expected >= 0 and < vocab size [%d], found [%d]",
                                        vocabSize(),
                                        *out_of_bound_token_id);
                    logits_result = torch::empty({0, 0}, update_info.logits.options());
                }
            } else {
                logits_result = update_info.logits;
            }
            if (logits_result.size(0) <= 1) {
                generate_output.logits = logits_result.cpu().clone();
            } else {
                generate_output.logits = logits_result.narrow(0, i, 1).cpu().clone();
            }
        }

        if (generate_input_->generate_config->return_hidden_states && update_info.hidden_states.defined()) {
            if (update_info.hidden_states.size(0) == 1) {
                generate_output.hidden_states = update_info.hidden_states.cpu();
            } else {
                generate_output.hidden_states = update_info.hidden_states.narrow(0, i, 1).cpu();
            }
        }
        if (generate_input_->generate_config->return_all_hidden_states && update_info.all_hidden_states.defined()
            && iter_count_ == 1) {
            generate_output.all_hidden_states = update_info.all_hidden_states.cpu();
        }
        if (loss_.defined()) {
            RTP_LLM_CHECK_WITH_INFO(loss_index_ == inputLength() - 1,
                                    "loss index should be input len [%d] - 1 but is [%d]",
                                    inputLength(),
                                    loss_index_);
            if (generate_input_->generate_config->calculate_loss == 1) {
                generate_output.loss = torch::mean(loss_).exp().cpu().unsqueeze(0);
            } else {
                generate_output.loss = loss_;
            }
        }

        generate_output.finished = isSubGenerateDoneWithoutLock(i);
        if (generate_input_->generate_config->aux_info) {
            generate_output.aux_info.iter_count   = iter_count_;
            generate_output.aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            generate_output.aux_info.first_token_cost_time_us = complete_token_ids_->firstTokenLatencyUs();
            generate_output.aux_info.wait_time_us             = wait_time_us_;
            generate_output.aux_info.input_len                = generate_input_->promptLength();
            generate_output.aux_info.prefix_len               = generate_input_->prefix_length;
            // TODO(xinfei.sxf) 提前结束的query，output len要设置正确
            generate_output.aux_info.output_len       = seqLength() - generate_input_->inputLength();
            generate_output.aux_info.step_output_len  = output_len;
            generate_output.aux_info.reuse_len        = initial_reuse_length_;
            generate_output.aux_info.pd_sep           = queryPdSep();
            generate_output.aux_info.local_reuse_len  = local_reuse_length_;
            generate_output.aux_info.remote_reuse_len = remote_reuse_length_;
            generate_output.aux_info.memory_reuse_len = memory_reuse_length_;
            if (generate_input_->generate_config->return_softmax_probs && softmax_probs_.defined()) {
                generate_output.aux_info.softmax_probs =
                    softmax_probs_[i].narrow(0, last_output_pos_, output_len).clone();
            }
            if (update_info.cum_log_probs.defined()) {
                generate_output.aux_info.cum_log_probs = cum_log_probs_.narrow(0, i, 1).cpu().clone();
            }
            if (generate_input_->generate_config->return_all_probs) {
                if (!update_info.all_probs.defined()) {
                    throw std::runtime_error("all_probs is not while generate_config return_all_probs is true");
                }
                generate_output.aux_info.all_probs = all_probs_.narrow(0, i, 1).clone();
            }
        }
        // hidden_states post process
        if (generate_output.finished && generate_input_->generate_config->return_hidden_states
            && generate_output.hidden_states.has_value()
            && (generate_input_->generate_config->hidden_states_cut_dim > 0
                || generate_input_->generate_config->normalized_hidden_states)) {
            auto hidden_states_tensor = generate_output.hidden_states.value();
            if (generate_input_->generate_config->hidden_states_cut_dim > 0) {
                hidden_states_tensor = hidden_states_tensor.index(
                    {torch::indexing::Slice(),
                     torch::indexing::Slice(0, generate_input_->generate_config->hidden_states_cut_dim)});
            }
            if (generate_input_->generate_config->normalized_hidden_states) {
                hidden_states_tensor = torch::nn::functional::normalize(
                    hidden_states_tensor, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
            }
            generate_output.hidden_states = hidden_states_tensor.cpu().clone();
        }

        generate_results.generate_outputs.emplace_back(std::move(generate_output));
    }
    return generate_results;
}

void NormalGenerateStream::enqueueGenerateOutput(GenerateOutputs&& generate_results) {
    if (generate_outputs_queue_.getSize() >= generate_outputs_queue_.getCapacity()) {
        /* No matter if the queue is full for any reason,
           the stream will be set to stop directly to prevent the push to queue from getting stuck. */
        reportEventWithoutLock(StreamEvents::Error, ErrorCode::OUTPUT_QUEUE_FULL, "output queue is full");
    } else {
        generate_outputs_queue_.push(std::move(generate_results));
        // 调用链（update/specUpdate -> updateOutput）全程持有 mutex_，因此这里的通知不会丢。
        cv_->notify_all();
    }
}

void NormalGenerateStream::updateOutput(const StreamUpdateInfo& update_info) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // TODO(xinfei.sxf) consider the case of pd-sep first token finished.

    if (update_info.loss.defined()) {
        setLoss(update_info.loss);
    }

    // TODO(wangyin.yx): check behaviour of update_info.hidden_states under mtp/eagle model
    if (needReturnHiddenStates() && update_info.all_hidden_states.defined()) {
        last_hidden_states_ = update_info.all_hidden_states;
    }

    if (generate_input_->generate_config->return_softmax_probs && update_info.softmax_probs.defined()) {
        RTP_LLM_CHECK(update_info.softmax_probs.dim() == 2);
        RTP_LLM_CHECK(update_info.softmax_probs.size(1) == update_info.num_new_tokens);
        setSoftmaxProbs(update_info.softmax_probs, seqLength() - update_info.num_new_tokens);
    }

    finished_ = needFinish();
    if (finished_) {
        reportEventWithoutLock(StreamEvents::GenerateDone);
        fillSubGenerateStatus(StreamState::FINISHED);
    }
    if (update_info.cum_log_probs.defined()) {
        cum_log_probs_ = update_info.cum_log_probs.cpu();
    }
    if (update_info.all_probs.defined()) {
        all_probs_ = update_info.all_probs.cpu();
    }

    // TODO: move it to better position
    RTP_LLM_LOG_DEBUG("stream [%ld] finished: %d, pd_sep: %d, is_streaming: %d, need_remote_generate: %d",
                      streamId(),
                      finished_,
                      queryPdSep(),
                      isStreaming(),
                      update_info.update_remote_generate);

    if (!finished_ && queryPdSep() && update_info.update_remote_generate) {
        holdKVCacheForPDSep();
        reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
        reportEventWithoutLock(StreamEvents::GenerateDone);
    }

    bool pd_sep_first_token = queryPdSep();
    bool need_update        = pd_sep_first_token || isStreaming() || finished_;
    if (!need_update) {
        return;
    }

    if (seqLength() - last_output_pos_ == 0) {
        return;
    }

    RTP_LLM_LOG_DEBUG("stream [%ld] enqueue generate output", streamId());
    enqueueGenerateOutput(prepareGenerateOutput(update_info));

    if (hasError()) {
        return;
    }

    last_output_pos_ = seqLength();
}
};  // namespace rtp_llm
