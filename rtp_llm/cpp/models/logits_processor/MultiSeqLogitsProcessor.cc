#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"

#include <limits>

namespace rtp_llm {

std::shared_ptr<MultiSeqLogitsProcessor>
MultiSeqLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input, int64_t eos_token_id) {

    if (generate_input->generate_config->num_return_sequences <= 1 && !generate_input->generate_config->hasNumBeams()) {
        return nullptr;
    }

    auto processor_ptr           = std::make_shared<MultiSeqLogitsProcessor>();
    processor_ptr->eos_token_id_ = eos_token_id;

    return processor_ptr;
}

void MultiSeqLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    size_t batch_size = finish_idx - start_idx;
    size_t vocab_size = inputs.logits.size(1);

    auto logits            = inputs.logits.narrow(0, start_idx, batch_size);
    auto finished_mask_ptr = reinterpret_cast<bool*>(inputs.finished_mask.data_ptr()) + start_idx;

    // return early when no sequence needs processing
    if (!std::any_of(finished_mask_ptr, finished_mask_ptr + batch_size, [](bool v) { return v; })) {
        return;
    }

    // Mask all logits of the finished sequences except the eos token.
    // Expressed as device-side fills over runs of adjacent finished rows rather than a
    // [batch_size, vocab_size] host mask: that mask has to reach the device through a pageable
    // H2D copy, and a pageable H2D implicitly drains the whole CUDA stream.
    // An out-of-range eos leaves the row fully masked, matching an all-ones mask.
    const auto neg_inf_logit = -std::numeric_limits<float>::infinity();
    const bool restore_eos   = eos_token_id_ < vocab_size;

    for (size_t idx = 0; idx < batch_size;) {
        if (!finished_mask_ptr[idx]) {
            ++idx;
            continue;
        }
        size_t run_end = idx + 1;
        while (run_end < batch_size && finished_mask_ptr[run_end]) {
            ++run_end;
        }

        auto rows = logits.narrow(0, idx, run_end - idx);
        if (restore_eos) {
            auto eos_column = rows.narrow(1, eos_token_id_, 1).clone();
            rows.fill_(neg_inf_logit);
            rows.narrow(1, eos_token_id_, 1).copy_(eos_column);
        } else {
            rows.fill_(neg_inf_logit);
        }
        idx = run_end;
    }
}

void MultiSeqLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    // do nothing
}

void MultiSeqLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    // do nothing
}

}  // namespace rtp_llm