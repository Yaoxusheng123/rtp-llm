from rtp_llm.server.server_args.util import str2bool


def init_fifo_scheduler_group_args(parser, fifo_scheduler_config):
    ##############################################################################################################
    # FIFO 调度器配置
    ##############################################################################################################
    fifo_scheduler_group = parser.add_argument_group("FIFO Scheduler")

    fifo_scheduler_group.add_argument(
        "--max_context_batch_size",
        env_name="MAX_CONTEXT_BATCH_SIZE",
        bind_to=[(fifo_scheduler_config, "max_context_batch_size")],
        type=int,
        default=1,
        help="（设备参数）为设备参数设置的最大 context batch size，影响默认调度器的凑批决策。",
    )
    fifo_scheduler_group.add_argument(
        "--max_batch_tokens_size",
        env_name="MAX_BATCH_TOKENS_SIZE",
        bind_to=[(fifo_scheduler_config, "max_batch_tokens_size")],
        type=int,
        default=0,
        help="最大 batch tokens 大小。",
    )
    fifo_scheduler_group.add_argument(
        "--enable_mixed_batch",
        env_name="ENABLE_MIXED_BATCH",
        bind_to=[(fifo_scheduler_config, "enable_mixed_batch")],
        type=str2bool,
        default=False,
        help="是否允许 prefill 与 decode 混在同一批调度。关闭时新请求必须等当前批全部 decode 完成才能入场。"
        "不支持投机采样、多模态输入和 prefill context parallel。",
    )
    fifo_scheduler_group.add_argument(
        "--mixed_batch_max_prefill_tokens",
        env_name="MIXED_BATCH_MAX_PREFILL_TOKENS",
        bind_to=[(fifo_scheduler_config, "mixed_batch_max_prefill_tokens")],
        type=int,
        default=0,
        help="混批时 context 部分的 token 预算，用于限制 prefill 对 decode 每步耗时的影响。0 表示复用 max_batch_tokens_size。",
    )
