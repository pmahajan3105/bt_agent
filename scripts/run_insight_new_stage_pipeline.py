#!/usr/bin/env python3
from __future__ import annotations

import insight_new_stage_pipeline_lib as pipeline_lib
from insight_pipeline_common import PipelineConfig, build_parser, run_pipeline

CONFIG = PipelineConfig(
    description="Run full insight new-stage golden pipeline",
    output_prefix="insight_new_stage_bg",
    report_title="Insight New Stage Golden Dataset Report",
    query_result_line="slug and prompt-name both checked; slug used for fetch consistency.",
    prompt_name_default="account-overview-new-summarizer-background",
    sample_per_window_default=40,
    prompt_name_count_source="prompt_name",
    prompt_name_count_label="`metadata.prompt_name`",
)


def main() -> int:
    parser = build_parser(CONFIG, pipeline_lib.PROJECT_ID, pipeline_lib.PROMPT_SLUG)
    args = parser.parse_args()
    return run_pipeline(CONFIG, pipeline_lib, args)


if __name__ == "__main__":
    raise SystemExit(main())
