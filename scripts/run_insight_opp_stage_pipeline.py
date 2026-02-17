#!/usr/bin/env python3
from __future__ import annotations

import insight_opp_stage_pipeline_lib as pipeline_lib
from insight_pipeline_common import PipelineConfig, build_parser, run_pipeline

CONFIG = PipelineConfig(
    description="Run full insight opp-stage golden pipeline",
    output_prefix="insight_opp_stage_bg",
    report_title="Insight Opportunity Stage Golden Dataset Report",
    query_result_line="slug-based filtering is required for this prompt's logs.",
    prompt_name_default=None,
    sample_per_window_default=1000,
    prompt_name_count_source="prompt_slug",
    prompt_name_count_label="`metadata.prompt_name` (same slug string)",
)


def main() -> int:
    parser = build_parser(CONFIG, pipeline_lib.PROJECT_ID, pipeline_lib.PROMPT_SLUG)
    args = parser.parse_args()
    return run_pipeline(CONFIG, pipeline_lib, args)


if __name__ == "__main__":
    raise SystemExit(main())
