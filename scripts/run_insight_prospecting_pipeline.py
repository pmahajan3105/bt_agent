#!/usr/bin/env python3
from __future__ import annotations

import insight_prospecting_pipeline_lib as pipeline_lib
from insight_pipeline_common import PipelineConfig, build_parser, run_pipeline

CONFIG = PipelineConfig(
    description="Run full insight prospecting golden pipeline",
    output_prefix="insight_prospecting_bg",
    report_title="Insight Prospecting Stage Golden Dataset Report",
    query_result_line="both prompt_name and prompt_slug are present, slug filter used for consistency with logs pipeline.",
    prompt_name_default="account-overview-prospecting-summarizer-background",
    sample_per_window_default=1000,
    prompt_name_count_source="prompt_name",
    prompt_name_count_label="`metadata.prompt_name`",
)


def main() -> int:
    parser = build_parser(CONFIG, pipeline_lib.PROJECT_ID, pipeline_lib.PROMPT_SLUG)
    args = parser.parse_args()
    return run_pipeline(CONFIG, pipeline_lib, args)


if __name__ == "__main__":
    raise SystemExit(main())
