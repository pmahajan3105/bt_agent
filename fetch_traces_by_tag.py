#!/usr/bin/env python3
"""
Fetch email-tagger traces filtered by output.email_tags category.

This script queries Braintrust for traces from the email-tagger prompt,
filtering by the classification output (email_tags), and saves them
to organized folders.

Usage Examples:
    # Fetch 100 traces for all categories
    python fetch_traces_by_tag.py

    # Fetch 50 traces for specific categories
    python fetch_traces_by_tag.py --categories automated,ooo --limit 50

    # Fetch from last 48 hours instead of default 7 days
    python fetch_traces_by_tag.py --hours 48

    # Use custom output directory
    python fetch_traces_by_tag.py --output-dir ./my_traces

    # Specify project and prompt
    python fetch_traces_by_tag.py --project-id <UUID> --prompt-name my-prompt

Environment Variables:
    BRAINTRUST_API_KEY    - Required: Your Braintrust API key
    BRAINTRUST_PROJECT_ID - Optional: Default project ID
    BRAINTRUST_BASE_URL   - Optional: API base URL (default: https://api.braintrust.dev)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Defaults
DEFAULT_BASE_URL = "https://api.braintrust.dev"
DEFAULT_CATEGORIES = [
    "ooo",
    "automated",
    "bounced",
    "calendar_invite",
    "positive_sentiment",
    "negative_sentiment",
    "default",
]
DEFAULT_HOURS = 168  # 7 days
DEFAULT_LIMIT = 100
DEFAULT_PROMPT_NAME = "email-tagger"


def load_dotenv() -> None:
    """Load .env file from script directory if it exists."""
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass
class FetchConfig:
    """Configuration for trace fetching."""
    api_key: str
    base_url: str
    project_id: str
    prompt_name: str
    hours: int
    limit: int
    output_dir: Path
    categories: List[str]


@dataclass
class FetchResult:
    """Result of fetching traces for a single category."""
    category: str
    count: int
    success: bool
    error: Optional[str] = None
    file_path: Optional[Path] = None


class BraintrustClient:
    """Simple Braintrust API client for BTQL queries."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def btql(self, query: str) -> Dict[str, Any]:
        """Execute a BTQL query and return results."""
        resp = self.session.post(
            f"{self.base_url}/btql",
            json={"query": query, "fmt": "json"},
            timeout=self.timeout,
        )

        if resp.status_code >= 400:
            try:
                err = resp.json()
                msg = err.get("Message", resp.text[:500])
            except Exception:
                msg = resp.text[:500]
            raise RuntimeError(f"BTQL error ({resp.status_code}): {msg}")

        return resp.json()


def build_query(
    project_id: str,
    prompt_name: str,
    category: str,
    hours: int,
    limit: int,
) -> str:
    """
    Build a BTQL query to fetch traces filtered by email_tags category.

    Note: The output field is stored as a JSON string in Braintrust, so we use
    LIKE pattern matching instead of JSON operators.
    """
    # Escape the category for safe string matching
    escaped_category = category.replace("'", "''")

    query = f"""select: *
from: project_logs('{project_id}')
spans
filter: created > now() - interval {hours} hour and created < now() and span_attributes.parent_span_id is null and span_attributes.name = 'chat_completions' and metadata.prompt_name = '{prompt_name}' and output LIKE '%"{escaped_category}"%'
sort: created desc
limit: {limit}"""

    return query


def fetch_category(
    client: BraintrustClient,
    config: FetchConfig,
    category: str,
) -> FetchResult:
    """Fetch traces for a single category."""
    try:
        query = build_query(
            project_id=config.project_id,
            prompt_name=config.prompt_name,
            category=category,
            hours=config.hours,
            limit=config.limit,
        )

        result = client.btql(query)
        data = result.get("data", [])
        count = len(data)

        # Create output directory
        category_dir = config.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Save traces
        traces_file = category_dir / "traces.json"
        traces_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Save metadata
        metadata = {
            "category": category,
            "count": count,
            "limit": config.limit,
            "hours": config.hours,
            "prompt_name": config.prompt_name,
            "project_id": config.project_id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "complete": count >= config.limit,
        }
        metadata_file = category_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return FetchResult(
            category=category,
            count=count,
            success=True,
            file_path=traces_file,
        )

    except Exception as e:
        return FetchResult(
            category=category,
            count=0,
            success=False,
            error=str(e),
        )


def fetch_all_categories(config: FetchConfig, parallel: bool = True) -> List[FetchResult]:
    """Fetch traces for all configured categories."""
    client = BraintrustClient(
        api_key=config.api_key,
        base_url=config.base_url,
    )

    results: List[FetchResult] = []

    if parallel:
        # Fetch categories in parallel
        with ThreadPoolExecutor(max_workers=min(len(config.categories), 7)) as executor:
            futures = {
                executor.submit(fetch_category, client, config, cat): cat
                for cat in config.categories
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Print progress
                status = "✓" if result.success else "✗"
                if result.success:
                    complete = "✓" if result.count >= config.limit else f"({result.count}/{config.limit})"
                    print(f"  {status} {result.category}: {result.count} traces {complete}")
                else:
                    print(f"  {status} {result.category}: ERROR - {result.error}")
    else:
        # Fetch sequentially
        for cat in config.categories:
            print(f"  Fetching {cat}...", end=" ", flush=True)
            result = fetch_category(client, config, cat)
            results.append(result)

            if result.success:
                print(f"{result.count} traces")
            else:
                print(f"ERROR - {result.error}")

    return results


def print_summary(results: List[FetchResult], config: FetchConfig) -> None:
    """Print a summary of the fetch operation."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_traces = 0
    complete_count = 0

    for r in sorted(results, key=lambda x: x.category):
        if r.success:
            total_traces += r.count
            status = "✓ Complete" if r.count >= config.limit else f"⚠ Partial ({r.count})"
            if r.count >= config.limit:
                complete_count += 1
        else:
            status = f"✗ Failed: {r.error}"
        print(f"  {r.category:20s} {status}")

    print()
    print(f"Total traces fetched: {total_traces}")
    print(f"Categories complete:  {complete_count}/{len(results)}")
    print(f"Output directory:     {config.output_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Fetch email-tagger traces filtered by output category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--api-key",
        default=os.environ.get("BRAINTRUST_API_KEY"),
        help="Braintrust API key (or set BRAINTRUST_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BRAINTRUST_BASE_URL", DEFAULT_BASE_URL),
        help=f"Braintrust API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("BRAINTRUST_PROJECT_ID"),
        help="Braintrust project ID (or set BRAINTRUST_PROJECT_ID)",
    )
    parser.add_argument(
        "--prompt-name",
        default=DEFAULT_PROMPT_NAME,
        help=f"Prompt name to filter by (default: {DEFAULT_PROMPT_NAME})",
    )
    parser.add_argument(
        "--categories", "-c",
        default=",".join(DEFAULT_CATEGORIES),
        help=f"Comma-separated list of categories (default: {','.join(DEFAULT_CATEGORIES)})",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help=f"Hours of history to fetch (default: {DEFAULT_HOURS})",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max traces per category (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(Path(__file__).resolve().parent / "traces"),
        help="Output directory for traces (default: ./traces)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Fetch categories sequentially instead of in parallel",
    )

    args = parser.parse_args(argv)

    # Validate required args
    if not args.api_key:
        print("Error: Missing API key. Set --api-key or BRAINTRUST_API_KEY", file=sys.stderr)
        return 1

    if not args.project_id:
        print("Error: Missing project ID. Set --project-id or BRAINTRUST_PROJECT_ID", file=sys.stderr)
        return 1

    # Parse categories
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        print("Error: No categories specified", file=sys.stderr)
        return 1

    config = FetchConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        project_id=args.project_id,
        prompt_name=args.prompt_name,
        hours=args.hours,
        limit=args.limit,
        output_dir=Path(args.output_dir).resolve(),
        categories=categories,
    )

    # Print configuration
    print("Fetching email-tagger traces by category")
    print("=" * 60)
    print(f"  Project ID:   {config.project_id}")
    print(f"  Prompt:       {config.prompt_name}")
    print(f"  Time range:   last {config.hours} hours")
    print(f"  Limit/cat:    {config.limit}")
    print(f"  Categories:   {', '.join(config.categories)}")
    print(f"  Output dir:   {config.output_dir}")
    print("=" * 60)
    print()
    print("Fetching...")

    # Fetch traces
    results = fetch_all_categories(config, parallel=not args.sequential)

    # Print summary
    print_summary(results, config)

    # Return non-zero if any failures
    if any(not r.success for r in results):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
