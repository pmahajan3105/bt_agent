## `bt_agent.py` (local Braintrust CLI + “plan/confirm/execute” agent)

This is a **local** script that talks directly to the Braintrust API using your `BRAINTRUST_API_KEY`.

It supports:
- **Dataset update** via `PATCH /v1/dataset/{dataset_id}`
- **Trace/log fetching** via **BTQL** (`POST /btql`)
- An optional **LLM planner** that turns an instruction into a concrete plan, then asks you to **confirm** before applying any mutations

### Setup

- **Install dependencies**:

```bash
cd /Users/prashant/bt_agent && python3 -m pip install -r requirements.txt
```

- **Create a `.env` file (recommended)**:
  - Create `/Users/prashant/bt_agent/.env` and put your keys in it. The script **auto-loads** this file on every run.

Example `.env`:

```bash
BRAINTRUST_API_KEY="..."
# Recommended for `traces` so you don't have to pass --project-id every time:
BRAINTRUST_PROJECT_ID="..."
# Optional:
# BRAINTRUST_BASE_URL="https://api.braintrust.dev"
# OPENAI_API_KEY="..."     # (only needed for `agent` mode; you can also use LLM_API_KEY)
# LLM_MODEL="gpt-4o"
```

- **Braintrust key (alternative)**:
  - If you don’t want a `.env`, you can still use: `export BRAINTRUST_API_KEY="..."` (required)
- **Optional base URL** (defaults to `https://api.braintrust.dev`):
  - `export BRAINTRUST_BASE_URL="https://api.braintrust.dev"`
- **Optional LLM key** (only needed for `agent` mode):
  - `export OPENAI_API_KEY="..."` (or `LLM_API_KEY`)

### `.env` loading details

- By default, the script loads `.env` from the same directory as `bt_agent.py` (the repo root).
- To load a different env file, set `BT_AGENT_ENV_FILE` to a path, e.g.:

```bash
BT_AGENT_ENV_FILE="/absolute/path/to/some.env" python3 bt_agent.py --help
```

### Run

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py --help
```

### Update a dataset (PATCH)

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py   dataset patch   --id "<dataset_id>"   --json-file "/absolute/path/to/patch.json"
```

### Fetch 1 example row from a dataset

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py dataset fetch --id "<dataset_id>" --limit 1
```

### Auto-generate a dataset description (and optionally apply it)

Dry-run (prints the suggested description):

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py dataset describe --id "<dataset_id>"
```

Apply (will prompt to confirm unless you add `--yes`):

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py dataset describe --id "<dataset_id>" --apply
```

Note: `--yes` is a global flag, so it must come before the subcommand, e.g.:

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py --yes dataset describe --id "<dataset_id>" --apply
```

### Fetch traces for a prompt (BTQL) + basic local analysis

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py   traces   --project-id "<project_id>"   --prompt-name "chat-query-parser"   --hours 24   --max-traces 200   --save-traces-as "/tmp/bt_traces.json"   --save-analysis-as "/tmp/bt_traces_analysis.json"
```

### Run a raw BTQL query

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py   btql   --query-file "/absolute/path/to/query.btql"   --save-as "/tmp/btql_result.json"
```

### Agent mode (LLM plans actions; you confirm before execute)

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py   agent   "Patch dataset 123e4567-e89b-12d3-a456-426614174000 to rename it to 'My New Dataset'"   --save-plan-as "/tmp/bt_agent_plan.json"
```

If you want to skip confirmations (dangerous), add `--yes`.

### Agent safety instructions file (recommended)

There is an editable file at `AGENT_INSTRUCTIONS.md`. You can pass it to agent mode like this:

```bash
cd /Users/prashant/bt_agent && python3 bt_agent.py agent "..." --instruction-file "/Users/prashant/bt_agent/AGENT_INSTRUCTIONS.md"
```
