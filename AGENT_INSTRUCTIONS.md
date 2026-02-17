## bt_agent safety instructions (read this before running)

These are **operational rules** for using `bt_agent.py` safely, especially when the tool can mutate Braintrust datasets.

### Golden rules

- **Never use `--yes` when you intend to review changes.**
  - `--yes` is a global flag that skips confirmation prompts.
- **Always run a dry-run first**, inspect output, then apply.
- **Prefer small, reversible changes** (e.g., patch only one field at a time).

### Safe workflow: update dataset description

1) Preview what the tool *would* write:

```bash
cd /path/to/bt_agent && python3 bt_agent.py dataset describe --id "<dataset_id>"
```

2) If it looks good, apply it (this should prompt you to confirm):

```bash
cd /path/to/bt_agent && python3 bt_agent.py dataset describe --id "<dataset_id>" --apply
```

### Agent mode guidance (LLM planner)

If you use `bt_agent.py agent ...`, keep these constraints:

- **Planner must never invent IDs** (dataset_id/project_id).
- **Planner must propose a plan first** (read-only steps first if possible).
- If the plan includes a mutation (`dataset.patch`), it should:
  - keep the patch minimal (only required fields)
  - include a clear reason
  - ask the user to confirm before execution

### Notes

- `.env` is loaded automatically from the repo root (or `BT_AGENT_ENV_FILE`).
- Keep secrets only in `.env` and do not commit it.

