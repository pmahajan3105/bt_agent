---
name: golden-trace-selector
description: Select high-quality, diverse traces for golden evaluation datasets. Use when asked to find traces for a prompt, build a golden dataset, or select eval examples. First reads the prompt to understand it, then reasons about what makes good traces for that specific prompt.
---

# Golden Trace Selector Skill

## Purpose

Select high-quality, diverse production traces that can serve as golden examples for evaluating a Braintrust prompt. The key insight is that **what makes a "good" trace depends entirely on what the prompt does** - so you must first understand the prompt before selecting traces.

## When to Use

Use this skill when asked to:
- Select golden traces for a prompt
- Build an evaluation dataset
- Find representative examples for a prompt
- Create test cases from production data

---

## The Process: Read → Reason → Select

### Phase 1: Understand the Prompt

**First, fetch and read the prompt definition:**

```bash
python /Users/prashant/bt_agent/prompt_utils.py fetch-prompt --slug <PROMPT_SLUG>
```

**Read the system message carefully and answer these questions:**

1. **What task does this prompt perform?**
   - Classification? (assigns categories/labels)
   - Extraction? (pulls structured data from unstructured input)
   - Generation? (creates new content)
   - Parsing? (interprets user intent)
   - Summarization? (condenses information)
   - Other?

2. **What output format/schema does it expect?**
   - Look for JSON schemas, field definitions, or examples in the prompt
   - Identify required vs optional fields
   - Note any enums, categories, or constrained values

3. **What are the key output fields?**
   - Which fields are critical for the task?
   - Are there classification fields with specific allowed values?
   - Are there array fields that could have 0, 1, or many items?

4. **What input does it expect?**
   - What kind of data does the user provide?
   - Are there different input types or scenarios?

---

### Phase 2: Define Quality Criteria

**Based on your prompt understanding, determine what makes a HIGH-QUALITY trace:**

Think through these dimensions:

| Dimension | Question to Answer |
|-----------|-------------------|
| **Valid Input** | Does the trace have non-empty, realistic input data? |
| **No Errors** | Did the trace complete without errors? |
| **Schema Match** | Does the output match the expected schema from the prompt? |
| **Substantive** | Is the output meaningful (not empty defaults)? |
| **Complete** | Are all required fields present and populated? |

**Write down your quality criteria specific to this prompt.**

Example for a classifier prompt:
> "Quality trace = has email input, no error, output parses as JSON with email_tags array"

Example for an extractor prompt:
> "Quality trace = has conversation input, no error, output has tasks array with valid task objects"

---

### Phase 3: Define Diversity Criteria

**Based on the prompt's purpose, determine what DIVERSITY means for this prompt:**

Different prompt types need different diversity:

| Prompt Type | Diversity Should Cover |
|-------------|----------------------|
| **Classifier** | All output categories (balanced representation) |
| **Extractor** | Varying extraction counts (0, 1, few, many items) |
| **Generator** | Different input scenarios and use cases |
| **Parser** | Different intent types and complexity levels |
| **Summarizer** | Different input lengths and content types |

**Think about:**
- What are the "buckets" or "categories" outputs can fall into?
- What edge cases should be represented?
- What input variations matter?

**Write down your diversity criteria specific to this prompt.**

Example for email-tagger:
> "Diversity = balance across 7 categories: ooo, automated, bounced, calendar_invite, positive_sentiment, negative_sentiment, default"

Example for task-extraction:
> "Diversity = vary by task count: some with 0 tasks, some with 1, some with 2-5, some with 6+"

---

### Phase 4: Fetch and Analyze Traces

**Fetch recent traces:**

```bash
python /Users/prashant/bt_agent/prompt_utils.py fetch-traces --slug <PROMPT_SLUG> --hours 168 --limit 500 --output /tmp/traces.json
```

**Read the traces file and analyze:**

1. For each trace, evaluate against your quality criteria
2. Categorize each trace by your diversity criteria
3. Track the distribution across diversity buckets

**Build a mental model of what's available:**
- How many traces pass quality filters?
- What's the distribution across diversity buckets?
- Are any buckets under-represented?

---

### Phase 5: Select the Golden Set

**Target: ~50 traces** (adjustable based on needs)

**Selection strategy:**

1. **Start with diversity** - ensure all buckets are represented
2. **Within each bucket** - pick highest quality traces
3. **Verify coverage** - check that edge cases are included
4. **Balance quantity** - aim for roughly equal traces per bucket

**Example selection for email-tagger (50 traces, 7 categories):**
- ooo: 7 traces
- automated: 7 traces
- bounced: 7 traces
- calendar_invite: 7 traces
- positive_sentiment: 7 traces
- negative_sentiment: 7 traces
- default: 8 traces

**If a bucket is under-represented:**
- Note this in the output
- Fetch more traces with longer time range if needed
- Accept partial coverage if data doesn't exist

---

### Phase 6: Save Results

**After selecting traces, save them:**

Create a JSON file with selected trace IDs, then use the save tool:

```bash
python /Users/prashant/bt_agent/prompt_utils.py save-golden --input /tmp/selected_traces.json --output golden_dataset.json
```

**Document your selection:**
- What quality criteria you used
- What diversity criteria you used
- The distribution across buckets
- Any gaps or limitations

---

## Output Template

When delivering results, provide:

```
## Golden Trace Selection: {prompt_slug}

### Prompt Understanding
- **Task type**: [classifier/extractor/generator/parser/summarizer]
- **Output schema**: [key fields and their types]
- **Key categories/buckets**: [list them]

### Quality Criteria
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

### Diversity Criteria
- [Dimension 1]: [values to balance]
- [Dimension 2]: [values to balance]

### Selection Results

| Bucket | Target | Selected | Notes |
|--------|--------|----------|-------|
| [bucket1] | X | Y | |
| [bucket2] | X | Y | |

**Total selected**: N traces
**Saved to**: [file path]

### Gaps/Limitations
- [Any missing coverage]
- [Any quality issues noted]
```

---

## Tools Reference

**List all prompts:**
```bash
python /Users/prashant/bt_agent/prompt_utils.py list-prompts
```

**Fetch prompt definition:**
```bash
python /Users/prashant/bt_agent/prompt_utils.py fetch-prompt --slug <name>
python /Users/prashant/bt_agent/prompt_utils.py fetch-prompt --slug <name> --json  # Full JSON
```

**Fetch traces:**
```bash
python /Users/prashant/bt_agent/prompt_utils.py fetch-traces --slug <name> --hours 168 --limit 500 --output traces.json
```

**Save golden dataset:**
```bash
python /Users/prashant/bt_agent/prompt_utils.py save-golden --input traces.json --output golden.json
```

---

## Key Principles

1. **Always read the prompt first** - Never assume what a prompt does based on its name
2. **Let the prompt guide your criteria** - Quality and diversity definitions come FROM the prompt
3. **Document your reasoning** - Explain why you chose specific criteria
4. **Verify coverage** - Check that the final set actually achieves diversity goals
5. **Note limitations** - If data is sparse, say so rather than forcing bad selections

---

## Example Walkthrough

**User**: "Select golden traces for email-tagger"

**Step 1**: Fetch prompt
```bash
python /Users/prashant/bt_agent/prompt_utils.py fetch-prompt --slug email-tagger
```

**Step 2**: Read and understand
> "This prompt classifies emails into categories. Output is `{email_tags: [...], explanation: ...}`.
> Categories are: ooo, automated, bounced, calendar_invite, positive_sentiment, negative_sentiment, default"

**Step 3**: Define criteria
> Quality: has email input, no error, valid JSON with email_tags array
> Diversity: balance across 7 categories

**Step 4**: Fetch traces
```bash
python /Users/prashant/bt_agent/prompt_utils.py fetch-traces --slug email-tagger --hours 168 --limit 500 --output /tmp/traces.json
```

**Step 5**: Analyze and select
> Read traces, filter by quality, categorize by tag, select ~7 per category

**Step 6**: Save and document
> Save selected traces, report distribution
