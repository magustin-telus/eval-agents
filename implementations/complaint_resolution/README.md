# Complaint Resolution Agent

A retrieval-augmented (RAG) customer complaint-resolution agent and a four-dimension
evaluation framework. The agent reads a customer complaint, classifies it, retrieves the
applicable bank policy from a small policy knowledge base, and produces a recommended
resolution grounded in that policy. The evaluation framework measures how well it does.

## What it demonstrates

- **RAG with a local vector store** — policy documents embedded into an in-process NumPy
  index (no external vector database), retrieved via a Google ADK tool.
- **Structured-output agent** — a single ADK `LlmAgent` that emits a typed result
  (`predicted_category`, `cited_policy_ids`, `resolution`, `reasoning`).
- **Rigorous, layered evaluation** — rules-based graders for everything with a ground-truth
  answer, LLM-judge for quality, and trace-level evaluators for groundedness and efficiency.

## Architecture

```
complaint ─▶ ComplaintResolutionAgent
                ├─ classify        → predicted_category
                ├─ retrieve_policy (tool → policy KB vector store)
                └─ ground resolution in retrieved policy text
             ─▶ { predicted_category, cited_policy_ids, resolution, reasoning }
                └─▶ evaluation harness scores the output
```

Library code lives in `aieng/agent_evals/complaint_resolution/`; this folder holds the
runnable entrypoints, the LLM-judge rubric, and the teaching notebooks.

## Dataset

Kaggle `adhamelkomy/bank-customer-complaint-analysis` (CC0), file `complaints.csv`:
`narrative` (complaint text) + `product` (one of five categories). Downloaded on demand via
`kagglehub`; the raw CSV is not committed. Narratives are pre-processed (lemmatized,
stop-words removed), which the agent prompt and judge rubric account for.

The five categories map one-to-one to policy documents in
`aieng/agent_evals/complaint_resolution/data/policies/`:

| category | policy id |
|---|---|
| `credit_card` | `POL-CREDIT-CARD` |
| `credit_reporting` | `POL-CREDIT-REPORTING` |
| `debt_collection` | `POL-DEBT-COLLECTION` |
| `mortgages_and_loans` | `POL-MORTGAGE-LOANS` |
| `retail_banking` | `POL-RETAIL-BANKING` |

## Evaluation dimensions

| Dimension | Type | Grader |
|---|---|---|
| Classification accuracy | rules | `item_level_deterministic_grader` (`category_correct`) + `run_level_grader` (macro precision/recall/F1, accuracy, confusion matrix) |
| Retrieval quality | rules | `item_level_deterministic_grader` (`policy_match`, `retrieval_precision`, `retrieval_coverage`) |
| Groundedness | LLM judge (trace) | `create_trace_groundedness_evaluator` over the `retrieve_policy` tool output |
| Response quality | LLM judge (item) | `create_llm_as_judge_evaluator` + `rubrics/resolution_quality.md` |
| Efficiency | trace | `create_trace_usage_evaluator` (tool calls, latency, cost) |

## Usage

Requirements: a valid Gemini API key in `.env`. ADK's model path reads `GOOGLE_API_KEY`
(and/or `GEMINI_API_KEY`); the embedding/retrieval path reads `OPENAI_API_KEY`. Setting all
three to the same valid key is the simplest setup.

```bash
# Try the agent on one complaint
uv run --env-file .env complaint-resolution resolve "charged twice credit card duplicate"

# Inspect the knowledge base and dataset
uv run complaint-resolution policies
uv run --env-file .env complaint-resolution sample --per-category 2

# 1) Upload a balanced evaluation set to Langfuse
uv run --env-file .env python implementations/complaint_resolution/data/langfuse_upload.py \
    --dataset-name Complaint-Resolution-Subset --per-category 10

# 2) Run the four-dimension evaluation
uv run --env-file .env python implementations/complaint_resolution/evaluate.py \
    --dataset-name Complaint-Resolution-Subset
```

Run the agent in the ADK web UI:

```bash
uv run adk web --port 8000 --reload --reload_agents implementations/
```

## Notebooks

1. `01_dataset_and_tools.ipynb` — the dataset, the policy KB, and the retrieval tool.
2. `02_running_the_agent.ipynb` — running the agent on complaints end-to-end.
3. `03_evaluation.ipynb` — the four-dimension evaluation and reading the results.
