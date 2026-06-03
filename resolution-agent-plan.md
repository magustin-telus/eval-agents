# Plan: Complaint-Resolution RAG Agent + Evaluation Framework

## Context

Bootcamp use case for Vector Institute's **Agentic AI Evaluation Bootcamp**. We are adding a
fifth-style reference module to the `eval-agents` repo: a **RAG-based customer complaint-resolution
agent** with a rigorous evaluation framework. The agent reads a customer complaint narrative,
classifies it, retrieves the relevant resolution/policy guidance from a knowledge base, and produces
a grounded recommended resolution. The bootcamp's value is the *evaluation*, so the framework measures
four dimensions: classification accuracy, groundedness (no hallucination), response quality, and
retrieval quality + efficiency.

This mirrors the existing **`knowledge_qa`** module (document-grounded Q&A) and reuses the
**`aml_investigation`** patterns for structured output + classification graders. The dataset is the
Kaggle "adhamelkomy/bank-customer-complaint-analysis" set (CC0, approved), loaded via `kagglehub` the
same way `knowledge_qa` loads DeepSearchQA.

### Confirmed design decisions
1. **Knowledge base** = a small, hand-authored **policy KB** (one markdown doc per complaint category).
2. **Retrieval** = a **lightweight in-process vector store** (numpy cosine similarity) built from the
   already-staged embedding config (`Configs.embedding_base_url` / `embedding_api_key` /
   `embedding_model_name="@cf/baai/bge-m3"`). No Vertex, no Weaviate server.
3. **Scope** = full parity with `knowledge_qa`: library subpackage + `implementations/` runner + 3 notebooks.
4. **Eval dimensions** = all four, reusing existing graders.

### Key fact that de-risks the design
The AML agent already combines `tools=[FunctionTool(...)]` **with** `output_schema=AnalystOutput` in a
single `LlmAgent` ([aml_investigation/agent.py:241-263](aieng-eval-agents/aieng/agent_evals/aml_investigation/agent.py#L241-L263)).
So our agent can call the retrieval tool **and** emit parseable structured JSON the graders read directly.

---

## File tree to create

### Library subpackage `aieng-eval-agents/aieng/agent_evals/complaint_resolution/`
| File | Purpose |
|---|---|
| `__init__.py` | Re-export agent factory, `ComplaintResolutionOutput`, dataset class, graders. |
| `agent.py` | `create_complaint_resolution_agent()` — ADK `LlmAgent` with retrieval tool + `output_schema`. Mirror [aml_investigation/agent.py](aieng-eval-agents/aieng/agent_evals/aml_investigation/agent.py). |
| `system_instructions.py` | Prompt: classify into fixed category set → call `retrieve_policy` → ground resolution in retrieved docs → emit structured JSON. |
| `embeddings.py` | `embed_texts(texts) -> np.ndarray` — async call to OpenAI-compatible `/embeddings` via `embedding_base_url`. |
| `kb.py` | `PolicyKnowledgeBase`: load policy markdown, embed, hold numpy index, `retrieve(query, k)`; `CATEGORY_TO_POLICY_ID`, `CATEGORY_LABELS`. |
| `retrieval_tool.py` | `create_policy_retrieval_tool(kb) -> FunctionTool`. Mirror [tools/search.py::create_google_search_tool](aieng-eval-agents/aieng/agent_evals/tools/search.py#L310). |
| `data/__init__.py` | Re-export dataset, Example, `ComplaintResolutionOutput`, category constants. |
| `data/bank_complaints.py` | `BankComplaintsDataset` + `BankComplaintExample` (mirror [knowledge_qa/data/deepsearchqa.py](aieng-eval-agents/aieng/agent_evals/knowledge_qa/data/deepsearchqa.py)). Defines `ComplaintResolutionOutput`. |
| `data/policies/*.md` | One policy doc per canonical category (~5–8 docs). |
| `graders/__init__.py` | Re-export the grader functions. |
| `graders/_common.py` | `get_field`, `normalize_category`, `normalize_doc_ids` (mirror AML `_common.py`). |
| `graders/item.py` | `item_level_deterministic_grader`: `category_correct` + retrieval `id_precision`/`id_coverage`. Mirror [aml_investigation/graders/item.py](aieng-eval-agents/aieng/agent_evals/aml_investigation/graders/item.py). |
| `graders/run.py` | `run_level_grader`: sklearn precision/recall/F1, macro-F1, confusion matrix over categories. |

### Runner `implementations/complaint_resolution/`
| File | Purpose |
|---|---|
| `agent.py` | `root_agent = create_complaint_resolution_agent()` for `adk web`. Mirror [implementations/knowledge_qa/agent.py](implementations/knowledge_qa/agent.py). |
| `evaluate.py` | click CLI wiring the four eval dimensions. Mirror [implementations/knowledge_qa/evaluate.py](implementations/knowledge_qa/evaluate.py) + [implementations/aml_investigation/evaluate.py](implementations/aml_investigation/evaluate.py). |
| `cli.py` | Optional single-complaint interactive runner with `main()`. |
| `data/langfuse_upload.py` | Build JSONL records from dataset → `upload_dataset_to_langfuse`. |
| `rubrics/resolution_quality.md` | LLM-judge rubric. Model on [implementations/aml_investigation/rubrics/narrative_pattern_quality.md](implementations/aml_investigation/rubrics/narrative_pattern_quality.md). |
| `01_dataset_and_tools.ipynb`, `02_running_the_agent.ipynb`, `03_evaluation.ipynb` | Teaching notebooks. |
| `README.md` | Module docs. |

---

## Component designs

### Embedding client + vector store (`embeddings.py`, `kb.py`)
- **Client**: read `embedding_base_url` / `embedding_api_key` / `embedding_model_name` from `Configs`. Build a
  dedicated `AsyncOpenAI(base_url=..., api_key=...)` (do **not** reuse the chat client) and call
  `client.embeddings.create(model=..., input=texts)`; return `np.float32` array. bge-m3 → 1024-dim. Add a small
  `tenacity` retry like the rest of the repo.
- **Store**: in-process numpy. KB is tiny (<20 docs). Normalize rows, cosine sim = `mat @ q`. Cache to a `.npz`
  (vectors + parallel doc-id list) keyed by a hash of doc contents; rebuild on miss/change. `retrieve(query, k)`
  returns `PolicyDoc(id, title, text, score)`, `k` small (1–3).

### Retrieval FunctionTool (`retrieval_tool.py`)
```python
async def retrieve_policy(query: str) -> dict:
    """Returns {"status", "results": [{"id","title","text","score"}], "result_count"}"""
```
Returning per-doc `id`/`title`/`text` is **load-bearing**: ADK records the tool return as the Langfuse
observation `output`, giving the groundedness judge the policy text as context. Wrap with `FunctionTool(func=retrieve_policy)`.

### Agent factory + structured output (`agent.py`, `data/bank_complaints.py`)
`ComplaintResolutionOutput(BaseModel)`:
- `predicted_category: str` — classification prediction (prompt-constrained to `CATEGORY_LABELS`; normalized in grader).
- `cited_policy_ids: str` — comma-separated policy doc ids the resolution is grounded in (AML string convention).
- `resolution: str` — customer-facing resolution text.
- `reasoning: str` — short justification.

`create_complaint_resolution_agent()` mirrors `create_aml_investigation_agent`: `AsyncClientManager.get_instance()`,
`init_tracing(service_name=name)`, build `LlmAgent(model=configs.default_planner_model,
instruction=SYSTEM_INSTRUCTIONS, tools=[retrieval_tool], output_schema=ComplaintResolutionOutput,
generate_content_config=GenerateContentConfig(thinking_config=ThinkingConfig(include_thoughts=True), temperature=...))`.
The eval task wrapper parses the final text with `ComplaintResolutionOutput.model_validate_json(...)` (json fallback)
and returns `.model_dump()` so graders read fields as a dict.

### Dataset + Langfuse (`data/bank_complaints.py`, `data/langfuse_upload.py`)
- **VERIFIED schema** (downloaded & inspected `adhamelkomy/bank-customer-complaint-analysis`, version 1):
  use the file **`complaints.csv`** (162,421 rows). Columns: `product` (category label), `narrative` (complaint text),
  and an ignorable `Unnamed: 0` index. The other artifacts in the archive (`final_dataframe (1).csv` = Id/Target,
  a notebook, a `.txt` report) are not used.
- **5 clean product classes** (counts): `credit_reporting` (91,179), `debt_collection` (23,150),
  `mortgages_and_loans` (18,990), `credit_card` (15,566), `retail_banking` (13,536). These ARE the canonical
  `CATEGORY_LABELS` — no taxonomy collapsing needed. `CATEGORY_TO_POLICY_ID` maps each of the 5 → one policy doc
  (so the KB is exactly 5 docs).
- `BankComplaintsDataset` mirrors `DeepSearchQADataset`: `kagglehub.dataset_download(...)`, `pd.read_csv(.../complaints.csv)`,
  filter to the `narrative`/`product` columns. Build `BankComplaintExample(example_id, narrative, category)`; provide
  `get_by_category`, `sample`, `get_categories` for parity. 162k rows is far too many — the Langfuse upload takes a
  **balanced sample** (e.g. N per category) so classes aren't dominated by `credit_reporting`.
- **DECIDED — narratives are pre-processed (lemmatized, stop-words removed, no punctuation)**, e.g.
  *"purchase order day shipping amount receive product week sent followup email..."*. We feed them to the agent
  **as-is** (no rewrite step). Classification & retrieval are unaffected (topic signal intact). The
  `resolution_quality.md` rubric and the `03_evaluation.ipynb` notebook will explicitly note that the input is
  preprocessed, so the resolution-quality judge grades within that constraint (e.g. judge structure/grounding/tone,
  not verbatim fluency tied to the garbled input).
- `langfuse_upload.py` emits per example:
  `{"input": narrative, "expected_output": {"category": cat, "gold_policy_id": CATEGORY_TO_POLICY_ID[cat]},
  "metadata": {"example_id":..., "category": cat}}` → temp JSONL → `await upload_dataset_to_langfuse(path, name)`.

### Four-dimension eval wiring (`evaluate.py`)
Use `run_experiment_with_trace_evals` (like AML):
1. **Classification accuracy** — `item_level_deterministic_grader` (item-level) emits `category_correct`;
   `run_level_grader` (run-level) computes precision/recall/F1, macro-F1, confusion matrix via sklearn over
   `predicted_category` vs `expected_output["category"]` (use an `INVALID` bucket for unparseable categories, like AML).
2. **Retrieval quality** — same item grader adds `retrieval_precision`/`retrieval_coverage` comparing normalized
   `cited_policy_ids` vs `{gold_policy_id}` (reuse AML id_precision/coverage math). Read predicted ids from
   `output.cited_policy_ids` (robust, no trace dependency).
3. **Groundedness** — `create_trace_groundedness_evaluator(tool_observation_predicate=<matches name "retrieve_policy">,
   model_config=...)` in `trace_evaluators=`.
4. **Response quality** — `create_llm_as_judge_evaluator(name="resolution_quality",
   rubric_markdown="implementations/complaint_resolution/rubrics/resolution_quality.md", model_config=...)`.
5. **Efficiency** — `create_trace_usage_evaluator(name="trace_usage",
   metrics={"tool_call_count": True, "latency_sec": True, "total_cost": True})` (import from
   `misalignment_qa.evaluation.hard_metrics`).

### Policy KB (`data/policies/`)
One markdown doc per canonical category (~5–8). Parse stable front-matter into `PolicyDoc`:
```
# id: POL-CREDIT-CARD
# title: Credit Card Complaint Resolution Policy
# category: credit_card
<body: eligibility, investigation steps, remediation/timeframes, escalation>
```
`CATEGORY_TO_POLICY_ID` maps each dataset category → one `id`. The category label is both classification ground
truth and the single gold retrieval target per example. Keep docs short/self-contained for crisp groundedness judging.

### pyproject + deps
- Add to `[project.scripts]` in [aieng-eval-agents/pyproject.toml](aieng-eval-agents/pyproject.toml#L35) (only if `cli.py` ships):
  `complaint-resolution = "aieng.agent_evals.complaint_resolution.cli:main"`.
- **No new dependencies** — `numpy`, `kagglehub`, `scikit-learn`, `langfuse`, `openai` already present.

---

## Remaining implementation-time risks (verify, don't re-decide)
1. **Dataset column names + category cardinality** — confirm CSV columns; collapse product taxonomy into canonical
   `CATEGORY_LABELS` so each maps to exactly one policy doc and sklearn labels stay stable.
2. **Embedding endpoint unproven** — no existing code calls it. Validate response shape (`resp.data[i].embedding`)
   and dimensionality before building the index; `.npz` cache means the endpoint isn't hit every run. If the
   bootcamp env has no embedding endpoint configured, fall back to a local sentence-transformers / hashing embedder
   (decide with user only if endpoint is unavailable).
3. **Groundedness predicate** — confirm the retrieval observation name is `retrieve_policy` in a real trace so the
   custom `tool_observation_predicate` matches.
4. **Category normalization** — `predicted_category` must normalize to the exact `CATEGORY_LABELS` set;
   reuse an AML-style `normalize_category` + `INVALID` bucket.

---

## Verification (end-to-end, with `uv run`)
1. **Embedding + retrieval**:
   `uv run --env-file .env python -c "import asyncio; from aieng.agent_evals.complaint_resolution.kb import PolicyKnowledgeBase as K; print(asyncio.run(K().retrieve('charged twice on my credit card', k=2)))"`
2. **Dataset load** (verify columns):
   `uv run python -c "from aieng.agent_evals.complaint_resolution.data import BankComplaintsDataset as D; d=D(); print(len(d), d.get_categories(), d[0])"`
3. **Single complaint** through the agent → assert output parses to `ComplaintResolutionOutput` with non-empty
   `predicted_category` and `cited_policy_ids` (via `cli.py` or a one-liner running the task).
4. **Small eval**:
   `uv run --env-file .env python implementations/complaint_resolution/data/langfuse_upload.py --samples 5`
   then `uv run --env-file .env python implementations/complaint_resolution/evaluate.py --dataset-name Complaint-Resolution-Smoke`
   — confirm item/run/trace metrics compute and upload to Langfuse.
5. **Quality gates**: `uv run ruff format . && uv run ruff check . && uv run mypy aieng-eval-agents/aieng` and
   `uv run pytest -m "not integration_test"`.
