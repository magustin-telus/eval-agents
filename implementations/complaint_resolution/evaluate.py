"""Evaluate the complaint-resolution agent across four dimensions.

Runs the complaint-resolution agent against a Langfuse dataset and scores it with:

1. Classification accuracy — deterministic item + run graders (rules).
2. Retrieval quality — deterministic item grader (policy match, precision/coverage).
3. Groundedness — trace-level LLM judge over the retrieved policy text.
4. Response quality — item-level LLM judge against a rubric.

Efficiency metrics (tool calls, latency, cost) are recorded as trace usage metrics.

Upload a dataset first with ``data/langfuse_upload.py``, then run this against that
dataset name. Mirrors ``implementations/aml_investigation/evaluate.py``.

Example
-------
$ uv run --env-file .env python implementations/complaint_resolution/evaluate.py \
    --dataset-name Complaint-Resolution-Subset
"""

import logging

import click
from aieng.agent_evals.complaint_resolution.agent import create_complaint_resolution_agent
from aieng.agent_evals.complaint_resolution.graders import (
    item_level_deterministic_grader,
    run_level_grader,
)
from aieng.agent_evals.complaint_resolution.retrieval_tool import RETRIEVE_POLICY_TOOL_NAME
from aieng.agent_evals.complaint_resolution.task import ComplaintResolutionTask
from aieng.agent_evals.display import create_console, display_info, display_metrics_table
from aieng.agent_evals.evaluation import TraceWaitConfig
from aieng.agent_evals.evaluation.experiment import run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import (
    create_llm_as_judge_evaluator,
    create_trace_groundedness_evaluator,
)
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.misalignment_qa.evaluation.hard_metrics import create_trace_usage_evaluator
from rich.logging import RichHandler


logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)], force=True)
logging.getLogger("google_adk").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_RUBRIC_PATH = "implementations/complaint_resolution/rubrics/resolution_quality.md"


def _retrieval_observation_predicate(observation: object) -> bool:
    """Select the ``retrieve_policy`` tool observations as groundedness evidence.

    The groundedness judge will be shown only these observations' outputs (the
    retrieved policy text), so it checks the resolution against the policy the agent
    actually retrieved.
    """
    name = (getattr(observation, "name", "") or "").strip().lower()
    return RETRIEVE_POLICY_TOOL_NAME in name


@click.command()
@click.option("--dataset-name", type=str, required=True, help="Name of the (already-uploaded) Langfuse dataset.")
@click.option("--agent-timeout", type=click.IntRange(min=1), default=300, help="Agent timeout in seconds.")
@click.option("--llm-judge-timeout", type=click.IntRange(min=1), default=120, help="LLM judge timeout in seconds.")
@click.option("--llm-judge-retries", type=click.IntRange(min=0), default=3, help="LLM judge retry attempts.")
@click.option("--max-concurrency", type=click.IntRange(min=1, max=10), default=3, help="Max concurrent agent runs.")
@click.option("--max-trace-concurrency", type=click.IntRange(min=1, max=10), default=5, help="Max concurrent traces.")
@click.option("--max-trace-wait-time", type=click.IntRange(min=1), default=300, help="Max seconds to wait for traces.")
def cli(
    dataset_name: str,
    agent_timeout: int,
    llm_judge_timeout: int,
    llm_judge_retries: int,
    max_concurrency: int,
    max_trace_concurrency: int,
    max_trace_wait_time: int,
) -> None:
    """Evaluate the complaint-resolution agent on a Langfuse dataset."""
    console = create_console(force_jupyter=False)
    judge_config = LLMRequestConfig(timeout_sec=llm_judge_timeout, retry_max_attempts=llm_judge_retries)

    # Item-level LLM judge: resolution quality against the rubric.
    resolution_quality_evaluator = create_llm_as_judge_evaluator(
        name="resolution_quality",
        rubric_markdown=_RUBRIC_PATH,
        model_config=judge_config,
    )

    # Trace-level: groundedness over retrieved policy text + efficiency usage metrics.
    groundedness_evaluator = create_trace_groundedness_evaluator(
        name="groundedness",
        tool_observation_predicate=_retrieval_observation_predicate,
        model_config=judge_config,
    )
    usage_evaluator = create_trace_usage_evaluator(
        name="trace_usage",
        metrics={"tool_call_count": True, "latency_sec": True, "total_cost": True},
    )

    agent = create_complaint_resolution_agent(timeout_sec=agent_timeout)
    results = run_experiment_with_trace_evals(
        dataset_name=dataset_name,
        name="Complaint Resolution Evaluation",
        description="RAG complaint-resolution agent: classification, retrieval, groundedness, quality.",
        task=ComplaintResolutionTask(agent=agent),
        evaluators=[item_level_deterministic_grader, resolution_quality_evaluator],
        trace_evaluators=[groundedness_evaluator, usage_evaluator],
        run_evaluators=[run_level_grader],
        max_concurrency=max_concurrency,
        trace_max_concurrency=max_trace_concurrency,
        trace_wait=TraceWaitConfig(max_wait_sec=max_trace_wait_time),
    )

    # Item-level results
    console.print("\n[bold cyan]📋 Item-Level Results[/bold cyan]\n")
    for idx, item_result in enumerate(results.experiment.item_results, start=1):
        item_metrics = {eval_.name: eval_.value for eval_ in item_result.evaluations}
        display_metrics_table(metrics=item_metrics, title=f"Item {idx}", console=console)

    # Run-level metrics
    if getattr(results.experiment, "run_evaluations", None):
        console.print("\n[bold green]📊 Run-Level Metrics[/bold green]\n")
        run_metrics = {eval_.name: eval_.value for eval_ in results.experiment.run_evaluations}
        display_metrics_table(metrics=run_metrics, title="Aggregate Classification", console=console)

    # Trace evaluation summary
    if results.trace_evaluations:
        console.print("\n[bold magenta]🔍 Trace Evaluation Summary[/bold magenta]\n")
        trace_summary: dict[str, float | int | str] = {
            "Successful Traces": len(results.trace_evaluations.evaluations_by_trace_id),
            "Skipped Traces": len(results.trace_evaluations.skipped_trace_ids),
            "Failed Traces": len(results.trace_evaluations.failed_trace_ids),
        }
        display_metrics_table(metrics=trace_summary, title="Trace Processing", console=console)

    display_info("Evaluation complete! Results have been uploaded to Langfuse.", console=console)


if __name__ == "__main__":
    cli()
