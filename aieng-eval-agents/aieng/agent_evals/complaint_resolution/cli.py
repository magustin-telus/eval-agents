"""Command-line interface for the complaint-resolution agent.

Provides a few convenience commands for trying the agent and inspecting its inputs:

- ``resolve`` — run the agent on a single complaint (text or a dataset example).
- ``sample``  — print a balanced sample of complaints from the dataset.
- ``policies``— list the policy knowledge-base documents.

Examples
--------
    uv run --env-file .env complaint-resolution resolve "charged twice credit card"
    uv run complaint-resolution policies
    uv run complaint-resolution sample --per-category 2
"""

import asyncio
import json
import logging

import click
from dotenv import load_dotenv

from .data import BankComplaintsDataset
from .kb import PolicyKnowledgeBase
from .task import ComplaintResolutionTask


load_dotenv(verbose=False)
logging.basicConfig(level=logging.WARNING, format="%(message)s")


@click.group()
def main() -> None:
    """Complaint-resolution agent CLI."""


@main.command()
@click.argument("complaint", required=False)
@click.option("--example-id", type=int, default=None, help="Run on a dataset example by id instead of text.")
def resolve(complaint: str | None, example_id: int | None) -> None:
    """Run the agent on a single COMPLAINT and print the structured resolution."""
    if example_id is not None:
        example = BankComplaintsDataset().get_by_id(example_id)
        if example is None:
            raise click.ClickException(f"No example with id {example_id}")
        complaint = example.narrative
        click.echo(f"[dataset example {example_id} | true category: {example.category}]\n")
    if not complaint:
        raise click.ClickException("Provide a COMPLAINT argument or --example-id.")

    async def _run() -> dict | None:
        task = ComplaintResolutionTask()
        try:
            return await task(item={"input": complaint})
        finally:
            await task.close()

    result = asyncio.run(_run())
    if result is None:
        raise click.ClickException("Agent produced no output.")
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@main.command()
@click.option("--per-category", type=int, default=2, help="Examples per category to show.")
@click.option("--random-state", type=int, default=42, help="Random seed.")
def sample(per_category: int, random_state: int) -> None:
    """Print a balanced sample of complaints from the dataset."""
    dataset = BankComplaintsDataset()
    for ex in dataset.sample_balanced(n_per_category=per_category, random_state=random_state):
        click.echo(f"[{ex.category}] (id={ex.example_id}) {ex.narrative[:120]}...")


@main.command()
def policies() -> None:
    """List the policy knowledge-base documents."""
    kb = PolicyKnowledgeBase()
    for doc in kb.docs:
        click.echo(f"{doc.id:22s} {doc.category:20s} {doc.title}")


if __name__ == "__main__":
    main()
