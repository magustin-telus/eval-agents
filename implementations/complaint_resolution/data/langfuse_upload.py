"""Upload a balanced sample of bank complaints to Langfuse.

This script selects a class-balanced sample of complaints from the bank customer
complaints dataset, converts each to a Langfuse dataset record (input = complaint
narrative; expected_output = category + gold policy id), and uploads them using the
shared upload utility. Mirrors ``implementations/knowledge_qa/data/langfuse_upload.py``.

Usage (from the repo root):
    cd implementations/complaint_resolution/data
    uv run --env-file ../../../.env python langfuse_upload.py --per-category 10
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import click
from aieng.agent_evals.complaint_resolution.data import BankComplaintsDataset
from aieng.agent_evals.complaint_resolution.kb import CATEGORY_TO_POLICY_ID
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "Complaint-Resolution-Subset"


async def upload_complaints_to_langfuse(
    dataset_name: str,
    per_category: int = 10,
    random_state: int = 42,
) -> None:
    """Upload a balanced sample of complaints to Langfuse.

    Parameters
    ----------
    dataset_name : str
        Name for the dataset in Langfuse.
    per_category : int, optional, default=10
        Number of examples to sample per complaint category.
    random_state : int, optional, default=42
        Random seed for reproducible sampling.
    """
    logger.info("Loading bank complaints dataset...")
    dataset = BankComplaintsDataset()
    examples = dataset.sample_balanced(n_per_category=per_category, random_state=random_state)
    logger.info(f"Selected {len(examples)} balanced examples across {len(dataset.get_categories())} categories")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix=f"complaints_{dataset_name}_",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        for example in examples:
            gold_policy_id = CATEGORY_TO_POLICY_ID.get(example.category, "")
            record = {
                "input": example.narrative,
                "expected_output": {
                    "category": example.category,
                    "gold_policy_id": gold_policy_id,
                },
                "metadata": {
                    "example_id": example.example_id,
                    "category": example.category,
                },
            }
            temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        await upload_dataset_to_langfuse(dataset_path=str(temp_path), dataset_name=dataset_name)
    finally:
        if temp_path.exists():
            temp_path.unlink()
            logger.debug(f"Removed temporary file: {temp_path}")


@click.command()
@click.option("--dataset-name", default=DEFAULT_DATASET_NAME, help="Name for the dataset in Langfuse.")
@click.option("--per-category", default=10, type=int, help="Number of examples per category (default: 10).")
@click.option("--random-state", default=42, type=int, help="Random seed for sampling (default: 42).")
def cli(dataset_name: str, per_category: int, random_state: int) -> None:
    """Upload a balanced sample of bank complaints to Langfuse."""
    asyncio.run(upload_complaints_to_langfuse(dataset_name, per_category, random_state))


if __name__ == "__main__":
    cli()
