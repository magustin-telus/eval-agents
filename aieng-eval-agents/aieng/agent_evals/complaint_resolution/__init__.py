"""Complaint-resolution RAG agent and evaluation module.

A reference implementation of a retrieval-augmented (RAG) customer complaint
resolution agent plus a four-dimension evaluation framework (classification
accuracy, retrieval quality, groundedness, and response quality).
"""

from .agent import create_complaint_resolution_agent
from .data import (
    BankComplaintExample,
    BankComplaintsDataset,
    ComplaintResolutionOutput,
)
from .embeddings import EmbeddingClient
from .graders import item_level_deterministic_grader, run_level_grader
from .kb import (
    CATEGORY_LABELS,
    CATEGORY_TO_POLICY_ID,
    PolicyDoc,
    PolicyKnowledgeBase,
)
from .retrieval_tool import create_policy_retrieval_tool
from .task import ComplaintResolutionTask


__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_TO_POLICY_ID",
    "BankComplaintExample",
    "BankComplaintsDataset",
    "ComplaintResolutionOutput",
    "ComplaintResolutionTask",
    "EmbeddingClient",
    "PolicyDoc",
    "PolicyKnowledgeBase",
    "create_complaint_resolution_agent",
    "create_policy_retrieval_tool",
    "item_level_deterministic_grader",
    "run_level_grader",
]
