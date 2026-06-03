"""Policy retrieval tool for the complaint-resolution agent.

Wraps the in-process ``PolicyKnowledgeBase`` vector store as a Google ADK
``FunctionTool`` the agent can call to find the policy document(s) relevant to a
complaint. This mirrors the ``create_google_search_tool`` factory pattern in
:mod:`aieng.agent_evals.tools.search`.

The tool's return dict includes each retrieved document's ``id``, ``title``, and full
``text``. This is deliberate: ADK records the tool return value as the Langfuse
observation output, so the policy text becomes the evidence the groundedness evaluator
checks the resolution against, and the document ids are recoverable from the trace.
"""

import logging
from typing import Any

from google.adk.tools.function_tool import FunctionTool

from .kb import PolicyKnowledgeBase


logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
RETRIEVE_POLICY_TOOL_NAME = "retrieve_policy"


def create_policy_retrieval_tool(kb: PolicyKnowledgeBase, top_k: int = DEFAULT_TOP_K) -> FunctionTool:
    """Create an ADK tool that retrieves policy documents for a complaint.

    Parameters
    ----------
    kb : PolicyKnowledgeBase
        The policy knowledge base to search.
    top_k : int, optional, default=3
        Number of policy documents to return per query.

    Returns
    -------
    FunctionTool
        A tool named ``retrieve_policy`` the agent can call with a query string.

    Examples
    --------
    >>> from aieng.agent_evals.complaint_resolution.kb import PolicyKnowledgeBase
    >>> kb = PolicyKnowledgeBase()
    >>> tool = create_policy_retrieval_tool(kb)
    >>> tool.name
    'retrieve_policy'
    """

    async def retrieve_policy(query: str) -> dict[str, Any]:
        """Retrieve the bank policy documents most relevant to a complaint.

        Use this tool to find the official resolution policy that applies to the
        customer's complaint. Search with the key facts of the complaint (product,
        issue, what went wrong). Ground your resolution ONLY in the returned policy
        text, and cite the ``id`` of each policy you rely on.

        Parameters
        ----------
        query : str
            A query describing the complaint (key product and issue terms).

        Returns
        -------
        dict
            Retrieval results with the following keys:

            - **status** (str): ``"success"`` or ``"error"``.
            - **results** (list[dict]): retrieved documents, each containing:
                - **id** (str): policy document id (e.g. ``"POL-CREDIT-CARD"``).
                - **title** (str): policy title.
                - **text** (str): full policy body to ground the resolution in.
                - **score** (float): similarity score for the match.
            - **result_count** (int): number of documents returned (success only).
            - **error** (str): error message (error case only).
        """
        try:
            docs = await kb.retrieve(query, k=top_k)
            return {
                "status": "success",
                "results": [{"id": doc.id, "title": doc.title, "text": doc.text, "score": doc.score} for doc in docs],
                "result_count": len(docs),
            }
        except Exception as exc:  # noqa: BLE001 - surface a tool error rather than crashing the agent
            logger.error("Policy retrieval failed: %s", exc)
            return {"status": "error", "error": str(exc), "results": [], "result_count": 0}

    return FunctionTool(func=retrieve_policy)
