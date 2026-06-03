"""Task function for complaint-resolution experiment execution.

Provides a Langfuse-compatible task callable that runs the complaint-resolution
agent on one dataset item and returns the parsed structured output as a dict for
the evaluators. Mirrors
:class:`aieng.agent_evals.aml_investigation.task.AmlInvestigationTask`.
"""

import getpass
import json
import logging
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse.experiment import ExperimentItem

from .agent import create_complaint_resolution_agent
from .data import ComplaintResolutionOutput
from .kb import PolicyKnowledgeBase


logger = logging.getLogger(__name__)

_APP_NAME = "complaint_resolution"


class ComplaintResolutionTask:
    """Langfuse-compatible task wrapper for complaint resolution.

    Implements the ``TaskFunction`` callable protocol expected by Langfuse
    experiments: ``__call__(*, item, **kwargs)``. Each instance owns one agent, one
    knowledge base, and one ADK runner.

    Parameters
    ----------
    agent : LlmAgent | None, optional
        Pre-configured agent. If ``None``, one is built via
        :func:`create_complaint_resolution_agent` using ``kb``.
    kb : PolicyKnowledgeBase | None, optional
        Knowledge base backing retrieval. If ``None``, a default one is created.

    Examples
    --------
    >>> import asyncio
    >>> task = ComplaintResolutionTask()
    >>> result = asyncio.run(task(item={"input": "charged twice on my credit card"}))
    >>> sorted(result)  # doctest: +SKIP
    ['cited_policy_ids', 'predicted_category', 'reasoning', 'resolution']
    """

    def __init__(self, *, agent: LlmAgent | None = None, kb: PolicyKnowledgeBase | None = None) -> None:
        self._kb = kb or PolicyKnowledgeBase()
        self._agent = agent or create_complaint_resolution_agent(kb=self._kb)
        self._runner = Runner(
            app_name=_APP_NAME,
            agent=self._agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

    @staticmethod
    def _extract_input_text(item: ExperimentItem) -> str:
        """Extract the complaint narrative text from a dict-like or dataset item."""
        item_input = item.get("input") if isinstance(item, dict) else item.input
        if isinstance(item_input, str):
            return item_input
        # Defensive: if the input is structured, serialize it to text.
        return json.dumps(item_input, ensure_ascii=False)

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """Run the agent on one complaint and return parsed structured output.

        Parameters
        ----------
        item : ExperimentItem
            One Langfuse experiment item whose ``input`` is the complaint narrative.
        **kwargs : Any
            Additional keyword arguments forwarded by Langfuse (ignored).

        Returns
        -------
        dict[str, Any] | None
            Parsed :class:`ComplaintResolutionOutput` as a dict, or ``None`` if no
            valid final response was produced.
        """
        narrative = self._extract_input_text(item)
        message = types.Content(parts=[types.Part(text=narrative)], role="user")

        final_text: str | None = None
        async for event in self._runner.run_async(
            session_id=str(uuid.uuid4()), user_id=getpass.getuser(), new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(part.text or "" for part in event.content.parts if part.text)

        if not final_text:
            logger.warning("No resolution output produced for item.")
            return None

        try:
            return ComplaintResolutionOutput.model_validate_json(final_text.strip()).model_dump()
        except Exception:
            return ComplaintResolutionOutput.model_validate(json.loads(final_text)).model_dump()

    async def close(self) -> None:
        """Close the runner and knowledge-base resources used by this task."""
        await self._runner.close()
        await self._kb.aclose()
