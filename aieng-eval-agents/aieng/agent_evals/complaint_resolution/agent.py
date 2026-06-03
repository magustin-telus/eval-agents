"""Complaint-resolution agent factory.

Builds a Google ADK ``LlmAgent`` that reads a customer complaint, classifies it,
retrieves the applicable bank policy via the ``retrieve_policy`` tool, and returns a
grounded recommended resolution conforming to
:class:`~aieng.agent_evals.complaint_resolution.data.ComplaintResolutionOutput`.

This mirrors the AML investigation agent factory, which likewise combines tool use
with a strict ``output_schema``.

Examples
--------
>>> from aieng.agent_evals.complaint_resolution import create_complaint_resolution_agent
>>> agent = create_complaint_resolution_agent()
>>> agent.name
'ComplaintResolutionAnalyst'
"""

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.langfuse import init_tracing
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig

from .data import ComplaintResolutionOutput
from .kb import PolicyKnowledgeBase
from .retrieval_tool import DEFAULT_TOP_K, create_policy_retrieval_tool
from .system_instructions import build_system_instructions


_DEFAULT_AGENT_DESCRIPTION = (
    "Classifies customer complaints and recommends resolutions grounded in retrieved bank policy."
)


def create_complaint_resolution_agent(
    name: str = "ComplaintResolutionAnalyst",
    *,
    kb: PolicyKnowledgeBase | None = None,
    top_k: int = DEFAULT_TOP_K,
    description: str | None = None,
    instructions: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    seed: int | None = None,
    timeout_sec: int | None = None,
    enable_tracing: bool = True,
) -> LlmAgent:
    """Create a configured complaint-resolution agent.

    Parameters
    ----------
    name : str, default="ComplaintResolutionAnalyst"
        Name assigned to the agent (appears in traces and logs).
    kb : PolicyKnowledgeBase | None, optional
        Policy knowledge base backing the retrieval tool. If ``None``, a default
        ``PolicyKnowledgeBase()`` is created.
    top_k : int, optional, default=3
        Number of policy documents the retrieval tool returns per query.
    description : str | None, optional
        Short description of the agent's purpose. Defaults to a standard description.
    instructions : str | None, optional
        System prompt. If omitted, the module's standard instructions are used.
    temperature : float | None, optional
        Sampling temperature. ``None`` uses provider/model defaults.
    max_output_tokens : int | None, optional
        Maximum tokens the model may generate in a single response.
    seed : int | None, optional
        Optional random seed for more repeatable generations where supported.
    timeout_sec : int | None, optional
        Optional per-call timeout in seconds.
    enable_tracing : bool, optional, default=True
        Whether to initialize Langfuse tracing (service name = the agent's name).

    Returns
    -------
    LlmAgent
        Configured agent with the policy retrieval tool and a strict
        ``ComplaintResolutionOutput`` response schema.
    """
    client_manager = AsyncClientManager.get_instance()

    if enable_tracing:
        init_tracing(service_name=name)

    kb = kb or PolicyKnowledgeBase()
    retrieval_tool = create_policy_retrieval_tool(kb, top_k=top_k)

    return LlmAgent(
        name=name,
        description=description or _DEFAULT_AGENT_DESCRIPTION,
        model=client_manager.configs.default_planner_model,
        instruction=instructions or build_system_instructions(),
        tools=[retrieval_tool],
        generate_content_config=GenerateContentConfig(
            http_options=HttpOptions(timeout=timeout_sec * 1000) if timeout_sec is not None else None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed,
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
        output_schema=ComplaintResolutionOutput,
    )
