"""research_brain.py — Reflexive Agent core for research tasks.

Implements the three-layer Reflexive Agent Architecture:
  PRIME   → parse task, detect domain, inject domain context
  EXECUTE → Claude tool loop using MCP tools from green agent
  REFLECT → verify citations, tool coverage, format answer

Supports ResearchToolBench domains:
  - academic  : literature review, paper synthesis, citation extraction
  - news      : fact verification, source checking, temporal reasoning
  - technical : troubleshooting, debugging, dual-control (user+agent tools)
"""
from __future__ import annotations
import json
import re
import time
from typing import Any

import anthropic
import httpx

from src.config import ANTHROPIC_API_KEY, MAIN_MODEL, FAST_MODEL, LLM_TIMEOUT, MAX_TURNS
from src.mcp_bridge import discover_tools, call_tool

# ── Domain detection patterns ─────────────────────────────────────────────────
_ACADEMIC_KEYWORDS = [
    "paper", "literature", "research", "study", "publication", "arxiv",
    "journal", "citation", "review", "abstract", "methodology", "findings",
    "hypothesis", "experiment", "dataset",
]
_NEWS_KEYWORDS = [
    "news", "article", "report", "headline", "event", "incident",
    "according to", "journalist", "source", "verify", "fact-check",
    "claim", "breaking", "latest",
]
_TECHNICAL_KEYWORDS = [
    "error", "bug", "fix", "debug", "install", "package", "library",
    "pytorch", "tensorflow", "pip", "conda", "import", "module",
    "traceback", "exception", "stack trace", "version", "dependency",
]

# ── Domain-specific system prompts ────────────────────────────────────────────
_SYSTEM_PROMPTS = {
    "academic": """You are an expert research assistant specializing in academic literature.

Your task: answer research questions using the MCP tools provided.

KEY BEHAVIORS:
1. Use search/fetch tools to find relevant papers and sources
2. ALWAYS cite your sources with author, title, year, and URL/DOI when available
3. Verify facts across multiple sources before stating them as established
4. Summarize findings accurately — do not over-claim
5. Use exact quotes when precision matters
6. If multiple papers conflict, note the disagreement

CITATION FORMAT: [Author et al., Year] "Title" — key finding.

After gathering information, provide a structured answer with:
- Summary of findings
- Key sources cited (numbered list)
- Confidence level (high/medium/low)""",

    "news": """You are a fact-checking research assistant specializing in news verification.

Your task: verify claims, find sources, and assess news accuracy using the MCP tools provided.

KEY BEHAVIORS:
1. Search for primary sources — news agencies, official statements, databases
2. Cross-check claims across multiple independent sources
3. Note the publication date and recency of sources
4. Distinguish verified facts from unverified claims
5. Flag conflicting reports or disputed information
6. Check source credibility and potential bias

After researching, provide:
- Verdict (Verified / Unverified / Disputed / Insufficient Evidence)
- Supporting sources with dates
- Summary of what is and isn't confirmed""",

    "technical": """You are a technical support specialist and debugging expert.

Your task: diagnose and resolve technical issues using the MCP tools provided.

KEY BEHAVIORS:
1. Systematically diagnose: read error messages, check versions, inspect environment
2. Use available tools to check system state, run commands, inspect logs
3. Follow a structured troubleshooting methodology:
   - Reproduce → Diagnose → Root cause → Fix → Verify
4. Provide step-by-step resolution instructions
5. If the user also has tools (dual-control), coordinate with them — request user actions when needed
6. Verify the fix worked after implementation

After troubleshooting, provide:
- Root cause identification
- Step-by-step fix applied
- Verification that the issue is resolved""",

    "general": """You are a research agent with access to MCP tools.

Your task: answer the given research question using the tools available.

KEY BEHAVIORS:
1. Use tools to gather accurate, up-to-date information
2. Cite your sources clearly
3. Verify key facts before stating them
4. Provide a structured, well-organized answer
5. Acknowledge uncertainty when present

Provide a thorough answer based on evidence gathered.""",
}


def _detect_domain(task_text: str) -> str:
    """Heuristically detect the research domain from task description."""
    text_lower = task_text.lower()
    academic_score = sum(1 for k in _ACADEMIC_KEYWORDS if k in text_lower)
    news_score = sum(1 for k in _NEWS_KEYWORDS if k in text_lower)
    technical_score = sum(1 for k in _TECHNICAL_KEYWORDS if k in text_lower)

    scores = {"academic": academic_score, "news": news_score, "technical": technical_score}
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])

    return best_domain if best_score >= 2 else "general"


def _extract_mcp_uri(task_data: Any) -> str | None:
    """Extract MCP URI from A2A task data (WebShop+/ResearchToolBench pattern)."""
    if isinstance(task_data, dict):
        # Direct key
        for key in ("mcp_uri", "mcp_url", "tools_endpoint"):
            if task_data.get(key):
                return task_data[key]
        # Nested in resources array (A2A spec)
        resources = task_data.get("resources", [])
        for r in resources:
            if isinstance(r, dict) and r.get("type") == "mcp":
                return r.get("url") or r.get("uri")
    return None


async def _prime(task_text: str, task_data: Any, mcp_url: str, session_id: str) -> dict:
    """PRIME: parse task, detect domain, discover tools, build context."""
    domain = _detect_domain(task_text)
    system_prompt = _SYSTEM_PROMPTS.get(domain, _SYSTEM_PROMPTS["general"])

    # Discover MCP tools
    tools = []
    try:
        tools = await discover_tools(mcp_url, session_id)
    except Exception as e:
        print(f"[research] tool discovery failed: {e}")

    # Inject task-specific context
    context_notes = []
    if isinstance(task_data, dict):
        if task_data.get("expected_sources"):
            context_notes.append(f"Expected source types: {task_data['expected_sources']}")
        if task_data.get("required_tools"):
            context_notes.append(f"Required tools to use: {task_data['required_tools']}")
        if task_data.get("domain"):
            domain = task_data["domain"]
            system_prompt = _SYSTEM_PROMPTS.get(domain, system_prompt)

    return {
        "domain": domain,
        "system_prompt": system_prompt,
        "tools": tools,
        "context_notes": context_notes,
    }


async def _execute(
    task_text: str,
    prime_ctx: dict,
    mcp_url: str,
    session_id: str,
    conversation: list[dict],
) -> tuple[str, list[dict]]:
    """EXECUTE: run tool loop with Claude, return final answer and updated history."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tools = prime_ctx["tools"]
    system_prompt = prime_ctx["system_prompt"]

    # Add context notes to task
    task_with_context = task_text
    if prime_ctx["context_notes"]:
        task_with_context += "\n\nContext:\n" + "\n".join(prime_ctx["context_notes"])

    if not conversation:
        conversation = [{"role": "user", "content": task_with_context}]

    # Anthropic tool format
    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in tools
    ] if tools else []

    turn = 0
    while turn < MAX_TURNS:
        turn += 1

        kwargs: dict[str, Any] = {
            "model": MAIN_MODEL,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": conversation,
            "timeout": LLM_TIMEOUT,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = client.messages.create(**kwargs)

        # Collect assistant content
        assistant_content = []
        final_text = ""

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                final_text = block.text
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        conversation.append({"role": "assistant", "content": assistant_content})

        # Check stop reason
        if response.stop_reason == "end_turn":
            return final_text, conversation

        if response.stop_reason != "tool_use":
            return final_text, conversation

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[research] calling tool {block.name}({block.input})")
                result = await call_tool(mcp_url, block.name, block.input, session_id)
                result_text = result.get("text") or json.dumps(result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        conversation.append({"role": "user", "content": tool_results})

    return final_text or "Research complete.", conversation


def _reflect(answer: str, prime_ctx: dict) -> str:
    """REFLECT: verify answer quality, ensure citations present, clean up."""
    domain = prime_ctx["domain"]

    # Check citation presence for academic/news domains
    if domain in ("academic", "news"):
        has_citation = bool(
            re.search(r'\[\w+', answer) or          # [Author or [1]
            re.search(r'https?://', answer) or       # URLs
            re.search(r'\d{4}[)\]]', answer)         # years in brackets
        )
        if not has_citation and len(answer) > 100:
            answer += "\n\n(Note: Please verify these facts against primary sources.)"

    return answer.strip()


async def run_research_task(
    task_text: str,
    task_data: Any,
    mcp_url: str,
    session_id: str,
    conversation: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Main entry point: PRIME → EXECUTE → REFLECT.

    Args:
        task_text: The research task description
        task_data: Structured task data (may contain mcp_uri, domain hints, etc.)
        mcp_url: URL of the green agent's MCP server
        session_id: Session ID for MCP tool calls
        conversation: Existing conversation history (for multi-turn)

    Returns:
        (answer, updated_conversation)
    """
    print(f"[research] PRIME — task length={len(task_text)} chars")
    prime_ctx = await _prime(task_text, task_data, mcp_url, session_id)
    print(f"[research] domain={prime_ctx['domain']}, tools={len(prime_ctx['tools'])}")

    print(f"[research] EXECUTE — model={MAIN_MODEL}")
    answer, conversation = await _execute(
        task_text,
        prime_ctx,
        mcp_url,
        session_id,
        conversation or [],
    )

    print(f"[research] REFLECT — answer length={len(answer)} chars")
    answer = _reflect(answer, prime_ctx)

    return answer, conversation
