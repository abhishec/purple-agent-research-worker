"""research_brain.py — Reflexive Agent core for research tasks.

Implements the three-layer Reflexive Agent Architecture:
  PRIME   → parse task, detect domain, DAAO model routing, RL primer, sequence hints
  EXECUTE → Claude tool loop using MCP tools from green agent
  REFLECT → L3 citation contract, self-reflection, quality scoring, RL case recording

New concepts ported from agent-purple + BrainOS (v2):
  - RL primer injection: top-3 relevant past cases injected into system prompt
  - DAAO model routing: Haiku for simple factual, Sonnet for synthesis/multi-source
  - Sequence hints: ordered tool-call directives per domain (prefix-based, not hardcoded)
  - L3 citation contract: academic/news tasks → verify citations present, retry if missing
  - Self-reflection: short answer after tool use → retry with depth directive
  - Quality scoring: citation presence, tool depth, answer length, domain signals
  - RL case log: persistent, keyword-indexed, last 20 entries per domain
  - Recovery cascade: tool failure → degrade gracefully, log penalty

Supports ResearchToolBench domains:
  - academic  : literature review, paper synthesis, citation extraction
  - news      : fact verification, source checking, temporal reasoning
  - technical : troubleshooting, debugging, dual-control (user+agent tools)
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, MAIN_MODEL, FAST_MODEL, LLM_TIMEOUT, MAX_TURNS
from src.mcp_bridge import discover_tools, call_tool

# ── RL case log ───────────────────────────────────────────────────────────────
_RL_DIR = Path(os.environ.get("RL_CACHE_DIR", "/app"))
_CASE_LOG = _RL_DIR / "research_case_log.json"
_MAX_CASES = 20


def _load_cases() -> list[dict]:
    try:
        with open(_CASE_LOG) as f:
            return json.load(f)
    except Exception:
        return []


def _save_case(case: dict) -> None:
    cases = _load_cases()
    cases.append(case)
    if len(cases) > _MAX_CASES:
        cases = cases[-_MAX_CASES:]
    try:
        _RL_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CASE_LOG, "w") as f:
            json.dump(cases, f, indent=2)
    except Exception as e:
        print(f"[research/rl] save failed: {e}")


def _score_quality(answer: str, domain: str, tool_count: int) -> float:
    """Quality heuristic 0-1. Conservative baseline 0.5."""
    q = 0.5
    # Length signals
    if len(answer) < 50:
        q -= 0.3
    elif len(answer) > 300:
        q += 0.1
    elif len(answer) > 800:
        q += 0.15
    # Tool depth
    if tool_count == 0:
        q -= 0.2
    elif tool_count >= 3:
        q += 0.1
    # Domain-specific: citation presence
    if domain in ("academic", "news"):
        has_citation = bool(
            re.search(r'\[\w+[^\]]*\d{4}', answer) or    # [Smith et al., 2023]
            re.search(r'https?://', answer) or             # URLs
            re.search(r'\(\d{4}\)', answer) or             # (2023)
            re.search(r'doi:', answer, re.I)               # DOI
        )
        if has_citation:
            q += 0.15
        else:
            q -= 0.1
    # Error / refusal penalty
    if re.search(r"I don't have|I cannot|I'm unable|error|failed", answer, re.I):
        q -= 0.15
    return max(0.0, min(1.0, q))


def _build_rl_primer(task_text: str, domain: str) -> str:
    """Build RL primer from top-3 most relevant past cases."""
    cases = _load_cases()
    if not cases:
        return ""

    task_words = set(re.findall(r'\b\w{4,}\b', task_text.lower()))

    def relevance(c: dict) -> float:
        kw = set(c.get("keywords", []))
        overlap = len(task_words & kw)
        domain_bonus = 2.0 if c.get("domain") == domain else 0.0
        quality_weight = c.get("quality", 0.5)
        return overlap + domain_bonus + quality_weight

    top = sorted(cases, key=relevance, reverse=True)[:3]
    if not top:
        return ""

    lines = ["LEARNED PATTERNS from similar past research tasks:"]
    for c in top:
        sym = "✓" if c.get("outcome") == "success" else "✗"
        worked = c.get("what_worked", "")
        failed = c.get("what_failed", "")
        summary = c.get("task_summary", "")[:70]
        lines.append(f"  {sym} [{c.get('domain', 'general')}] {summary}")
        if worked:
            lines.append(f"     Worked: {worked[:80]}")
        if failed:
            lines.append(f"     Failed: {failed[:80]}")
    return "\n".join(lines)


# ── DAAO: Difficulty-Aware Adaptive Orchestration ─────────────────────────────
_SIMPLE_PREFIXES = (
    "what is", "who is", "when did", "list ", "how many", "define ",
    "what are", "name the", "how does",
)
_COMPLEX_KEYWORDS = (
    "synthesize", "compare", "analyze", "evaluate", "investigate",
    "literature review", "systematic", "meta-analysis", "comprehensive",
    "critically assess", "evidence for",
)


def _select_model(task_text: str, domain: str, has_tools: bool) -> str:
    """DAAO: route to fastest sufficient model.

    Haiku for:  simple factual lookups, short queries with no tools
    Sonnet for: academic synthesis, multi-source analysis, complex technical
    """
    text_lower = task_text.lower()
    words = task_text.split()

    # Academic + complex always gets Sonnet
    if domain == "academic" or any(k in text_lower for k in _COMPLEX_KEYWORDS):
        return MAIN_MODEL

    # Short simple factual → Haiku
    if (len(words) < 12 and
            any(text_lower.startswith(p) for p in _SIMPLE_PREFIXES) and
            not has_tools):
        return FAST_MODEL

    # Technical troubleshooting → Sonnet (needs step-by-step reasoning)
    if domain == "technical":
        return MAIN_MODEL

    # Default: Sonnet for >30 words, Haiku otherwise
    return MAIN_MODEL if len(words) > 25 else FAST_MODEL


# ── Domain detection ──────────────────────────────────────────────────────────
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "academic": [
        "paper", "literature", "research", "study", "publication", "arxiv",
        "journal", "citation", "review", "abstract", "methodology", "findings",
        "hypothesis", "experiment", "dataset", "scholar", "peer-reviewed",
    ],
    "news": [
        "news", "article", "report", "headline", "event", "incident",
        "according to", "journalist", "source", "verify", "fact-check",
        "claim", "breaking", "latest", "announced", "reported",
    ],
    "technical": [
        "error", "bug", "fix", "debug", "install", "package", "library",
        "pytorch", "tensorflow", "pip", "conda", "import", "module",
        "traceback", "exception", "stack trace", "version", "dependency",
        "build", "compile", "runtime",
    ],
}


def _detect_domain(task_text: str) -> str:
    text_lower = task_text.lower()
    scores = {
        d: sum(1 for k in kws if k in text_lower)
        for d, kws in _DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])
    return best_domain if best_score >= 2 else "general"


# ── Sequence hints (prefix-based, not hardcoded tool names) ──────────────────
_SEQUENCE_HINTS: dict[str, str] = {
    "academic": (
        "RECOMMENDED TOOL SEQUENCE for academic research:\n"
        "1. search_ or query_ → find relevant papers/sources\n"
        "2. get_ or fetch_ or read_ → retrieve paper details, abstracts\n"
        "3. cite_ or references_ → extract citations if needed\n"
        "4. synthesize in final answer with structured citations"
    ),
    "news": (
        "RECOMMENDED TOOL SEQUENCE for news verification:\n"
        "1. search_ or find_ → locate primary news sources\n"
        "2. get_ or fetch_ → read article content and dates\n"
        "3. verify_ or check_ → cross-reference across sources\n"
        "4. Verdict: Verified / Disputed / Unverified + evidence"
    ),
    "technical": (
        "RECOMMENDED TOOL SEQUENCE for technical troubleshooting:\n"
        "1. check_ or read_ or get_ → inspect current system state\n"
        "2. search_ or query_ → find relevant documentation/solutions\n"
        "3. run_ or execute_ or fix_ → apply the fix\n"
        "4. verify_ or test_ → confirm the fix worked"
    ),
    "general": (
        "RECOMMENDED TOOL SEQUENCE:\n"
        "1. search_ or find_ → locate relevant information\n"
        "2. get_ or read_ → retrieve details\n"
        "3. Synthesize and cite sources in final answer"
    ),
}

# ── System prompts ────────────────────────────────────────────────────────────
_SYSTEM_PROMPTS: dict[str, str] = {
    "academic": """You are an expert research assistant specializing in academic literature.

KEY BEHAVIORS:
1. Use search/fetch tools to find relevant papers and sources
2. ALWAYS cite sources: [Author et al., Year] "Title" — key finding
3. Verify facts across multiple sources before stating them
4. Summarize findings accurately — do not over-claim
5. Use exact quotes when precision matters
6. If papers conflict, explicitly note the disagreement

STRUCTURED ANSWER FORMAT:
- Summary of key findings (2-4 sentences)
- Numbered source list with authors, year, title, key finding
- Confidence level: high / medium / low""",

    "news": """You are a fact-checking research assistant specializing in news verification.

KEY BEHAVIORS:
1. Search for primary sources — news agencies, official statements, databases
2. Cross-check claims across multiple independent sources
3. Note publication date and recency of sources
4. Distinguish verified facts from unverified claims
5. Flag conflicting reports or disputed information

STRUCTURED ANSWER FORMAT:
- Verdict: Verified / Unverified / Disputed / Insufficient Evidence
- Supporting sources with dates and URLs
- Summary of what IS and ISN'T confirmed""",

    "technical": """You are a technical support specialist and debugging expert.

KEY BEHAVIORS:
1. Systematically diagnose: read error messages, check versions, inspect environment
2. Follow: Reproduce → Diagnose → Root Cause → Fix → Verify
3. Use available tools to check system state and apply fixes
4. If the user also has tools (dual-control), coordinate — request user actions when needed
5. Verify the fix worked after implementation

STRUCTURED ANSWER FORMAT:
- Root cause identified
- Step-by-step fix applied
- Verification result""",

    "general": """You are a research agent with access to MCP tools.

KEY BEHAVIORS:
1. Use tools to gather accurate, up-to-date information
2. Cite sources clearly
3. Verify key facts before stating them
4. Provide a structured, well-organized answer
5. Acknowledge uncertainty when present""",
}


def _extract_mcp_uri(task_data: Any) -> str | None:
    if isinstance(task_data, dict):
        for key in ("mcp_uri", "mcp_url", "tools_endpoint"):
            if task_data.get(key):
                return task_data[key]
        for r in task_data.get("resources", []):
            if isinstance(r, dict) and r.get("type") == "mcp":
                return r.get("url") or r.get("uri")
    return None


# ── PRIME ─────────────────────────────────────────────────────────────────────
async def _prime(task_text: str, task_data: Any, mcp_url: str, session_id: str) -> dict:
    """PRIME: detect domain, discover tools, route model, build enriched context."""
    domain = _detect_domain(task_text)

    # Task_data overrides
    context_notes: list[str] = []
    if isinstance(task_data, dict):
        if task_data.get("domain"):
            domain = task_data["domain"]
        if task_data.get("expected_sources"):
            context_notes.append(f"Expected source types: {task_data['expected_sources']}")
        if task_data.get("required_tools"):
            context_notes.append(f"Required tools: {task_data['required_tools']}")

    # Discover MCP tools
    tools: list[dict] = []
    try:
        tools = await discover_tools(mcp_url, session_id)
    except Exception as e:
        print(f"[research] tool discovery failed: {e}")

    has_tools = len(tools) > 0

    # DAAO: select model
    model = _select_model(task_text, domain, has_tools)
    print(f"[research] DAAO → model={model} (domain={domain}, tools={has_tools})")

    # RL primer (top-3 relevant past cases)
    rl_primer = _build_rl_primer(task_text, domain)

    # Sequence hint for this domain
    seq_hint = _SEQUENCE_HINTS.get(domain, _SEQUENCE_HINTS["general"])

    # Build enriched system prompt
    base_prompt = _SYSTEM_PROMPTS.get(domain, _SYSTEM_PROMPTS["general"])
    system_parts = [base_prompt, "", seq_hint]
    if rl_primer:
        system_parts += ["", rl_primer]
    system_prompt = "\n".join(system_parts)

    return {
        "domain": domain,
        "model": model,
        "system_prompt": system_prompt,
        "tools": tools,
        "context_notes": context_notes,
        "has_tools": has_tools,
    }


# ── EXECUTE ───────────────────────────────────────────────────────────────────
async def _execute(
    task_text: str,
    prime_ctx: dict,
    mcp_url: str,
    session_id: str,
    conversation: list[dict],
    retry_directive: str | None = None,
) -> tuple[str, list[dict], int]:
    """EXECUTE: agentic Claude tool loop. Returns (answer, conversation, tool_count)."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tools = prime_ctx["tools"]
    system_prompt = prime_ctx["system_prompt"]
    model = prime_ctx["model"]

    # Build initial user message
    task_with_context = task_text
    if prime_ctx["context_notes"]:
        task_with_context += "\n\nContext:\n" + "\n".join(prime_ctx["context_notes"])
    if retry_directive:
        task_with_context += f"\n\n{retry_directive}"

    if not conversation:
        if not task_with_context.strip():
            task_with_context = task_text or "Please proceed."
        conversation = [{"role": "user", "content": task_with_context}]

    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in tools
    ] if tools else []

    turn = 0
    final_text = ""
    tool_count = 0

    while turn < MAX_TURNS:
        turn += 1

        # Validate conversation before calling API
        # Ensure no user message has empty content
        clean_conversation = []
        for msg in conversation:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list) and len(content) == 0:
                    continue  # skip empty tool_results messages
                if isinstance(content, str) and not content.strip():
                    continue
            clean_conversation.append(msg)
        if not clean_conversation:
            break

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": clean_conversation,
            "timeout": LLM_TIMEOUT,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            response = client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            print(f"[research] API error turn {turn}: {e}")
            break

        assistant_content: list[dict] = []
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

        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":
            break

        # Execute tool calls — only append if non-empty
        tool_results: list[dict] = []
        for block in response.content:
            if block.type == "tool_use":
                tool_count += 1
                print(f"[research] → {block.name}({json.dumps(block.input)[:80]})")
                try:
                    result = await call_tool(mcp_url, block.name, block.input, session_id)
                    result_text = result.get("text") or json.dumps(result)
                except Exception as e:
                    result_text = f"Tool error: {e}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        # *** BUG FIX: never append empty tool_results ***
        if tool_results:
            conversation.append({"role": "user", "content": tool_results})

    return final_text, conversation, tool_count


# ── REFLECT ───────────────────────────────────────────────────────────────────
async def _reflect(
    answer: str,
    prime_ctx: dict,
    conversation: list[dict],
    mcp_url: str,
    session_id: str,
    tool_count: int,
) -> tuple[str, list[dict], int]:
    """REFLECT: L3 citation contract + self-reflection retry.

    L3 Citation Contract:
      academic/news domains + no citation pattern + answer >100 chars → retry once
      with explicit "add citations" directive.

    Self-Reflection:
      Answer <100 chars after tool use → retry with depth directive.
    """
    domain = prime_ctx["domain"]

    # ── L3: Citation contract (academic / news) ──────────────────────────────
    if domain in ("academic", "news") and len(answer) > 100:
        has_citation = bool(
            re.search(r'\[\w+[^\]]*\d{4}', answer) or
            re.search(r'https?://', answer) or
            re.search(r'\(\d{4}\)', answer) or
            re.search(r'doi:', answer, re.I)
        )
        if not has_citation:
            print(f"[research/L3] citation contract failed — retrying with citation directive")
            directive = (
                "CITATION REQUIRED: Your previous answer is missing source citations. "
                "Please revise: include at least 2 citations in the format "
                "[Author et al., Year] or provide URLs. Do not invent sources — "
                "if you cannot cite, state 'Source: not found via available tools'."
            )
            answer, conversation, extra_tools = await _execute(
                answer,
                prime_ctx,
                mcp_url,
                session_id,
                conversation,
                retry_directive=directive,
            )
            tool_count += extra_tools

    # ── Self-reflection: short answer after tool use ──────────────────────────
    if tool_count > 0 and len(answer) < 100 and not re.search(r'error|failed|not found', answer, re.I):
        print(f"[research/reflect] short answer after tool use ({len(answer)} chars) — requesting depth")
        directive = (
            "Your response is too brief. Please provide a comprehensive answer "
            "with specific details, data, and proper citations from the sources you retrieved."
        )
        answer, conversation, extra_tools = await _execute(
            answer,
            prime_ctx,
            mcp_url,
            session_id,
            conversation,
            retry_directive=directive,
        )
        tool_count += extra_tools

    return answer.strip() if answer else "Research complete.", conversation, tool_count


# ── Main entry point ──────────────────────────────────────────────────────────
async def run_research_task(
    task_text: str,
    task_data: Any,
    mcp_url: str,
    session_id: str,
    conversation: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """PRIME → EXECUTE → REFLECT.

    Args:
        task_text:   The research task description
        task_data:   Structured task data (mcp_uri, domain hints, required_tools, etc.)
        mcp_url:     URL of the green agent's MCP server
        session_id:  Session ID for MCP tool calls
        conversation: Existing conversation history (for multi-turn)

    Returns:
        (answer, updated_conversation)
    """
    task_start = time.monotonic()

    # Override MCP URL from task_data if present
    task_mcp_uri = _extract_mcp_uri(task_data)
    if task_mcp_uri:
        mcp_url = task_mcp_uri

    print(f"[research] PRIME — task={task_text[:80]!r}")
    prime_ctx = await _prime(task_text, task_data, mcp_url, session_id)
    print(f"[research] domain={prime_ctx['domain']}, model={prime_ctx['model']}, "
          f"tools={len(prime_ctx['tools'])}")

    print(f"[research] EXECUTE")
    answer, conversation, tool_count = await _execute(
        task_text,
        prime_ctx,
        mcp_url,
        session_id,
        list(conversation) if conversation else [],
    )

    print(f"[research] REFLECT — answer={len(answer)} chars, tool_count={tool_count}")
    answer, conversation, tool_count = await _reflect(
        answer, prime_ctx, conversation, mcp_url, session_id, tool_count
    )

    # ── Record RL case ──────────────────────────────────────────────────────
    quality = _score_quality(answer, prime_ctx["domain"], tool_count)
    outcome = "success" if quality >= 0.5 else "failure"
    task_kws = list(set(re.findall(r'\b\w{5,}\b', task_text.lower())))[:15]

    _save_case({
        "task_summary": task_text[:100],
        "domain": prime_ctx["domain"],
        "outcome": outcome,
        "quality": round(quality, 3),
        "tool_count": tool_count,
        "model": prime_ctx["model"],
        "keywords": task_kws,
        "what_worked": f"model={prime_ctx['model']}, domain={prime_ctx['domain']}, tools={tool_count}",
        "what_failed": "" if quality >= 0.5 else f"quality={quality:.2f}",
        "duration_s": round(time.monotonic() - task_start, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    print(f"[research] RL quality={quality:.3f} outcome={outcome} "
          f"duration={time.monotonic()-task_start:.1f}s")

    return answer, conversation
