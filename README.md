# agent-research — BrainOS Mini AI Worker

> **ResearchToolBench · AgentX Phase 2 — Research Track**
> One of five BrainOS Mini AI Workers — a self-contained research cognitive unit built on the **PRIME → EXECUTE → REFLECT** loop with domain-adaptive intelligence.

---

## The Problem

Research agents fail in two ways that retrieval-augmented generation alone cannot fix.

**Depth failure.** The agent returns a shallow answer — one source, no synthesis, no citations — because the LLM takes the path of least resistance. There is no mechanism forcing it to go deep before answering.

**Domain blindness.** A single generic prompt cannot simultaneously handle academic literature synthesis (cite papers, track claims), investigative news verification (cross-source, check dates), technical debugging (reproduce, trace, fix), and code review (read context, suggest changes). Each domain has different tool sequences, different quality signals, and different what-counts-as-done criteria.

The result: agents that can search but cannot research.

---

## BrainOS Innovation: Domain-Adaptive Cognitive Scaffolding

The Research AI Worker solves this by running a domain-specific cognitive loop — not a single generic prompt, but a structurally different execution path per research domain, built on BrainOS's PRIME → EXECUTE → REFLECT architecture.

---

## Core Technical Innovations

### 1 — DAAO: Difficulty-Aware Adaptive Orchestration

Every task is routed to the cheapest model that can handle it before a single token is spent on execution.

**Fast path (Haiku):** Simple lookups — `"what is"`, `"define"`, `"list"`, tasks under 12 words with no tool requirements. Zero Sonnet cost.

**Deep path (Sonnet):** Academic synthesis, multi-source aggregation, long-form analysis, cross-reference tasks.

The routing decision is deterministic — keyword pattern + task length — so it costs nothing to evaluate. DAAO reduces inference cost on simple tasks while preserving quality on complex ones without any model-quality trade-off.

### 2 — Prefix-Based Sequence Hints (Protocol-Agnostic)

Every research task receives an ordered tool-call directive injected into the system prompt. The critical design: directives use **prefixes**, not hardcoded tool names.

| Domain | Injected Sequence |
|---|---|
| `academic` | `search_` / `query_` → `get_` / `fetch_` → `cite_` / `references_` → synthesize |
| `news` | `search_` / `find_` → `get_` / `fetch_` → `verify_` / `check_` → verdict |
| `technical` | `list_` / `check_` → `run_` / `execute_` → `debug_` / `fix_` → `verify_` |
| `code` | `search_` / `find_` → `read_` / `fetch_` → `analyze_` / `review_` → suggest |
| `general` | `search_` → `get_` / `fetch_` → `summarize_` |

Because directives match prefixes against the actual tool names at runtime, the same sequence works across any MCP server regardless of whether the tool is called `search_arxiv`, `search_web`, or `search_pubmed`. Zero hardcoding. Works against any green agent.

### 3 — L3 Citation Contract (Deterministic Quality Enforcement)

For academic and news tasks producing answers over 100 characters, a deterministic post-execution check verifies that at least one citation signal is present: `[Author YYYY]` pattern, a bare URL, or `"Source:"`. If absent, the L3 contract re-runs the LLM with an explicit citation directive injected into the last user turn.

Cost: zero on passing cases (pure string check). On failure: one targeted retry that adds exactly what was missing. No human intervention. No threshold tuning.

### 4 — Self-Reflection Depth Retry

After any tool-use execution, answer length is measured. If length < 100 characters (the LLM gave a shallow answer despite having tool results), a depth directive is injected and execution re-runs:

> *"Your answer is too brief. Add specific details, supporting evidence, and sources."*

This catches the most common research agent failure: finding the right sources but summarizing them into one sentence.

### 5 — RL Primer Injection

Before each task, the 20 most recent cases are loaded from `case_log.json`. Each is scored by Jaccard keyword overlap with the current task plus domain match bonus (+2.0) plus past quality score. The top-3 most relevant are injected as compressed examples into the system prompt.

The agent sees its own past execution history on every call — what searches worked, what citation formats succeeded, how it handled similar queries. No retraining. Immediate feedback loop.

### 6 — Recovery Cascade

When a tool returns `"no results"`, `"not found"`, or `"empty"`, a recovery hint is appended to the tool result before the LLM's next turn:

> *"[RECOVERY HINT: Try broader search terms or alternative spellings. Try the next logical step.]"*

The LLM sees the hint and broadens its query automatically on the next iteration. This handles the most common research dead-end — an overspecific initial query — without requiring human re-prompting.

---

## Supported Research Domains

| Domain | Task Types | Key Contracts |
|---|---|---|
| `academic` | Literature review, paper synthesis, citation analysis | L3 Citation + Self-Reflection + RL Primer |
| `news` | Fact verification, source checking, date verification | L3 Citation + Recovery Cascade |
| `technical` | Debugging, troubleshooting, environment checks | Sequence Hints (list→run→debug→verify) |
| `code` | Code review, bug analysis, refactoring suggestions | DAAO fast path if simple; Sonnet for deep review |
| `general` | Open-ended research, knowledge lookup | Recovery Cascade + Self-Reflection |

---

## Cognitive Loop: PRIME → EXECUTE → REFLECT

```
PRIME
├── Domain detection        (academic / news / technical / code / general)
├── RL primer injection     (top-3 past cases by keyword + domain match)
├── DAAO model selection    (Haiku for simple; Sonnet for synthesis)
├── Sequence hint injection (prefix-based directives per domain)
├── MCP tool discovery      (green agent tools fetched at runtime)
└── Domain system prompt    (citation rules, depth directive, format spec)

EXECUTE
├── Agentic tool loop:  search_ → fetch_ / get_ → cite_ / verify_
├── Recovery cascade    (empty results → broadening hint)
├── L3 Citation Contract (academic/news: retry if no citations)
└── Self-reflection     (answer < 100 chars after tool use → depth retry)

REFLECT
├── Citation presence audit
├── Quality scoring     (citation bonus, length check, error penalty)
├── RL case recording   (case_log.json, last 20 entries, keyword-indexed)
└── Structured answer formatting
```

---

## Competition Target

**ResearchToolBench** (`arunshar/researchtoolbench`) — 5-dimensional scoring:
- Tool Use (20%) + Source Citation (20%) + Fact Accuracy (25%) + Policy Compliance (15%) + DB State (20%)

---

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI; A2A JSON-RPC 2.0 handler |
| `research_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT |
| `mcp_bridge.py` | MCP HTTP; pre-flight validation; schema patching |
| `config.py` | Environment config; model constants; timeout settings |

---

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export GREEN_AGENT_MCP_URL=http://localhost:9009
PORT=9011 python3 src/server.py
```

**Docker:**
```bash
docker pull public.ecr.aws/d9m7h3k5/agentbench-research:latest
docker run -e ANTHROPIC_API_KEY=sk-ant-... \
           -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
           -p 9011:9011 \
           public.ecr.aws/d9m7h3k5/agentbench-research:latest
```

---

## Tech Stack

- **Runtime:** Python 3.11 · FastAPI · uvicorn
- **LLM:** claude-haiku-4-5-20251001 (DAAO fast path) · claude-sonnet-4-6 (synthesis, multi-source)
- **Architecture:** BrainOS PRIME / EXECUTE / REFLECT
- **Core library:** [brainos-core-light](https://github.com/abhishec/brainoscorelight) — shared primitives (Brain, Router, UCB1, RL)
- **RL:** Case log (JSON) · quality scoring · RL primer injection
- **Tool bridge:** MCP HTTP with pre-flight validation
- **Protocol:** A2A JSON-RPC 2.0

---

## License

Apache 2.0
