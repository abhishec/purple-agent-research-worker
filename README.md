# Research AI Worker

> One of four **mini AI workers built on BrainOS** — the Reflexive Agent Architecture framework that achieved **3/3 (100%)** on τ²-Bench. Each worker is a lightweight, self-contained cognitive unit that runs the same PRIME → EXECUTE → REFLECT loop tuned to its domain.

**AgentX Phase 2 — Research Agent Track**

---

## What This Worker Does

The Research AI Worker connects to any MCP tool server and performs deep-research tasks across academic, news, technical, and general domains. It discovers available tools at runtime, classifies the task domain, injects domain-specific cognitive scaffolding, executes an agentic tool loop with citation tracking, and verifies source quality before answering.

---

## BrainOS Cognitive Loop: PRIME → EXECUTE → REFLECT

```
POST /  (A2A JSON-RPC 2.0)
        │
        ▼
    PRIME  ← Reflex Layer
    ├── Domain detection        (academic / news / technical / general / code)
    ├── RL primer injection     (top-3 past cases by keyword + domain relevance)
    ├── DAAO model selection    (Haiku for simple lookups, Sonnet for synthesis)
    ├── Sequence hint injection (prefix-based tool-call directives per domain)
    ├── MCP tool discovery      (green agent's tools fetched at runtime)
    └── Domain system prompt    (citation rules, depth directive, format spec)
        │
        ▼
    EXECUTE  ← LLM Cortex (DAAO: Haiku → Sonnet)
    ├── Agentic tool loop:  search_ → fetch_ / get_ → cite_ / verify_
    ├── Recovery cascade    (empty results → inject broadening hint)
    ├── L3 Citation Contract (academic/news: retry if no citations in answer)
    ├── Self-reflection     (short answer < 100 chars after tool use → depth retry)
    └── Budget / constraint pass-through
        │
        ▼
    REFLECT  ← Verification Layer
    ├── Citation presence audit
    ├── Quality scoring     (0–1 heuristic: citation bonus, length penalty, error penalty)
    ├── RL case recording   (case_log.json, last 20 entries, keyword-indexed)
    └── Structured answer formatting
```

---

## Key BrainOS Concepts Applied

### DAAO — Difficulty-Aware Adaptive Orchestration
Routes each task to the cheapest model that can handle it. Haiku handles simple lookups (`what is`, `define`, `list`) and short tasks (< 12 words with no tools). Sonnet handles academic synthesis, multi-source aggregation, and long-form analysis. Reduces cost on simple tasks while maintaining quality on complex ones.

### RL Primer Injection
Before each task, loads the last 20 completed cases from `case_log.json`. Scores each case by keyword overlap with the current task (Jaccard on 4+ char words) plus domain match bonus (+2.0) plus past quality score. Injects the top-3 most relevant cases as compressed examples into the system prompt, letting the LLM learn from its own execution history without retraining.

### Prefix-Based Sequence Hints
Injects an ordered tool-call directive into every system prompt. Uses **prefixes** not hardcoded tool names — `search_` or `query_`, then `get_` or `fetch_`, then `cite_` or `references_` — so the directive works across any MCP server whose tools follow naming conventions. Per-domain seeds:

| Domain | Sequence |
|---|---|
| `academic` | search_ / query_ → get_ / fetch_ → cite_ / references_ → synthesize |
| `news` | search_ / find_ → get_ / fetch_ → verify_ / check_ → verdict |
| `technical` | list_ / check_ → run_ / execute_ → debug_ / fix_ → verify_ |
| `code` | search_ / find_ → read_ / fetch_ → analyze_ / review_ → suggest_ |
| `general` | search_ → get_ / fetch_ → summarize_ |

### L3 Citation Contract
For academic and news tasks with answers > 100 chars, checks whether the answer contains at least one citation signal: `[Author YYYY]`, a bare URL, or `"Source:"`. If absent, re-runs the LLM with an explicit citation directive injected into the last user turn. Deterministic check — zero extra API cost on passing cases.

### Self-Reflection Retry
After any tool-use execution, measures answer length. If length < 100 chars, injects a depth directive (`"Your answer is too brief. Add specific details, supporting evidence, and sources."`) and re-runs. Forces the model to expand shallow responses without human intervention.

### Recovery Cascade
If a tool result contains `"no results"`, `"not found"`, or `"empty"`, appends a recovery hint to the tool result: `"[RECOVERY HINT: Try broader search terms or alternative spellings. Try the next logical step.]"`. The LLM sees this hint on the next turn and adjusts its search strategy automatically.

---

## Supported Research Domains

| Domain | Task Types | Key Tool Prefixes |
|---|---|---|
| `academic` | Literature review, paper synthesis, citation analysis | search_, fetch_, cite_, references_ |
| `news` | Fact verification, source checking, date verification | search_, find_, verify_, check_ |
| `technical` | Debugging, troubleshooting, environment checks | list_, check_, run_, debug_, fix_ |
| `code` | Code review, bug analysis, refactoring suggestions | find_, read_, analyze_, review_ |
| `general` | Open-ended research, knowledge lookup | search_, get_, summarize_ |

---

## Competition Target

**Primary**: `arunshar/researchtoolbench` — ResearchToolBench
- 5-dimensional scoring: Tool Use (20%) + Source Citation (20%) + Fact Accuracy (25%) + Policy Compliance (15%) + DB State (20%)
- 3 domains with τ²-bench style dual-control environments

---

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI application; A2A JSON-RPC 2.0 handler |
| `research_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT; all BrainOS concepts |
| `mcp_bridge.py` | MCP tool bridge; pre-flight parameter validation; schema patching |
| `config.py` | Environment configuration; model constants; timeout settings |

---

## Requirements

Python 3.11+

```
fastapi>=0.115
uvicorn[standard]>=0.30
anthropic>=0.34
httpx>=0.27
pydantic>=2.0
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `GREEN_AGENT_MCP_URL` | Yes | — | MCP tool server base URL |
| `FALLBACK_MODEL` | No | `claude-sonnet-4-6` | Primary execution model |
| `FAST_MODEL` | No | `claude-haiku-4-5` | Fast model for DAAO simple routing |
| `TOOL_TIMEOUT` | No | `10` | Seconds per tool call |
| `TASK_TIMEOUT` | No | `120` | Seconds per task |
| `RL_CACHE_DIR` | No | `/app` | Directory for `case_log.json` |

---

## Docker

```bash
docker pull public.ecr.aws/d9m7h3k5/agentbench-research:latest
docker run -e ANTHROPIC_API_KEY=sk-ant-... \
           -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
           -p 9010:9010 \
           public.ecr.aws/d9m7h3k5/agentbench-research:latest
```

---

## API

All requests use A2A JSON-RPC 2.0.

| Endpoint | Method | Description |
|---|---|---|
| `/` | POST | `tasks/send` — submit a research task |
| `/.well-known/agent-card.json` | GET | Agent capability declaration |
| `/health` | GET | Health check → `{"status":"ok","agent":"research"}` |

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "id": "task-001",
  "params": {
    "id": "task-001",
    "message": {
      "role": "user",
      "parts": [{ "text": "Summarize the latest research on transformer attention mechanisms and cite key papers." }]
    },
    "metadata": {
      "tools_endpoint": "https://mcp.example.com",
      "session_id": "worker-abc"
    }
  }
}
```

---

## Tech Stack

- **Runtime:** Python 3.11, FastAPI, uvicorn
- **LLM:** Anthropic Claude — Haiku for simple lookups (DAAO fast path); Sonnet for synthesis and multi-source aggregation
- **Architecture:** BrainOS PRIME / EXECUTE / REFLECT cognitive loop
- **RL:** RL case log (JSON) + quality scoring + RL primer injection
- **Tool bridge:** MCP HTTP with pre-flight validation
- **Storage:** Local JSON (`case_log.json` — last 20 entries, keyword-indexed)

---

## License

Apache 2.0
