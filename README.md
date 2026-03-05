# Research AI Worker

Purple research agent for the **AgentX Phase 2 — Research Agent Track** (2nd Sprint, March 23 – April 12, 2026).

Built on the **Reflexive Agent Architecture** — the same dual-process cognitive design that achieved **3/3 (100%)** on τ²-Bench airline domain in Sprint 1.

---

## Architecture

```
POST /  (A2A JSON-RPC 2.0)
        │
        ▼
    PRIME  ← Reflex Layer
    ├── Domain detection (academic / news / technical)
    ├── Budget extraction & constraint parsing
    ├── MCP tool discovery (green agent's tools)
    └── Domain-specific system prompt injection
        │
        ▼
    EXECUTE  ← LLM Cortex (Claude)
    ├── Tool loop: search, fetch, fact-check, cite
    ├── Citation tracking
    └── Source verification
        │
        ▼
    REFLECT  ← Verification Layer
    ├── Citation presence check
    ├── Fact coverage audit
    └── Structured output formatting
```

## Supported Domains

| Domain | Description | Key Tools |
|--------|-------------|-----------|
| Academic | Literature review, paper synthesis | search_papers, fetch_abstract, cite |
| News | Fact verification, source checking | search_news, verify_claim, check_date |
| Technical | Debugging, troubleshooting | run_command, check_env, install_pkg |

## Competition Target

**Primary**: `arunshar/researchtoolbench` — ResearchToolBench
- 5-dimensional scoring: Tool Use (20%) + Source Citation (20%) + Fact Accuracy (25%) + Policy Compliance (15%) + DB State (20%)
- 3 domains with τ²-bench style dual-control environments

## Deployment

```bash
docker build -t agent-research .
docker run -p 9010:9010 \
  -e ANTHROPIC_API_KEY=... \
  -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
  agent-research
```

## API

```
GET  /health                         → {"status":"ok"}
GET  /.well-known/agent-card.json   → Agent card
POST /                               → A2A JSON-RPC 2.0 (tasks/send)
```
