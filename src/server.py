"""server.py — A2A JSON-RPC 2.0 server for agent-research.

Implements the purple-agent A2A protocol:
  POST /  — A2A JSON-RPC 2.0 (tasks/send, tasks/get, tasks/sendSubscribe)
  GET  /  — agent card (/.well-known/agent-card.json)
  GET  /health — liveness check

Architecture: Reflexive Agent — PRIME → EXECUTE → REFLECT
Domain: Research Agent (academic / news / technical)
"""
from __future__ import annotations
import asyncio
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.research_brain import run_research_task
from src.config import GREEN_AGENT_MCP_URL, PORT, TASK_TIMEOUT

app = FastAPI(title="agent-research", version="1.0.0")

# ── Per-context session state ──────────────────────────────────────────────────
_sessions: dict[str, dict] = {}          # context_id → {conversation, task_data}
_SESSION_TTL: int = 3600                 # 1 hour


def _evict_stale_sessions() -> None:
    now = time.time()
    stale = [k for k, v in _sessions.items() if now - v.get("ts", 0) > _SESSION_TTL]
    for k in stale:
        del _sessions[k]


def _agent_card() -> dict:
    base_url = os.environ.get("AGENT_URL", f"http://localhost:{PORT}")
    return {
        "name": "Research Agent",
        "description": (
            "Purple research agent built on Reflexive Agent Architecture. "
            "Handles academic literature review, news fact-checking, and technical "
            "troubleshooting using MCP tools. Supports dual-control environments "
            "(ResearchToolBench τ²-bench style)."
        ),
        "url": base_url,
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "skills": [
            {
                "id": "research",
                "name": "Research & Analysis",
                "description": "Academic literature review, news verification, technical troubleshooting",
                "tags": ["research", "academic", "news", "technical", "fact-checking"],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card_wellknown():
    return JSONResponse(_agent_card())


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "research", "version": "1.0.0"}


@app.get("/")
async def root_get():
    return JSONResponse(_agent_card())


@app.post("/")
async def root_post(request: Request):
    """Main A2A JSON-RPC 2.0 endpoint."""
    _evict_stale_sessions()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
            status_code=400,
        )

    method = body.get("method", "")
    params = body.get("params", {})
    req_id = body.get("id", str(uuid.uuid4()))

    # ── tasks/send — primary task assignment ──────────────────────────────────
    if method in ("tasks/send", "message/send"):
        return await _handle_task(params, req_id)

    # ── tasks/sendSubscribe — streaming variant (respond as non-streaming) ────
    if method == "tasks/sendSubscribe":
        return await _handle_task(params, req_id)

    # ── tasks/get — return last result ───────────────────────────────────────
    if method == "tasks/get":
        task_id = params.get("id", "")
        session = _sessions.get(task_id)
        if session and session.get("result"):
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "id": task_id,
                    "status": {"state": "completed"},
                    "artifacts": [{"parts": [{"type": "text", "text": session["result"]}]}],
                },
            })
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"id": task_id, "status": {"state": "working"}},
        })

    # ── agent card ────────────────────────────────────────────────────────────
    if method == "agent/getCard":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": _agent_card()})

    return JSONResponse({
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": f"Method not found: {method}"},
        "id": req_id,
    })


async def _handle_task(params: dict, req_id: Any) -> JSONResponse:
    """Core task handler — calls PRIME → EXECUTE → REFLECT."""
    # Extract task info from A2A params
    task_id = params.get("id") or str(uuid.uuid4())
    context_id = params.get("contextId") or task_id

    # Message extraction — support both tasks/send and message/send formats
    messages = params.get("message", {})
    if isinstance(messages, dict):
        parts = messages.get("parts", [])
    elif isinstance(messages, list):
        parts = messages
    else:
        parts = []

    task_text = ""
    task_data: Any = None
    mcp_url = GREEN_AGENT_MCP_URL
    session_id = context_id

    for part in parts:
        if isinstance(part, dict):
            if part.get("type") == "text":
                task_text = part.get("text", "")
            elif part.get("type") == "data":
                task_data = part.get("data", {})
                # Check for MCP URI in data
                if isinstance(task_data, dict):
                    for key in ("mcp_uri", "mcp_url", "tools_endpoint"):
                        if task_data.get(key):
                            mcp_url = task_data[key]
                            break
                    # Also check resources array
                    for r in task_data.get("resources", []):
                        if isinstance(r, dict) and r.get("type") == "mcp":
                            mcp_url = r.get("url") or r.get("uri") or mcp_url
                            break

    # If no structured text, try the raw params message field
    if not task_text and isinstance(params.get("message"), str):
        task_text = params["message"]

    # Get or create session conversation
    session = _sessions.get(context_id, {"conversation": [], "ts": time.time()})

    # Extract session_id from metadata if provided
    metadata = params.get("metadata", {})
    if isinstance(metadata, dict):
        session_id = metadata.get("session_id") or session_id
        if metadata.get("mcp_url"):
            mcp_url = metadata["mcp_url"]

    print(f"[research] task_id={task_id} ctx={context_id} mcp={mcp_url}")
    print(f"[research] task_text={task_text[:120]}...")

    try:
        answer, conversation = await asyncio.wait_for(
            run_research_task(
                task_text=task_text,
                task_data=task_data,
                mcp_url=mcp_url,
                session_id=session_id,
                conversation=session["conversation"],
            ),
            timeout=TASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print(f"[research] TIMEOUT after {TASK_TIMEOUT}s")
        answer = "Research task timed out. Please try a more focused query."
        conversation = session["conversation"]
    except Exception as e:
        print(f"[research] ERROR: {e}")
        import traceback
        traceback.print_exc()
        answer = f"Research task encountered an error: {str(e)}"
        conversation = session["conversation"]

    # Save session
    session["conversation"] = conversation
    session["result"] = answer
    session["ts"] = time.time()
    _sessions[context_id] = session

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "parts": [{"type": "text", "text": answer}],
                    "index": 0,
                }
            ],
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
