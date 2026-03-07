"""config.py — environment configuration for agent-research."""
from __future__ import annotations
import os

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
GREEN_AGENT_MCP_URL: str = os.environ.get("GREEN_AGENT_MCP_URL", "http://green-agent:9009")
PORT: int = int(os.environ.get("PORT", "9010"))

# Model selection
FAST_MODEL: str = os.environ.get("FAST_MODEL", "claude-haiku-4-5-20251001")
MAIN_MODEL: str = os.environ.get("MAIN_MODEL", "claude-sonnet-4-6")

# Timeouts
TOOL_TIMEOUT: int = int(os.environ.get("TOOL_TIMEOUT", "60"))
LLM_TIMEOUT: int = int(os.environ.get("LLM_TIMEOUT", "120"))
MAX_TURNS: int = int(os.environ.get("MAX_TURNS", "10"))
TASK_TIMEOUT: int = int(os.environ.get("TASK_TIMEOUT", "110"))
