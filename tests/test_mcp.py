"""Tests for NeuroCode MCP Server tools and JSON-RPC dispatch."""
import json
import pytest
from src.mcp_server import handle_slice, handle_impact, handle_seeds, TOOLS


def test_mcp_tools_list_schema():
    tool_names = [t["name"] for t in TOOLS]
    assert "neurocode_slice" in tool_names
    assert "neurocode_impact" in tool_names
    assert "neurocode_seeds" in tool_names
    assert "neurocode_repair_ollama" in tool_names
    assert "neurocode_hardware_health" in tool_names


def test_mcp_handle_impact():
    res = handle_impact({"repo_path": ".", "target": "RepoGraphBuilder"})
    assert "Structural Impact Analysis" in res
    assert "RepoGraphBuilder" in res
    assert "Upstream Callers" in res


def test_mcp_handle_slice():
    res = handle_slice({"repo_path": ".", "issue": "slicer", "budget": 1200})
    assert "REPOSITORY CONTEXT SLICE" in res
    assert "Tokens" in res or "tokens" in res


def test_mcp_handle_seeds():
    res_str = handle_seeds({"repo_path": ".", "issue": "graph"})
    data = json.loads(res_str)
    assert "identified_seeds" in data
    assert "analysis_latency_ms" in data
