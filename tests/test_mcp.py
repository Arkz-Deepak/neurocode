"""Tests for NeuroCode MCP Server tools and JSON-RPC dispatch."""
import json
import pytest
from src.mcp_server import handle_slice, handle_impact, handle_seeds, handle_symbol, handle_validate_patch, handle_graft, TOOLS


def test_mcp_tools_list_schema():
    tool_names = [t["name"] for t in TOOLS]
    assert "neurocode_slice" in tool_names
    assert "neurocode_impact" in tool_names
    assert "neurocode_seeds" in tool_names
    assert "neurocode_symbol" in tool_names
    assert "neurocode_validate_patch" in tool_names
    assert "neurocode_graft" in tool_names
    assert "neurocode_trace" in tool_names
    assert "neurocode_map" in tool_names
    assert "neurocode_repair_ollama" in tool_names
    assert "neurocode_hardware_health" in tool_names


def test_mcp_handle_symbol():
    res_str = handle_symbol({"repo_path": ".", "symbol": "RepoGraphBuilder"})
    data = json.loads(res_str)
    assert data["matches_found"] > 0
    assert any("RepoGraphBuilder" in str(e["entity"]) for e in data["symbols"])


def test_mcp_handle_validate_patch():
    res_str = handle_validate_patch({
        "repo_path": ".",
        "file_path": "tests/test_graph.py",
        "search_block": "import pytest",
        "replace_block": "import pytest\n# verified"
    })
    data = json.loads(res_str)
    assert data["is_valid"] is True


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


def test_mcp_handle_graft(tmp_path):
    py_code = "def greet(name: str) -> str:\n    return 'Hello ' + name\n"
    f = tmp_path / "greeting.py"
    f.write_text(py_code, encoding="utf-8")

    res_str = handle_graft({
        "repo_path": str(tmp_path),
        "file_path": "greeting.py",
        "symbol": "greet",
        "replacement_code": "def greet(name: str) -> str:\n    return f'Welcome, {name}!'",
        "apply": True
    })
    data = json.loads(res_str)
    assert data["success"] is True
    assert data["syntax_valid"] is True
    assert data["applied"] is True
    assert "Welcome" in f.read_text(encoding="utf-8")
