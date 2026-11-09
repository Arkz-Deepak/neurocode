"""Tests for Execution Path Tracing and Architecture Mapping."""
import pytest
from src.graph.builder import RepoGraphBuilder
from src.mcp_server import handle_trace, handle_map


def test_trace_path_direct():
    builder = RepoGraphBuilder(".")
    res = builder.trace_path("cmd_slice", "CodeGraphSlicer")
    assert res["path_found"] is True
    assert len(res["call_chain"]) >= 2
    assert res["call_chain"][0]["name"] == "cmd_slice"


def test_generate_mermaid_map():
    builder = RepoGraphBuilder(".")
    mermaid = builder.generate_mermaid_map()
    assert "flowchart TD" in mermaid
    assert "subgraph" in mermaid


def test_mcp_handle_trace():
    import json
    res_str = handle_trace({
        "repo_path": ".",
        "from": "cmd_slice",
        "to": "CodeGraphSlicer"
    })
    data = json.loads(res_str)
    assert data["path_found"] is True


def test_mcp_handle_map():
    res_str = handle_map({"repo_path": "."})
    assert "flowchart TD" in res_str
    assert "```mermaid" in res_str
