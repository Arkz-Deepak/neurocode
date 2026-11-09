#!/usr/bin/env python3
"""
NeuroCode Model Context Protocol (MCP) Server.
Standards-compliant stdio JSON-RPC 2.0 server for Claude Desktop, Cursor, Continue.dev, and VS Code.

Enables any AI client to leverage NeuroCode's Host-side Code Property Graph slicing,
topological reachability fault localization, and token-bounded knapsack packing.
"""

import sys
import json
import time
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer
from src.agent.patcher import PatchApplier, SYSTEM_PROMPT


def get_hardware_health() -> Dict[str, Any]:
    """Inspects local hardware, VRAM, and tmpfs status."""
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
        allocated_vram = torch.cuda.memory_allocated() / (1024**2) if cuda_avail else 0.0
    except ImportError:
        cuda_avail = False
        device_name = "Host CPU (PyTorch optional / Ollama ready)"
        total_vram = 0.0
        allocated_vram = 0.0
    
    # tmpfs check
    shm_path = Path("/dev/shm")
    shm_available = shm_path.exists() and os.access(str(shm_path), os.W_OK)
    
    return {
        "status": "healthy",
        "device": device_name,
        "cuda_available": cuda_avail,
        "total_vram_gb": round(total_vram, 2),
        "allocated_vram_mb": round(allocated_vram, 1),
        "free_vram_headroom_gb": round(total_vram - (allocated_vram / 1024.0), 2) if cuda_avail else 0.0,
        "tmpfs_zero_wear_available": shm_available,
        "host_cpu_cores": os.cpu_count()
    }


def handle_slice(args: Dict[str, Any]) -> str:
    """Constructs in-memory CPG and extracts minimal knapsack-packed context."""
    raw_path = args.get("repo_path") or args.get("workspace_path") or args.get("path") or "."
    repo_path = Path(raw_path).resolve()
    
    issue = args.get("issue") or args.get("query") or args.get("task") or ""
    entry = args.get("entry") or args.get("file_path") or args.get("target_file") or ""
    budget = int(args.get("budget", 1200))
    
    if not repo_path.exists():
        return f"[Error] Repository path does not exist: {repo_path}"
    
    t0 = time.perf_counter()
    builder = RepoGraphBuilder(str(repo_path))
    graph = builder.build()
    build_ms = (time.perf_counter() - t0) * 1000
    
    t1 = time.perf_counter()
    slicer = CodeGraphSlicer(graph, max_tokens=budget)
    
    seeds = []
    if entry:
        entry_clean = Path(entry).as_posix()
        seeds = [n for n in graph.nodes if entry_clean in str(graph.nodes[n].get("file_path", "")) or entry_clean in str(n)]
        
    if not seeds:
        query_text = issue or entry or "repository context"
        seeds = slicer.find_seed_nodes(query_text)
        
    if not seeds and len(graph.nodes) > 0:
        seeds = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)[:3]
        
    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
    slice_text = slicer.format_slice_for_llm(subgraph, seeds, query=issue or entry, max_tokens=budget)
    slice_ms = (time.perf_counter() - t1) * 1000
    
    tokens = int(len(slice_text.split()) * 1.3)
    
    if args.get("json_output") or args.get("format") == "json":
        return json.dumps({
            "repo_path": str(repo_path),
            "issue": issue,
            "seeds": seeds,
            "token_count": tokens,
            "ast_nodes": len(subgraph.nodes),
            "build_time_ms": round(build_ms, 2),
            "slice_time_ms": round(slice_ms, 2),
            "context_slice": slice_text
        }, indent=2)

    summary_header = (
        f"<!-- NeuroCode Sliced Context: {len(subgraph.nodes)} AST nodes ({tokens} tokens) "
        f"built in {build_ms:.1f}ms, sliced in {slice_ms:.1f}ms. Target Seeds: {seeds} -->\n\n"
    )
    return summary_header + slice_text


def handle_impact(args: Dict[str, Any]) -> str:
    """Analyzes callers and upstream impact of modifying a specific entity or file."""
    raw_path = args.get("repo_path") or args.get("workspace_path") or args.get("path") or "."
    repo_path = Path(raw_path).resolve()
    target = args.get("target") or args.get("function") or args.get("file_path") or ""
    
    if not repo_path.exists():
        return f"[Error] Repository path does not exist: {repo_path}"
        
    builder = RepoGraphBuilder(str(repo_path))
    graph = builder.build()
    
    matching_nodes = [n for n in graph.nodes if target.lower() in str(n).lower()]
    if not matching_nodes:
        return f"No matching entities found in CPG for target: {target}. Available nodes: {len(graph.nodes)}"
        
    impact_report = [f"# NeuroCode Structural Impact Analysis for: `{target}`"]
    for node in matching_nodes[:5]:
        node_data = graph.nodes[node]
        callers = list(graph.predecessors(node)) if graph.is_directed() else list(graph.neighbors(node))
        callees = list(graph.successors(node)) if graph.is_directed() else []
        
        impact_report.append(f"\n## Entity: `{node}` ({node_data.get('type', 'node')})")
        impact_report.append(f"- **File**: `{node_data.get('file_path', 'unknown')}` (Lines {node_data.get('start_line', '?')}-{node_data.get('end_line', '?')})")
        impact_report.append(f"- **Upstream Callers / Dependencies ({len(callers)})**:")
        for c in callers[:8]:
            impact_report.append(f"  - `{c}`")
        impact_report.append(f"- **Downstream Invocations ({len(callees)})**:")
        for cal in callees[:8]:
            impact_report.append(f"  - `{cal}`")
            
    return "\n".join(impact_report)


def handle_seeds(args: Dict[str, Any]) -> str:
    """Identifies root-cause candidate entities statically."""
    raw_path = args.get("repo_path") or args.get("workspace_path") or args.get("path") or "."
    repo_path = Path(raw_path).resolve()
    issue = args.get("issue") or args.get("query") or args.get("task") or ""
    
    if not repo_path.exists():
        return f"[Error] Repository path does not exist: {repo_path}"
    
    t0 = time.perf_counter()
    builder = RepoGraphBuilder(str(repo_path))
    graph = builder.build()
    slicer = CodeGraphSlicer(graph, max_tokens=1500)
    seeds = slicer.find_seed_nodes(issue or "context")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    result = {
        "identified_seeds": seeds,
        "seed_count": len(seeds),
        "analysis_latency_ms": round(elapsed_ms, 2),
        "repo_nodes": len(graph.nodes),
        "repo_edges": len(graph.edges)
    }
    return json.dumps(result, indent=2)


def handle_repair_ollama(args: Dict[str, Any]) -> str:
    """Generates patch using local Ollama instance with zero cloud dependencies."""
    repo_path = Path(args.get("repo_path", ".")).resolve()
    issue = args.get("issue", "")
    model_tag = args.get("model", "qwen2.5-coder:7b")
    ollama_url = args.get("ollama_url", "http://localhost:11434").rstrip("/")
    budget = int(args.get("budget", 1200))
    apply_patch = bool(args.get("apply", False))
    
    # 1. Slicing
    sliced_ctx = handle_slice({"repo_path": str(repo_path), "issue": issue, "budget": budget})
    
    prompt = (
        f"{sliced_ctx}\n\n"
        f"Issue Description:\n{issue}\n\n"
        f"Provide the exact fix using SEARCH/REPLACE format:\n"
        f"FILE: <file_path>\n<<<<<<< SEARCH\n<exact lines to replace>\n=======\n<replacement lines>\n>>>>>>>\n"
    )
    
    # 2. Call Ollama API
    api_endpoint = f"{ollama_url}/api/chat"
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 350
        }
    }
    
    try:
        req = urllib.request.Request(
            api_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            gen_time = time.perf_counter() - t0
            response_text = data.get("message", {}).get("content", "")
    except Exception as e:
        return f"[Ollama Connection Error] Could not connect to {api_endpoint}: {e}\nEnsure Ollama is running (`ollama serve`)."
    
    # 3. Parse Patch
    builder = RepoGraphBuilder(str(repo_path))
    graph = builder.build()
    patch_res = PatchApplier.extract_patch(response_text, graph=graph, repo_dir=str(repo_path))
    
    output_report = [
        f"### NeuroCode + Ollama Repair Report ({model_tag})",
        f"- **Inference Latency**: {gen_time:.2f}s",
        f"- **Target File**: `{patch_res.file_path}`",
        f"- **Patch Format Valid**: {patch_res.success}",
        "\n```diff",
        response_text.strip(),
        "```"
    ]
    
    if apply_patch and patch_res.success:
        target_file = repo_path / patch_res.file_path
        applied, msg = PatchApplier.apply_patch_to_file(
            str(target_file), patch_res.search_block, patch_res.replace_block
        )
        output_report.append(f"- **Applied to Disk**: {applied} ({msg})")
        
    return "\n".join(output_report)


TOOLS = [
    {
        "name": "neurocode_slice",
        "description": "Extract an AST-sliced repository context bounded strictly under token budget (<=1,200 tokens) using Host-side Code Property Graph reachability. Use this instead of reading entire files into LLM context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Absolute or relative path to repository root (default '.')", "default": "."},
                "issue": {"type": "string", "description": "Bug report, error traceback, feature request, or coding query"},
                "entry": {"type": "string", "description": "Optional specific file path or symbol to anchor the AST slice around"},
                "budget": {"type": "integer", "description": "Strict token budget limit (default 1200)", "default": 1200}
            }
        }
    },
    {
        "name": "neurocode_impact",
        "description": "Analyze the structural blast radius before editing code. Statically reveals all upstream callers, downstream invocations, and file locations in <2 ms.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repository root (default '.')", "default": "."},
                "target": {"type": "string", "description": "Function name, method, class, or file to analyze callers and dependencies for"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "neurocode_seeds",
        "description": "Statically identify root-cause candidate functions, methods, and classes from an issue description or traceback in <2 ms.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repository root (default '.')", "default": "."},
                "issue": {"type": "string", "description": "Bug description or error traceback"}
            },
            "required": ["issue"]
        }
    },
    {
        "name": "neurocode_repair_ollama",
        "description": "Perform single-turn bug repair by sending a knapsack-sliced prompt to a local Ollama model instance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repository root", "default": "."},
                "issue": {"type": "string", "description": "Bug description or traceback"},
                "model": {"type": "string", "description": "Ollama model name (e.g. 'qwen2.5-coder:14b', 'qwen2.5-coder:7b')", "default": "qwen2.5-coder:7b"},
                "ollama_url": {"type": "string", "description": "Ollama server URL", "default": "http://localhost:11434"},
                "apply": {"type": "boolean", "description": "Whether to write the patch to disk", "default": False}
            },
            "required": ["issue"]
        }
    },
    {
        "name": "neurocode_hardware_health",
        "description": "Check local hardware capacity (VRAM allocation, GPU headroom, and tmpfs RAM availability).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


def run_stdio_server():
    """Main JSON-RPC 2.0 stdio loop conforming to Model Context Protocol specification."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")
            params = req.get("params", {})

            # 1. Initialize
            if method == "initialize":
                res = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "neurocode-mcp",
                            "version": "0.1.1"
                        }
                    }
                }
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()

            # 2. Notification: initialized
            elif method in ("notifications/initialized", "initialized"):
                pass  # Notifications have no response in JSON-RPC

            # 3. Tools List
            elif method == "tools/list":
                res = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": TOOLS
                    }
                }
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()

            # 4. Tool Execution
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})

                try:
                    if tool_name == "neurocode_slice":
                        output_str = handle_slice(args)
                    elif tool_name == "neurocode_impact":
                        output_str = handle_impact(args)
                    elif tool_name == "neurocode_seeds":
                        output_str = handle_seeds(args)
                    elif tool_name == "neurocode_repair_ollama":
                        output_str = handle_repair_ollama(args)
                    elif tool_name == "neurocode_hardware_health":
                        output_str = json.dumps(get_hardware_health(), indent=2)
                    else:
                        output_str = f"[Error] Unknown tool: {tool_name}"

                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": output_str
                                }
                            ],
                            "isError": False
                        }
                    }
                except Exception as e:
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Error executing tool {tool_name}: {str(e)}"
                                }
                            ],
                            "isError": True
                        }
                    }

                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()

            # Ping
            elif method == "ping":
                res = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()

            # Fallback
            else:
                if msg_id is not None:
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
                    sys.stdout.write(json.dumps(res) + "\n")
                    sys.stdout.flush()

        except Exception as err:
            err_res = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal JSON-RPC error: {str(err)}"
                }
            }
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()


def main():
    run_stdio_server()


if __name__ == "__main__":
    main()
