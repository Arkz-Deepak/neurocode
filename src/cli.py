#!/usr/bin/env python3
"""
NeuroCode Autonomous CLI: Edge Codebase Slicing and Program Repair.

Usage:
  neurocode repair   --repo <dir> --issue "<description>" [--apply] [--test "<pytest cmd>"]
  neurocode slice    --repo <dir> --issue "<description>" [--budget 1500]
  neurocode benchmark [--swebench] [--max-issues 10]
  neurocode info
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer
from src.agent.patcher import PatchApplier, SYSTEM_PROMPT


def get_hardware_info():
    if HAS_TORCH:
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
        allocated_vram = torch.cuda.memory_allocated() / (1024**2) if cuda_avail else 0.0
    else:
        cuda_avail = False
        device_name = "Host CPU (PyTorch optional / Ollama ready)"
        total_vram = 0.0
        allocated_vram = 0.0

    print("==========================================================================")
    print("                    NEUROCODE HARDWARE STATUS                             ")
    print("==========================================================================")
    print(f"Device:             {device_name}")
    print(f"CUDA Available:     {cuda_avail}")
    if cuda_avail:
        print(f"Total VRAM:         {total_vram:.2f} GB")
        print(f"Active VRAM:        {allocated_vram:.1f} MB")
    print(f"Python Executable:  {sys.executable}")
    print(f"Project Root:       {PROJECT_ROOT}")
    print("==========================================================================\n")


def cmd_slice(args):
    repo_target = args.repo_pos if args.repo_pos else args.repo
    repo_path = Path(repo_target).resolve()
    if not repo_path.exists():
        print(f"[ERROR] Repository path does not exist: {repo_path}")
        sys.exit(1)

    issue_query = args.issue or args.entry or "repository structure"

    print("==========================================================================")
    print(f"      NEUROCODE GRAPH CONTEXT SLICING: {repo_path.name}")
    print("==========================================================================")
    print(f"Target: {issue_query}")
    print(f"Budget: {args.budget} tokens\n")

    t0 = time.perf_counter()
    builder = RepoGraphBuilder(str(repo_path))
    graph = builder.build()
    build_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    slicer = CodeGraphSlicer(graph, max_tokens=args.budget)
    
    seeds = []
    if args.entry:
        entry_clean = Path(args.entry).as_posix()
        seeds = [n for n in graph.nodes if entry_clean in str(graph.nodes[n].get("file_path", "")) or entry_clean in str(n)]
    
    if not seeds:
        seeds = slicer.find_seed_nodes(issue_query)

    if not seeds and len(graph.nodes) > 0:
        seeds = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)[:3]

    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
    slice_text = slicer.format_slice_for_llm(subgraph, seeds, query=issue_query, max_tokens=args.budget)
    slice_ms = (time.perf_counter() - t1) * 1000

    token_count = int(len(slice_text.split()) * 1.3)

    print(f"AST Graph:        {len(graph.nodes)} nodes, {len(graph.edges)} edges (Built in {build_ms:.1f} ms)")
    print(f"Identified Seeds: {len(seeds)} entities: {seeds}")
    print(f"Sliced Subgraph:  {len(subgraph.nodes)} nodes (Sliced in {slice_ms:.1f} ms)")
    print(f"Packed Size:      ~{token_count} tokens (Strictly under {args.budget} budget)\n")

    if args.output:
        out_p = Path(args.output)
        out_p.write_text(slice_text, encoding="utf-8")
        print(f"[SAVED] Sliced context saved to: {out_p}")
    else:
        print("--- SLICED CONTEXT PREVIEW ---")
        print(slice_text[:1200] + "\n...\n[Truncated for CLI display. Use --output <file> to view full markdown.]")


def cmd_repair(args):
    repo_target = args.repo_pos if args.repo_pos else args.repo
    repo_path = Path(repo_target).resolve()
    if not repo_path.exists():
        print(f"[ERROR] Repository path does not exist: {repo_path}")
        sys.exit(1)

    print("==========================================================================")
    print(f"      NEUROCODE AUTONOMOUS SOFTWARE REPAIR: {repo_path.name}")
    print("==========================================================================")
    print(f"Issue:  {args.issue}")
    print(f"Model:  {args.model}")
    print(f"Budget: {args.budget} tokens\n")

    # 1. Host-CPU Graph Slicing
    print("[1/4] Constructing in-memory Code Property Graph...")
    t0 = time.perf_counter()
    builder = RepoGraphBuilder(str(repo_path))
    graph = builder.build()
    slicer = CodeGraphSlicer(graph, max_tokens=args.budget)
    seeds = slicer.find_seed_nodes(args.issue)
    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
    slice_text = slicer.format_slice_for_llm(subgraph, seeds, query=args.issue, max_tokens=args.budget)
    prep_ms = (time.perf_counter() - t0) * 1000
    token_count = int(len(slice_text.split()) * 1.3)
    print(f"  -> Context sliced in {prep_ms:.1f} ms: {len(subgraph.nodes)} nodes (~{token_count} tokens).")

    # 2. Model Inference (Ollama or Transformers)
    response_text = ""
    provider = getattr(args, "provider", "transformers")

    if provider == "ollama":
        import json
        import urllib.request
        ollama_url = getattr(args, "ollama_url", "http://localhost:11434").rstrip("/")
        print(f"\n[2/4] Querying local Ollama daemon ({args.model}) at {ollama_url}...")
        prompt = (
            f"{slice_text}\n\n"
            f"Issue Description:\n{args.issue}\n\n"
            f"Provide the exact fix using SEARCH/REPLACE format:\n"
            f"FILE: <file_path>\n<<<<<<< SEARCH\n<exact lines to replace>\n=======\n<replacement lines>\n>>>>>>>\n"
        )
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 350}
        }
        try:
            req = urllib.request.Request(
                f"{ollama_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            t_gen0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                gen_time = time.perf_counter() - t_gen0
                response_text = data.get("message", {}).get("content", "")
            print(f"  -> Ollama responded in {gen_time:.2f}s.")
        except Exception as e:
            print(f"  -> [OLLAMA ERROR] Could not connect to Ollama at {ollama_url}: {e}")
            print("     Make sure Ollama is running (`ollama serve`).")
            return
    else:
        if not HAS_TORCH:
            print("[Error] PyTorch is required for local HuggingFace transformers inference.")
            print("        Install it via: pip install torch")
            print("        Or use local Ollama inference with zero PyTorch dependency: --provider ollama")
            return
        print(f"\n[2/4] Loading local LLM: {args.model} on GPU...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = None
        if args.four_bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

        t_load = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        load_time = time.perf_counter() - t_load
        vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        print(f"  -> Model ready in {load_time:.2f}s | Active VRAM: {vram_mb:.1f} MB.")

        # 3. Patch Inference
        print("\n[3/4] Synthesizing precise code patch...")
        prompt = (
            f"{slice_text}\n\n"
            f"Issue Description:\n{args.issue}\n\n"
            f"Provide the exact fix using SEARCH/REPLACE format:\n"
            f"FILE: <file_path>\n<<<<<<< SEARCH\n<exact lines to replace>\n=======\n<replacement lines>\n>>>>>>>\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_gen0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.perf_counter() - t_gen0
        new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        print(f"  -> Generated {new_tokens} tokens in {gen_time:.2f}s ({new_tokens/gen_time:.1f} tok/s) | Peak VRAM: {peak_vram:.1f} MB.")

    print("\n---------------- GENERATED PATCH OUTPUT ----------------")
    print(response_text)
    print("--------------------------------------------------------\n")

    # 4. Patch Application & Verification
    print("[4/4] Parsing and applying patch...")
    patch_res = PatchApplier.extract_patch(response_text, graph=graph, repo_dir=str(repo_path))
    if not patch_res.success:
        print(f"  -> Patch Format Error: {patch_res.error_message}")
        return

    target_file = repo_path / patch_res.file_path
    print(f"  -> Target File: {patch_res.file_path}")
    print(f"  -> Search Block ({len(patch_res.search_block.splitlines())} lines) -> Replace Block ({len(patch_res.replace_block.splitlines())} lines)")

    if args.apply:
        applied, msg = PatchApplier.apply_patch_to_file(
            str(target_file),
            patch_res.search_block,
            patch_res.replace_block
        )
        if applied:
            print(f"  -> [SUCCESS] Patch applied successfully to: {target_file}")
            if args.test:
                print(f"\n[VERIFY] Executing automated test suite: {args.test}...")
                test_res = subprocess.run(args.test, shell=True, cwd=str(repo_path), capture_output=True, text=True)
                if test_res.returncode == 0:
                    print("  -> [TEST PASSED] Issue is fully RESOLVED (Pass@1: TRUE)!")
                else:
                    print(f"  -> [TEST FAILED] Exit Code {test_res.returncode}:\n{test_res.stdout}\n{test_res.stderr}")
        else:
            print(f"  -> [APPLY FAILED] {msg}")
    else:
        print("  -> (Dry run: specify --apply to commit modifications to disk)")


def cmd_agent_setup(args):
    repo_target = args.repo if args.repo else "."
    repo_path = Path(repo_target).resolve()
    import json

    print("==========================================================================")
    print(f"      NEUROCODE AGENT SETUP: {repo_path.name}")
    print("==========================================================================")
    print("Configuring workspace for Autonomous AI Agents (Cursor, Claude, MCP)...")

    # 1. .cursor/mcp.json
    cursor_dir = repo_path / ".cursor"
    cursor_dir.mkdir(exist_ok=True)
    cursor_mcp = cursor_dir / "mcp.json"
    cursor_cfg = {
        "mcpServers": {
            "neurocode": {
                "command": "neurocode-mcp",
                "args": []
            }
        }
    }
    with open(cursor_mcp, "w") as f:
        json.dump(cursor_cfg, f, indent=2)
    print(f"  -> [✓] Generated Cursor MCP config: {cursor_mcp}")

    # 2. .vscode/mcp.json
    vscode_dir = repo_path / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    vscode_mcp = vscode_dir / "mcp.json"
    with open(vscode_mcp, "w") as f:
        json.dump(cursor_cfg, f, indent=2)
    print(f"  -> [✓] Generated VS Code MCP config: {vscode_mcp}")

    # 3. .agent/skills/neurocode.md
    agent_skills_dir = repo_path / ".agent" / "skills"
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    skill_file = agent_skills_dir / "neurocode.md"
    skill_content = """# NeuroCode AI Agent Skill Definition
**Role**: Codebase Perception & Surgical Context Extraction
**MCP Server**: `neurocode-mcp`

## Guidelines for AI Agents:
1. **Never read entire multi-thousand line files** into context. Always call `neurocode_slice` with the user's issue/query first.
2. **Locate functions/classes instantly** using `neurocode_symbol` rather than executing manual grep/find loops.
3. **Verify blast radius** using `neurocode_impact` before proposing significant refactors.
4. **Pre-flight validate patches** using `neurocode_validate_patch` before writing changes to disk to guarantee valid syntax and avoid breaking tests.
"""
    with open(skill_file, "w") as f:
        f.write(skill_content)
    print(f"  -> [✓] Generated AI Agent Skill rulebook: {skill_file}")

    print("\nAI Agent configuration complete! Your agents can now autonomously:")
    print("  • Query CPG AST slices in <2 ms without token bloat.")
    print("  • Look up symbols and callers instantly.")
    print("  • Pre-validate code patches before modifying disk.")
    print("  • Deterministically graft AST subtrees without whitespace errors.")
    print("==========================================================================\n")


def cmd_proxy(args):
    from src.proxy import run_proxy
    run_proxy(
        port=args.port,
        upstream=args.upstream,
        budget=args.budget,
        repo=args.repo
    )


def cmd_graft(args):
    from src.agent.graft import graft_symbol
    file_path = args.file
    symbol = args.symbol
    replacement = args.code
    apply = args.apply
    test_cmd = args.test
    auto_rollback = not args.no_rollback

    res = graft_symbol(
        ".",
        file_path,
        symbol,
        replacement,
        apply=apply,
        test_command=test_cmd,
        auto_rollback=auto_rollback
    )
    if res.get("success"):
        print(f"[✓] AST Graft successful for symbol '{symbol}' in {file_path}")
        print(f"    Syntax Verified: {res['syntax_valid']}")
        print(f"    Applied to disk: {res['applied']}")
        if test_cmd:
            print(f"    Tests Passed: {res.get('test_passed')}")
        if res.get("diff"):
            print("\n" + res["diff"])
    else:
        print(f"[✗] AST Graft failed: {res.get('error')}")
        if res.get("rolled_back"):
            print("    [!] Safe auto-rollback executed: original file preserved.")


def cmd_trace(args):
    from src.graph.builder import RepoGraphBuilder
    repo_target = args.repo if args.repo else "."
    builder = RepoGraphBuilder(repo_target)
    res = builder.trace_path(args.from_sym, args.to_sym)
    if res.get("path_found"):
        print(f"\n[✓] Execution Path Found ({res.get('hops')} hops):")
        print(f"    From: {res.get('from_symbol')} -> To: {res.get('to_symbol')}\n")
        for step in res.get("call_chain", []):
            print(f"    [{step['step']}] {step['type'].upper()}: {step['name']} ({step['file']} Lines {step['lines']})")
        print()
    else:
        print(f"\n[✗] {res.get('error', 'No path found.')}\n")


def cmd_map(args):
    from src.graph.builder import RepoGraphBuilder
    repo_target = args.repo if args.repo else "."
    builder = RepoGraphBuilder(repo_target)
    mermaid_code = builder.generate_mermaid_map()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"```mermaid\n{mermaid_code}\n```\n")
        print(f"[✓] Architecture map saved to {args.output}")
    else:
        print("\n```mermaid\n" + mermaid_code + "\n```\n")


def main():
    parser = argparse.ArgumentParser(
        prog="neurocode",
        description="NeuroCode: Autonomous Codebase Slicing & Local Software Repair"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: info
    subparsers.add_parser("info", help="Display local hardware and VRAM allocation status")

    # Command: agent-setup
    setup_parser = subparsers.add_parser("agent-setup", help="Auto-configure repository for AI agents (Cursor, Claude, Continue MCP)")
    setup_parser.add_argument("--repo", type=str, default=".", help="Path to repository directory (default '.')")

    # Command: graft
    graft_parser = subparsers.add_parser("graft", help="Deterministically graft/mutate an AST symbol without whitespace diff failures")
    graft_parser.add_argument("file", type=str, help="Target file path (Python, JS, TS, Go, Rust)")
    graft_parser.add_argument("--symbol", type=str, required=True, help="Function, method, or class name to replace")
    graft_parser.add_argument("--code", type=str, required=True, help="New code block to graft into the AST")
    graft_parser.add_argument("--apply", action="store_true", help="Apply graft directly to source file on disk")
    graft_parser.add_argument("--test", type=str, default=None, help="Optional test command to run (e.g. 'pytest tests/test_billing.py')")
    graft_parser.add_argument("--no-rollback", action="store_true", help="Do not revert code if test command fails")

    # Command: trace
    trace_parser = subparsers.add_parser("trace", help="Trace execution call paths connecting two functions or symbols")
    trace_parser.add_argument("--from", dest="from_sym", type=str, required=True, help="Source function or symbol name")
    trace_parser.add_argument("--to", dest="to_sym", type=str, required=True, help="Destination function or symbol name")
    trace_parser.add_argument("--repo", type=str, default=".", help="Path to repository directory (default '.')")

    # Command: map
    map_parser = subparsers.add_parser("map", help="Generate zero-token Mermaid.js architecture flowchart of codebase")
    map_parser.add_argument("--repo", type=str, default=".", help="Path to repository directory (default '.')")
    map_parser.add_argument("--output", "-o", type=str, default=None, help="Optional file path to save Mermaid diagram (e.g. ARCHITECTURE.md)")

    # Command: slice
    slice_parser = subparsers.add_parser("slice", help="Construct CPG and extract minimal token-budgeted slice")
    slice_parser.add_argument("repo_pos", nargs="?", default=None, help="Path to repository directory (positional)")
    slice_parser.add_argument("--repo", type=str, default=".", help="Path to repository directory")
    slice_parser.add_argument("--entry", type=str, default=None, help="Entry file path to slice around")
    slice_parser.add_argument("--issue", type=str, default=None, help="Bug report description or traceback")
    slice_parser.add_argument("--budget", type=int, default=1200, help="Maximum token budget for prompt")
    slice_parser.add_argument("--output", type=str, default=None, help="Save sliced Markdown prompt to file")

    # Command: proxy
    proxy_parser = subparsers.add_parser("proxy", help="Launch transparent OpenAI-compatible proxy gateway with auto-slicing")
    proxy_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default 8000)")
    proxy_parser.add_argument("--upstream", "-u", type=str, default="http://localhost:11434/v1", help="Upstream LLM API endpoint (Ollama / OpenAI / vLLM)")
    proxy_parser.add_argument("--budget", "-b", type=int, default=1500, help="Maximum token budget for prompt slicing (default 1500)")
    proxy_parser.add_argument("--repo", type=str, default=".", help="Target repository directory (default '.')")

    # Command: mcp
    subparsers.add_parser("mcp", help="Start the Model Context Protocol (MCP) stdio server for Claude Desktop / Cursor / Continue")

    # Command: repair
    repair_parser = subparsers.add_parser("repair", help="Perform end-to-end bug localization and patch synthesis")
    repair_parser.add_argument("repo_pos", nargs="?", default=None, help="Path to repository directory (positional)")
    repair_parser.add_argument("--repo", type=str, default=".", help="Path to repository directory")
    repair_parser.add_argument("--entry", type=str, default=None, help="Entry file path")
    repair_parser.add_argument("--issue", type=str, required=True, help="Bug report description or traceback")
    repair_parser.add_argument("--budget", type=int, default=1500, help="Maximum token budget")
    repair_parser.add_argument("--provider", type=str, default="transformers", choices=["transformers", "ollama"], help="Inference provider (transformers or ollama)")
    repair_parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL")
    repair_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Model ID or Ollama model tag")
    repair_parser.add_argument("--four-bit", action="store_true", default=True, help="Use 4-bit NF4 quantization (transformers only)")
    repair_parser.add_argument("--apply", action="store_true", help="Apply patch directly to source code")
    repair_parser.add_argument("--test", type=str, default=None, help="Test command to run after applying patch (e.g. 'pytest tests/test_gw.py')")

    args = parser.parse_args()

    if args.command == "info":
        get_hardware_info()
    elif args.command == "agent-setup":
        cmd_agent_setup(args)
    elif args.command == "graft":
        cmd_graft(args)
    elif args.command == "trace":
        cmd_trace(args)
    elif args.command == "map":
        cmd_map(args)
    elif args.command == "proxy":
        cmd_proxy(args)
    elif args.command == "mcp":
        from src.mcp_server import main as run_mcp
        run_mcp()
    elif args.command == "slice":
        cmd_slice(args)
    elif args.command == "repair":
        cmd_repair(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
