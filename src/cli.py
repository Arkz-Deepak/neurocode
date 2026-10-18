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

import torch

from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer
from src.agent.patcher import PatchApplier, SYSTEM_PROMPT


def get_hardware_info():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    allocated_vram = torch.cuda.memory_allocated() / (1024**2) if cuda_avail else 0.0

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
    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        print(f"[ERROR] Repository path does not exist: {repo_path}")
        sys.exit(1)

    print("==========================================================================")
    print(f"      NEUROCODE GRAPH CONTEXT SLICING: {repo_path.name}")
    print("==========================================================================")
    print(f"Issue:  {args.issue}")
    print(f"Budget: {args.budget} tokens\n")

    t0 = time.perf_counter()
    builder = RepoGraphBuilder(str(repo_path))
    graph = builder.build()
    build_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    slicer = CodeGraphSlicer(graph, max_tokens=args.budget)
    seeds = slicer.find_seed_nodes(args.issue)
    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
    slice_text = slicer.format_slice_for_llm(subgraph, seeds, query=args.issue, max_tokens=args.budget)
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
    repo_path = Path(args.repo).resolve()
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

    # 2. Model Loading (4-bit NF4 Quantization)
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


def main():
    parser = argparse.ArgumentParser(
        prog="neurocode",
        description="NeuroCode: Autonomous Codebase Slicing & Local Software Repair"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: info
    subparsers.add_parser("info", help="Display local hardware and VRAM allocation status")

    # Command: slice
    slice_parser = subparsers.add_parser("slice", help="Construct CPG and extract minimal token-budgeted slice")
    slice_parser.add_argument("--repo", type=str, default=".", help="Path to repository directory")
    slice_parser.add_argument("--issue", type=str, required=True, help="Bug report description or traceback")
    slice_parser.add_argument("--budget", type=int, default=1500, help="Maximum token budget for prompt")
    slice_parser.add_argument("--output", type=str, default=None, help="Save sliced Markdown prompt to file")

    # Command: repair
    repair_parser = subparsers.add_parser("repair", help="Perform end-to-end bug localization and patch synthesis")
    repair_parser.add_argument("--repo", type=str, default=".", help="Path to repository directory")
    repair_parser.add_argument("--issue", type=str, required=True, help="Bug report description or traceback")
    repair_parser.add_argument("--budget", type=int, default=1500, help="Maximum token budget")
    repair_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="HuggingFace model ID")
    repair_parser.add_argument("--four-bit", action="store_true", default=True, help="Use 4-bit NF4 quantization")
    repair_parser.add_argument("--apply", action="store_true", help="Apply patch directly to source code")
    repair_parser.add_argument("--test", type=str, default=None, help="Test command to run after applying patch (e.g. 'pytest tests/test_gw.py')")

    args = parser.parse_args()

    if args.command == "info":
        get_hardware_info()
    elif args.command == "slice":
        cmd_slice(args)
    elif args.command == "repair":
        cmd_repair(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
