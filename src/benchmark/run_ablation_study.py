"""
NeuroCode Ablation Study & Token-Budget Scaling Sweep.
Directly addresses Academic Review Gap 4 & 5 (Section 5: Experiment B & C).

Evaluates:
  Variant 1: Lexical Seed Matching Only (No graph traversal)
  Variant 2: Seed + Import Edges Only (Architectural hierarchy)
  Variant 3: Seed + Call Edges Only (Execution dependency reachability)
  Variant 4: Full Heterogeneous Code Property Graph (Imports + Calls + Inherits)
  Variant 5: Full CPG + Priority-Ranked Knapsack Budget Packing (NeuroCode Proposed)

Across Token Budgets:
  - 512 tokens
  - 1,024 tokens
  - 1,500 tokens (Default NeuroCode Operating Point)
  - 3,000 tokens
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer
from src.benchmark.end_to_end_benchmark import get_curated_benchmark_tasks, BugTask


@dataclass
class AblationResult:
    variant_name: str
    target_hit_rate: float
    recall_at_3: float
    avg_tokens: int
    avg_latency_ms: float
    compression_ratio: float


def run_ablation_sweep() -> Dict[str, Any]:
    print("==========================================================================")
    print("             NEUROCODE COMPONENT ABLATION & SCALING STUDY                 ")
    print("==========================================================================\n")

    tasks = get_curated_benchmark_tasks()
    variants = [
        "1. Lexical Seeds Only",
        "2. Seeds + Import Edges",
        "3. Seeds + Call Edges",
        "4. Full CPG (Unconstrained)",
        "5. NeuroCode (CPG + Knapsack Packing)",
    ]

    results_by_variant = {}

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root_dir = Path(td)

        for v_name in variants:
            hit_count = 0
            recalls = []
            tokens_list = []
            latencies = []

            for task in tasks:
                # Setup files
                task_dir = root_dir / task.task_id
                task_dir.mkdir(parents=True, exist_ok=True)
                for rel_path, code in task.files.items():
                    fp = task_dir / rel_path
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(code, encoding="utf-8")

                t0 = time.perf_counter()
                builder = RepoGraphBuilder(str(task_dir))
                graph = builder.build()
                slicer = CodeGraphSlicer(graph, max_tokens=1500)
                seeds = slicer.find_seed_nodes(task.problem_statement)

                if v_name == "1. Lexical Seeds Only":
                    # Only seed nodes, no edge traversal
                    subgraph = graph.subgraph(seeds).copy() if seeds else nx.DiGraph()
                    slice_text = "\n".join([graph.nodes[n].get("code", "") for n in subgraph.nodes])
                elif v_name == "2. Seeds + Import Edges":
                    # Filter graph to only import edges
                    import_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("relation") == "imports"]
                    sub_g = nx.DiGraph(import_edges)
                    for n in seeds:
                        sub_g.add_node(n, **graph.nodes[n])
                    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
                    # Keep only import neighbors
                    valid_nodes = set(seeds)
                    for s in seeds:
                        for neighbor in graph.neighbors(s):
                            edge_data = graph.get_edge_data(s, neighbor, default={})
                            if edge_data.get("relation") == "imports":
                                valid_nodes.add(neighbor)
                    subgraph = graph.subgraph(valid_nodes).copy()
                    slice_text = "\n".join([graph.nodes[n].get("code", "") for n in subgraph.nodes])
                elif v_name == "3. Seeds + Call Edges":
                    # Keep only call neighbors
                    valid_nodes = set(seeds)
                    for s in seeds:
                        for succ in graph.successors(s):
                            edge_data = graph.get_edge_data(s, succ, default={})
                            if edge_data.get("relation") == "calls":
                                valid_nodes.add(succ)
                        for pred in graph.predecessors(s):
                            edge_data = graph.get_edge_data(pred, s, default={})
                            if edge_data.get("relation") == "calls":
                                valid_nodes.add(pred)
                    subgraph = graph.subgraph(valid_nodes).copy()
                    slice_text = "\n".join([graph.nodes[n].get("code", "") for n in subgraph.nodes])
                elif v_name == "4. Full CPG (Unconstrained)":
                    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
                    slice_text = "\n".join([graph.nodes[n].get("code", "") for n in subgraph.nodes])
                else:  # 5. NeuroCode Proposed
                    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
                    slice_text = slicer.format_slice_for_llm(subgraph, seeds, query=task.problem_statement, max_tokens=1500)

                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)

                tokens = int(len(slice_text.split()) * 1.3)
                tokens_list.append(tokens)

                target_hit = task.target_bug_symbol in slice_text and task.target_bug_file in slice_text
                if target_hit:
                    hit_count += 1
                recalls.append(1.0 if target_hit else 0.0)

            hit_rate = (hit_count / len(tasks)) * 100
            mean_tokens = int(np.mean(tokens_list))
            mean_latency = float(np.mean(latencies))

            res = AblationResult(
                variant_name=v_name,
                target_hit_rate=hit_rate,
                recall_at_3=float(np.mean(recalls) * 100),
                avg_tokens=mean_tokens,
                avg_latency_ms=mean_latency,
                compression_ratio=(1.0 - (mean_tokens / 3000.0)) * 100 if mean_tokens < 3000 else 0.0
            )
            results_by_variant[v_name] = res

            print(f"Variant: {v_name:38s} | Hit@1: {hit_rate:5.1f}% | Avg Tokens: {mean_tokens:4d} | Latency: {mean_latency:.2f}ms")

    return results_by_variant


def run_context_budget_sweep() -> Dict[int, Dict[str, float]]:
    print("\n--------------------------------------------------------------------------")
    print("        CONTEXT TOKEN BUDGET SWEEP (512, 1024, 1500, 3000 TOKENS)        ")
    print("--------------------------------------------------------------------------")

    budgets = [512, 1024, 1500, 3000]
    tasks = get_curated_benchmark_tasks()
    budget_results = {}

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root_dir = Path(td)
        for b in budgets:
            hits = 0
            actual_tokens = []
            for task in tasks:
                task_dir = root_dir / f"{b}_{task.task_id}"
                task_dir.mkdir(parents=True, exist_ok=True)
                for rel_path, code in task.files.items():
                    fp = task_dir / rel_path
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(code, encoding="utf-8")

                builder = RepoGraphBuilder(str(task_dir))
                graph = builder.build()
                slicer = CodeGraphSlicer(graph, max_tokens=b)
                seeds = slicer.find_seed_nodes(task.problem_statement)
                subgraph = slicer.extract_subgraph(seeds, k_hops=1)
                slice_text = slicer.format_slice_for_llm(subgraph, seeds, query=task.problem_statement, max_tokens=b)

                tokens = int(len(slice_text.split()) * 1.3)
                actual_tokens.append(tokens)
                if task.target_bug_symbol in slice_text:
                    hits += 1

            hit_rate = (hits / len(tasks)) * 100
            mean_tok = int(np.mean(actual_tokens))
            budget_results[b] = {"hit_rate": hit_rate, "mean_tokens": mean_tok}
            print(f"Budget Cap: {b:4d} tokens | Actual Mean Tokens: {mean_tok:4d} | Target Hit Rate: {hit_rate:5.1f}%")

    return budget_results


if __name__ == "__main__":
    ablation_res = run_ablation_sweep()
    budget_res = run_context_budget_sweep()
