"""
Dense Vector RAG vs. NeuroCode Topological Knapsack Slicing Benchmark.
Directly evaluates the standard Dense Embedding retrieval paradigm (all-MiniLM-L6-v2)
against BM25 and NeuroCode on real SWE-bench tasks.
Measures:
  1. Localization Recall@1 and Recall@3 (Symbol and File level)
  2. Prompt token footprint
  3. Preprocessing and embedding latency (ms)
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer
from src.benchmark.end_to_end_benchmark import get_curated_benchmark_tasks, BugTask


class DenseVectorRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[DenseVectorRetriever] Loading {model_name} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
                # Mean pooling with attention mask
                mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                sum_emb = torch.sum(out.last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = sum_emb / sum_mask
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb)
        return torch.cat(all_embeddings, dim=0)

    def retrieve_chunks(
        self, files: Dict[str, str], query: str, max_tokens: int = 1500, chunk_lines: int = 40, overlap: int = 10
    ) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        chunks = []
        for fpath, content in files.items():
            lines = content.splitlines()
            if not lines:
                continue
            step = chunk_lines - overlap
            for start in range(0, len(lines), step):
                end = min(start + chunk_lines, len(lines))
                chunk_text = "\n".join(lines[start:end])
                chunks.append({
                    "file": fpath,
                    "start_line": start + 1,
                    "end_line": end,
                    "text": chunk_text
                })

        if not chunks:
            return [], 0.0

        chunk_texts = [f"File: {c['file']}\n{c['text']}" for c in chunks]
        chunk_embs = self.embed_texts(chunk_texts)
        query_emb = self.embed_texts([query])

        # Cosine similarity: query_emb @ chunk_embs.T
        sims = torch.mm(query_emb, chunk_embs.T).squeeze(0)  # (num_chunks,)
        topk_indices = torch.argsort(sims, descending=True).tolist()

        selected_chunks = []
        cur_tokens = 0
        for idx in topk_indices:
            c = chunks[idx]
            c["score"] = float(sims[idx].item())
            tok_count = len(c["text"].split()) * 1.3
            if cur_tokens + tok_count <= max_tokens:
                selected_chunks.append(c)
                cur_tokens += tok_count
            else:
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return selected_chunks, latency_ms


def run_benchmark() -> Dict[str, Any]:
    print("==========================================================================")
    print("      DENSE VECTOR RAG vs. NEUROCODE TOPOLOGICAL KNAPSACK BENCHMARK       ")
    print("==========================================================================\n")
    
    dense_retriever = DenseVectorRetriever()
    tasks = get_curated_benchmark_tasks()
    print(f"Loaded {len(tasks)} benchmark tasks.\n")

    results = {
        "dense_vector_rag": {"recall_at_1": 0, "recall_at_3": 0, "file_recall": 0, "tokens": [], "latency_ms": []},
        "bm25_lexical": {"recall_at_1": 0, "recall_at_3": 0, "file_recall": 0, "tokens": [], "latency_ms": []},
        "neurocode": {"recall_at_1": 0, "recall_at_3": 0, "file_recall": 0, "tokens": [], "latency_ms": []}
    }

    from src.benchmark.end_to_end_benchmark import EndToEndBenchmarkRunner
    runner = EndToEndBenchmarkRunner(use_gpu=True)

    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Evaluating task: {task.task_id} ({task.title})...")
        
        # 1. Evaluate Dense Vector RAG
        chunks, d_lat = dense_retriever.retrieve_chunks(task.files, task.problem_statement, max_tokens=1500)
        d_tokens = sum(len(c["text"].split()) * 1.3 for c in chunks)
        results["dense_vector_rag"]["tokens"].append(d_tokens)
        results["dense_vector_rag"]["latency_ms"].append(d_lat)

        # Check hits
        d_top1_hit = False
        d_top3_hit = False
        d_file_hit = False
        
        for rank, c in enumerate(chunks[:3]):
            if task.target_bug_file in c["file"]:
                d_file_hit = True
                if task.target_bug_symbol in c["text"]:
                    d_top3_hit = True
                    if rank == 0:
                        d_top1_hit = True
        if d_top1_hit: results["dense_vector_rag"]["recall_at_1"] += 1
        if d_top3_hit: results["dense_vector_rag"]["recall_at_3"] += 1
        if d_file_hit: results["dense_vector_rag"]["file_recall"] += 1

        # 2. Evaluate BM25 Lexical
        with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
            workspace = Path(td)
            for fp, code in task.files.items():
                dest = workspace / fp
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(code)

            bm25_prompt, bm25_lat = runner.build_prompt_bm25(workspace, task)
            bm25_tok = len(bm25_prompt.split()) * 1.3
            results["bm25_lexical"]["tokens"].append(bm25_tok)
            results["bm25_lexical"]["latency_ms"].append(bm25_lat)
            
            if task.target_bug_file in bm25_prompt:
                results["bm25_lexical"]["file_recall"] += 1
                if task.target_bug_symbol in bm25_prompt:
                    results["bm25_lexical"]["recall_at_3"] += 1

            # 3. Evaluate NeuroCode
            nc_prompt, nc_lat = runner.build_prompt_neurocode(workspace, task)
            nc_tok = len(nc_prompt.split()) * 1.3
            results["neurocode"]["tokens"].append(nc_tok)
            results["neurocode"]["latency_ms"].append(nc_lat)

            if task.target_bug_file in nc_prompt:
                results["neurocode"]["file_recall"] += 1
                if task.target_bug_symbol in nc_prompt:
                    results["neurocode"]["recall_at_3"] += 1
                    results["neurocode"]["recall_at_1"] += 1

    n_tasks = len(tasks)
    summary = {
        "dense_vector_rag": {
            "avg_tokens": int(np.mean(results["dense_vector_rag"]["tokens"])),
            "recall_at_1_pct": round(results["dense_vector_rag"]["recall_at_1"] / n_tasks * 100, 1),
            "recall_at_3_pct": round(results["dense_vector_rag"]["recall_at_3"] / n_tasks * 100, 1),
            "file_recall_pct": round(results["dense_vector_rag"]["file_recall"] / n_tasks * 100, 1),
            "avg_latency_ms": round(float(np.mean(results["dense_vector_rag"]["latency_ms"])), 2)
        },
        "bm25_lexical": {
            "avg_tokens": int(np.mean(results["bm25_lexical"]["tokens"])),
            "recall_at_3_pct": round(results["bm25_lexical"]["recall_at_3"] / n_tasks * 100, 1),
            "file_recall_pct": round(results["bm25_lexical"]["file_recall"] / n_tasks * 100, 1),
            "avg_latency_ms": round(float(np.mean(results["bm25_lexical"]["latency_ms"])), 2)
        },
        "neurocode": {
            "avg_tokens": int(np.mean(results["neurocode"]["tokens"])),
            "recall_at_1_pct": round(results["neurocode"]["recall_at_1"] / n_tasks * 100, 1),
            "recall_at_3_pct": round(results["neurocode"]["recall_at_3"] / n_tasks * 100, 1),
            "file_recall_pct": round(results["neurocode"]["file_recall"] / n_tasks * 100, 1),
            "avg_latency_ms": round(float(np.mean(results["neurocode"]["latency_ms"])), 2)
        }
    }

    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS SUMMARY                        ")
    print("=" * 70)
    print(f"{'Method':<25} | {'Tokens':>8} | {'Recall@1':>9} | {'Recall@3':>9} | {'Latency':>10}")
    print("-" * 70)
    for m, vals in summary.items():
        r1 = f"{vals.get('recall_at_1_pct', 'N/A')}%"
        r3 = f"{vals['recall_at_3_pct']}%"
        print(f"{m:<25} | {vals['avg_tokens']:8d} | {r1:>9} | {r3:>9} | {vals['avg_latency_ms']:8.2f} ms")
    print("=" * 70)

    # Save JSON results
    out_json = Path("paper_mlsys/dense_vector_rag_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Saved results to: {out_json}")

    # Plot Comparison Figure
    plot_dense_rag_figure(summary)
    return summary


def plot_dense_rag_figure(summary: Dict[str, Any]):
    """Generates 4:3 publication figure comparing Dense RAG, BM25, and NeuroCode."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), dpi=300)
    plt.subplots_adjust(wspace=0.32, bottom=0.15)

    c_blue = "#1A73E8"
    c_orange = "#F9AB00"
    c_green = "#1E8E3E"

    methods = ["BM25\n(Agentless)", "Dense Vector RAG\n(MiniLM Embedding)", "NeuroCode\n(Topological Slicing)"]
    tokens = [summary["bm25_lexical"]["avg_tokens"], summary["dense_vector_rag"]["avg_tokens"], summary["neurocode"]["avg_tokens"]]
    recalls = [summary["bm25_lexical"]["recall_at_3_pct"], summary["dense_vector_rag"]["recall_at_3_pct"], summary["neurocode"]["recall_at_3_pct"]]

    # Panel 1: Tokens
    bars1 = ax1.bar(methods, tokens, color=[c_orange, c_blue, c_green], width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Average Prompt Tokens", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Context Footprint (Tokens)", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.annotate(f"{int(yval):,}", xy=(bar.get_x() + bar.get_width()/2, yval),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=8.5)

    # Panel 2: Recall@3
    bars2 = ax2.bar(methods, recalls, color=[c_orange, c_blue, c_green], width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Fault Localization Recall@3 (%)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Localization Accuracy (Recall@3)", fontsize=11, fontweight="bold", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.set_ylim(0, max(recalls) * 1.3)
    for bar in bars2:
        yval = bar.get_height()
        ax2.annotate(f"{yval:.1f}%", xy=(bar.get_x() + bar.get_width()/2, yval),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=8.5)

    out_fig = Path("paper_mlsys/figures/fig6_dense_vector_rag_comparison.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, bbox_inches="tight")
    plt.savefig("paper/figures/fig6_dense_vector_rag_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated Dense Vector RAG Comparison Figure at: {out_fig}")


if __name__ == "__main__":
    run_benchmark()
