"""
Evaluation harness comparing:
1. Raw Context Dump (Naive Baseline)
2. BM25 Text Retrieval (Agentless Baseline)
3. NeuroCode Host-CPU Graph Slicer (Our Proposed Method)
"""

import time
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import networkx as nx
from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer


@dataclass
class BenchmarkResult:
    method_name: str
    prompt_tokens: int
    preprocessing_time_ms: float
    captured_target_entity: bool
    estimated_kv_vram_mb: float


def estimate_kv_cache_vram_mb(tokens: int, num_layers: int = 32, num_heads: int = 16, head_dim: int = 256) -> float:
    """
    Estimates KV-Cache VRAM usage for Gemma 2/4 9B architecture:
    VRAM = 2 (K and V) * num_layers * num_kv_heads * head_dim * tokens * 2 bytes (FP16/BF16)
    """
    bytes_per_token = 2 * num_layers * num_heads * head_dim * 2
    total_bytes = bytes_per_token * tokens
    return round(total_bytes / (1024 * 1024), 2)


class SimpleBM25Retriever:
    """Standard BM25 lexical retriever representing the Agentless baseline."""
    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus  # path -> content
        self.doc_lens = {k: len(v.split()) for k, v in corpus.items()}
        self.avg_doc_len = sum(self.doc_lens.values()) / max(len(self.doc_lens), 1)

    def retrieve(self, query: str, top_k: int = 2) -> Dict[str, str]:
        q_tokens = set(query.lower().split())
        scores = {}
        for path, text in self.corpus.items():
            text_tokens = text.lower().split()
            # Approximate TF-IDF matching
            score = sum(text_tokens.count(t) for t in q_tokens)
            scores[path] = score
        sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {k: self.corpus[k] for k, _ in sorted_files}


class PipelineEvaluator:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.raw_files: Dict[str, str] = {}
        self._load_raw_files()

    def _load_raw_files(self):
        import os
        for root, dirs, files in os.walk(self.repo_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", ".venv")]
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, self.repo_dir)
                    with open(full, "r", encoding="utf-8", errors="replace") as fh:
                        self.raw_files[rel] = fh.read()

    def run_raw_dump(self, issue: str, target_symbol: str) -> BenchmarkResult:
        """Baseline 1: Dumps all repository files into prompt."""
        t0 = time.perf_counter()
        full_text = "\n\n".join([f"### File: {k}\n{v}" for k, v in self.raw_files.items()])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tokens = int(len(full_text.split()) * 1.3)

        return BenchmarkResult(
            method_name="Baseline 1: Raw Context Dump",
            prompt_tokens=tokens,
            preprocessing_time_ms=round(elapsed_ms, 2),
            captured_target_entity=target_symbol in full_text,
            estimated_kv_vram_mb=estimate_kv_cache_vram_mb(tokens),
        )

    def run_bm25_rag(self, issue: str, target_symbol: str) -> BenchmarkResult:
        """Baseline 2: BM25 Lexical Retrieval (Agentless)."""
        t0 = time.perf_counter()
        retriever = SimpleBM25Retriever(self.raw_files)
        results = retriever.retrieve(issue, top_k=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = "\n\n".join([f"### File: {k}\n{v}" for k, v in results.items()])
        tokens = int(len(text.split()) * 1.3)

        return BenchmarkResult(
            method_name="Baseline 2: BM25 Lexical RAG (Agentless)",
            prompt_tokens=tokens,
            preprocessing_time_ms=round(elapsed_ms, 2),
            captured_target_entity=target_symbol in text,
            estimated_kv_vram_mb=estimate_kv_cache_vram_mb(tokens),
        )

    def run_neurocode(self, issue: str, target_symbol: str) -> BenchmarkResult:
        """Our Proposed Method: NeuroCode Host-CPU Graph Slicing."""
        t0 = time.perf_counter()
        builder = RepoGraphBuilder(self.repo_dir)
        graph = builder.build()
        slicer = CodeGraphSlicer(graph)
        seeds = slicer.find_seed_nodes(issue)
        subgraph = slicer.extract_subgraph(seeds, k_hops=1)
        slice_text = slicer.format_slice_for_llm(subgraph, seeds)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        tokens = int(len(slice_text.split()) * 1.3)

        return BenchmarkResult(
            method_name="Our Method: NeuroCode Host-CPU Slicer",
            prompt_tokens=tokens,
            preprocessing_time_ms=round(elapsed_ms, 2),
            captured_target_entity=target_symbol in slice_text,
            estimated_kv_vram_mb=estimate_kv_cache_vram_mb(tokens),
        )
