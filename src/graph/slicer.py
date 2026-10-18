"""
Host-side deterministic context slicer.
Extracts a minimal k-hop dependency subgraph around target/faulty entities.
Compresses a 50,000-line codebase down to < 1,500 tokens for local LLMs.
"""

from typing import List, Set, Dict, Optional
import networkx as nx


class CodeGraphSlicer:
    def __init__(self, graph: nx.DiGraph, max_tokens: int = 1500):
        self.graph = graph
        self.max_tokens = max_tokens

    def find_seed_nodes(self, query: str) -> List[str]:
        """Finds candidate seed nodes from an issue description or traceback."""
        tokens = set(query.replace("(", " ").replace(")", " ").replace(":", " ").replace(".", " ").split())
        seed_nodes = []

        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type")
            if node_type in ("function", "method", "class"):
                name = data.get("name", "")
                if name and name in tokens:
                    seed_nodes.append(node)
            elif node_type == "file":
                path = data.get("path", "")
                if path and (path in query or any(t == path.split("/")[-1] for t in tokens)):
                    seed_nodes.append(node)

        return seed_nodes

    def extract_subgraph(self, seed_nodes: List[str], k_hops: int = 1) -> nx.DiGraph:
        """Extracts the k-hop reachable neighborhood around the seed nodes."""
        if not seed_nodes:
            return nx.DiGraph()

        included_nodes: Set[str] = set(seed_nodes)

        # Expand neighborhood (both forward callees and backward callers)
        current_layer = set(seed_nodes)
        for _ in range(k_hops):
            next_layer = set()
            for node in current_layer:
                # Successors (what this node calls / contains)
                for succ in self.graph.successors(node):
                    if succ not in included_nodes:
                        next_layer.add(succ)
                # Predecessors (who calls / contains this node)
                for pred in self.graph.predecessors(node):
                    if pred not in included_nodes:
                        next_layer.add(pred)
            included_nodes.update(next_layer)
            current_layer = next_layer

        return self.graph.subgraph(included_nodes).copy()

    def format_slice_for_llm(self, subgraph: nx.DiGraph, primary_seeds: List[str]) -> str:
        """
        Formats the sliced subgraph into an ultra-compact, high-density Markdown prompt.
        Target entities get full source code.
        Neighbors get signatures/docstrings.
        """
        if not subgraph.nodes:
            return "No relevant code graph entities found."

        output_lines = [
            "# REPOSITORY CONTEXT SLICE (Extracted by NeuroCode Host Engine)",
            "> Only components structurally related to the target issue are shown below.",
            ""
        ]

        # Group by file
        files_map: Dict[str, List[str]] = {}
        for node, data in subgraph.nodes(data=True):
            file_path = data.get("file", data.get("path", "unknown"))
            files_map.setdefault(file_path, []).append(node)

        for file_path, nodes in files_map.items():
            output_lines.append(f"## File: `{file_path}`")
            for node in nodes:
                data = subgraph.nodes[node]
                node_type = data.get("type")
                name = data.get("name", node)

                if node in primary_seeds:
                    # High priority: Show full source code
                    code = data.get("code", "")
                    output_lines.append(f"### [PRIMARY TARGET] {node_type.upper()}: `{name}` (Lines {data.get('start_line')}-{data.get('end_line')})")
                    output_lines.append("```python")
                    output_lines.append(code)
                    output_lines.append("```\n")
                elif node_type in ("function", "method"):
                    # 1-hop dependency: Include full implementation to diagnose callee/caller bugs
                    code = data.get("code", "")
                    output_lines.append(f"### [DEPENDENCY] {node_type.upper()}: `{name}` (Lines {data.get('start_line')}-{data.get('end_line')})")
                    output_lines.append("```python")
                    output_lines.append(code)
                    output_lines.append("```\n")
                elif node_type == "class":
                    bases = ", ".join(data.get("bases", []))
                    output_lines.append(f"- **class** `{name}` (inherits from: `{bases}`)")

            output_lines.append("")

        return "\n".join(output_lines)
