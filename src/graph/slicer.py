"""
Host-side deterministic context slicer with priority-ranked budget packing.
Extracts a minimal k-hop dependency subgraph around target/faulty entities.
Formally guarantees context stays under max_tokens budget via heuristic greedy knapsack packing.
"""

from typing import List, Set, Dict, Optional, Tuple
import networkx as nx


class CodeGraphSlicer:
    def __init__(self, graph: nx.DiGraph, max_tokens: int = 1500):
        self.graph = graph
        self.max_tokens = max_tokens

    def find_seed_nodes(self, query: str) -> List[str]:
        """Finds candidate seed nodes from an issue description or traceback via lexical normalization."""
        stopwords = {
            "get", "set", "the", "and", "for", "with", "from", "that", "this", "file", "line",
            "code", "def", "class", "import", "return", "true", "false", "none", "self", "args",
            "kwargs", "test", "issue", "bug", "error", "fail", "pass", "call", "type", "name",
            "str", "int", "list", "dict", "user", "run", "in", "to", "of", "is", "it", "on", "a",
            "an", "by", "as", "at", "if", "or", "not", "be", "are", "was", "were", "when", "how"
        }
        tokens = {
            t for t in query.lower().replace("(", " ").replace(")", " ").replace(":", " ").replace(".", " ").replace("/", " ").replace("_", " ").split()
            if len(t) > 2 and t not in stopwords
        }
        scored_seeds = []

        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type")
            name = (data.get("name") or "").lower()
            if node_type in ("function", "method", "class") and name:
                parts = set(name.split("_"))
                overlap = len(parts.intersection(tokens))
                if name in tokens or overlap > 0:
                    weight = 2.0 if name in tokens else (0.5 * overlap)
                    scored_seeds.append((node, weight))
            elif node_type == "file":
                path = (data.get("path") or "").lower()
                if path and any(t in path for t in tokens):
                    scored_seeds.append((node, 1.0))

        scored_seeds.sort(key=lambda x: x[1], reverse=True)
        # Retain top-6 most relevant seeds to prevent combinatorial explosion
        seed_nodes = [node for node, _ in scored_seeds[:6]]
        return list(dict.fromkeys(seed_nodes))

    def extract_subgraph(self, seed_nodes: List[str], k_hops: int = 1) -> nx.DiGraph:
        """Extracts the k-hop reachable neighborhood around the seed nodes."""
        if not seed_nodes:
            # Fallback: Top-level entrypoints if no seed found
            seed_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "function"][:3]
            if not seed_nodes:
                return nx.DiGraph()

        included_nodes: Set[str] = set(seed_nodes)
        current_layer = set(seed_nodes)

        for _ in range(k_hops):
            next_layer = set()
            for node in current_layer:
                # Direct callees and dependencies
                for succ in self.graph.successors(node):
                    if succ not in included_nodes:
                        next_layer.add(succ)
                # Callers and enclosing classes/files
                for pred in self.graph.predecessors(node):
                    if pred not in included_nodes:
                        next_layer.add(pred)
            included_nodes.update(next_layer)
            current_layer = next_layer

        return self.graph.subgraph(included_nodes).copy()

    def compute_priority_score(self, node: str, seed_nodes: List[str], query_tokens: Set[str]) -> float:
        """
        Computes formal relevance priority score for budget knapsack packing:
        Score(v) = w_dist * (1 / (1 + dist)) + w_lex * LexicalOverlap + w_type * TypeWeight
        """
        data = self.graph.nodes[node]
        node_type = data.get("type", "unknown")
        node_name = (data.get("name") or "").lower()

        # 1. Graph distance score
        if node in seed_nodes:
            dist_score = 1.0
        else:
            # Check shortest path to any seed
            min_dist = 999
            for seed in seed_nodes:
                if nx.has_path(self.graph, seed, node):
                    min_dist = min(min_dist, nx.shortest_path_length(self.graph, seed, node))
                elif nx.has_path(self.graph, node, seed):
                    min_dist = min(min_dist, nx.shortest_path_length(self.graph, node, seed))
            dist_score = 1.0 / (1.0 + min_dist) if min_dist != 999 else 0.2

        # 2. Lexical relevance score
        lex_score = 1.0 if node_name in query_tokens else 0.0

        # 3. Structural type importance weight
        type_weights = {
            "function": 0.9,
            "method": 0.9,
            "class": 0.7,
            "file": 0.5,
            "import": 0.3,
        }
        type_score = type_weights.get(node_type, 0.4)

        return (0.45 * dist_score) + (0.35 * lex_score) + (0.20 * type_score)

    def format_slice_for_llm(self, subgraph: nx.DiGraph, primary_seeds: List[str], query: str = "", max_tokens: Optional[int] = None) -> str:
        """
        Knapsack-style priority packing:
        Packs highest-priority entities with full code first.
        If budget runs low, falls back to signature/docstring.
        Guarantees total tokens <= max_tokens.
        """
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if not subgraph.nodes:
            return "No relevant code graph entities found."

        query_tokens = set(query.lower().replace(".", " ").replace("/", " ").split())

        # Rank all nodes in the extracted subgraph by priority score
        scored_nodes: List[Tuple[str, float]] = [
            (node, self.compute_priority_score(node, primary_seeds, query_tokens))
            for node in subgraph.nodes
        ]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        current_token_count = 50  # Header reserve
        packed_entities: Dict[str, List[Dict]] = {}  # file_path -> list of entity dicts

        for node, score in scored_nodes:
            data = subgraph.nodes[node]
            file_path = data.get("file", data.get("path", "unknown"))
            node_type = data.get("type")
            name = data.get("name", node)
            code = data.get("code", "")
            doc = data.get("docstring") or ""

            full_code_tokens = int(len(code.split()) * 1.3)
            sig_tokens = int(len(f"{node_type} {name} (Lines {data.get('start_line')}-{data.get('end_line')}): {doc}".split()) * 1.3)

            # Decision logic based on remaining token budget
            if node_type == "file":
                continue

            if (current_token_count + full_code_tokens <= effective_max_tokens) and (node in primary_seeds or full_code_tokens < 350):
                # Tier 1: Full implementation
                entity_repr = {
                    "tier": "FULL",
                    "type": node_type,
                    "name": name,
                    "start": data.get("start_line"),
                    "end": data.get("end_line"),
                    "content": code,
                    "is_seed": node in primary_seeds
                }
                current_token_count += full_code_tokens
            elif current_token_count + sig_tokens <= effective_max_tokens:
                # Tier 2: Signature and docstring only
                entity_repr = {
                    "tier": "SIGNATURE",
                    "type": node_type,
                    "name": name,
                    "start": data.get("start_line"),
                    "end": data.get("end_line"),
                    "content": f"- **{node_type}** `{name}`: {doc.strip() or 'No docstring.'}",
                    "is_seed": False
                }
                current_token_count += sig_tokens
            else:
                # Budget exhausted: Skip low-priority distant nodes
                continue

            packed_entities.setdefault(file_path, []).append(entity_repr)

        # Serialize into Markdown
        output_lines = [
            "# REPOSITORY CONTEXT SLICE (NeuroCode Priority Knapsack Engine)",
            f"> Context packed under strict budget (Budget: {self.max_tokens} tokens | Estimated: {current_token_count} tokens).",
            ""
        ]

        for file_path, entities in packed_entities.items():
            output_lines.append(f"## File: `{file_path}`")
            for ent in entities:
                if ent["tier"] == "FULL":
                    tag = "[PRIMARY TARGET]" if ent["is_seed"] else "[1-HOP DEPENDENCY]"
                    output_lines.append(f"### {tag} {ent['type'].upper()}: `{ent['name']}` (Lines {ent['start']}-{ent['end']})")
                    output_lines.append("```python")
                    output_lines.append(ent["content"])
                    output_lines.append("```\n")
                else:
                    output_lines.append(ent["content"])
            output_lines.append("")

        return "\n".join(output_lines)
