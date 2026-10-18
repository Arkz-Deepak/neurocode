"""
Constructs a heterogeneous Code Property Graph (CPG) from a repository using NetworkX.
Runs in-memory on Host CPU and RAM in milliseconds.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
from src.graph.parser import parse_python_file, ParsedFile


class RepoGraphBuilder:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()
        self.graph = nx.DiGraph()
        self.symbol_table: Dict[str, str] = {}  # short_symbol -> qualified_node_id

    def build(self) -> nx.DiGraph:
        """Walks the repository, parses Python files, and creates graph nodes and edges."""
        parsed_files: List[ParsedFile] = []

        # 1. Collect and parse all Python files
        for root, dirs, files in os.walk(self.repo_root):
            # Exclude virtual environments, git, tests/caches
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", ".venv", "__pycache__", "build", "dist")]
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.repo_root)
                    try:
                        parsed = parse_python_file(full_path)
                        parsed.file_path = rel_path
                        parsed_files.append(parsed)
                    except Exception as e:
                        # Skip files with severe parse anomalies
                        continue

        # 2. Add File, Class, and Function nodes
        for pf in parsed_files:
            file_node = f"file:{pf.file_path}"
            self.graph.add_node(file_node, type="file", path=pf.file_path)

            # Add imports
            for imp in pf.imports:
                imp_id = f"import:{imp.module}"
                self.graph.add_node(imp_id, type="import", module=imp.module)
                self.graph.add_edge(file_node, imp_id, relation="imports")

            # Add top-level functions
            for func in pf.functions:
                func_id = f"func:{pf.file_path}:{func.name}"
                self.graph.add_node(
                    func_id,
                    type="function",
                    name=func.name,
                    file=pf.file_path,
                    start_line=func.start_line,
                    end_line=func.end_line,
                    docstring=func.docstring,
                    code=func.source_code,
                    calls=func.calls,
                )
                self.graph.add_edge(file_node, func_id, relation="contains")
                self.symbol_table[func.name] = func_id
                self.symbol_table[f"{pf.file_path}::{func.name}"] = func_id

            # Add classes and methods
            for cls in pf.classes:
                cls_id = f"class:{pf.file_path}:{cls.name}"
                self.graph.add_node(
                    cls_id,
                    type="class",
                    name=cls.name,
                    file=pf.file_path,
                    bases=cls.bases,
                    start_line=cls.start_line,
                    end_line=cls.end_line,
                    docstring=cls.docstring,
                    code=cls.source_code,
                )
                self.graph.add_edge(file_node, cls_id, relation="contains")
                self.symbol_table[cls.name] = cls_id
                self.symbol_table[f"{pf.file_path}::{cls.name}"] = cls_id

                for base in cls.bases:
                    if base in self.symbol_table:
                        self.graph.add_edge(cls_id, self.symbol_table[base], relation="inherits")

                for method in cls.methods:
                    method_id = f"method:{pf.file_path}:{cls.name}.{method.name}"
                    self.graph.add_node(
                        method_id,
                        type="method",
                        name=method.name,
                        class_name=cls.name,
                        file=pf.file_path,
                        start_line=method.start_line,
                        end_line=method.end_line,
                        docstring=method.docstring,
                        code=method.source_code,
                        calls=method.calls,
                    )
                    self.graph.add_edge(cls_id, method_id, relation="contains")
                    self.symbol_table[f"{cls.name}.{method.name}"] = method_id
                    self.symbol_table[f"{pf.file_path}::{cls.name}.{method.name}"] = method_id

        # 3. Resolve Caller-Callee Call Edges
        for node_id, data in list(self.graph.nodes(data=True)):
            if data.get("type") in ("function", "method"):
                calls = data.get("calls", [])
                for call_name in calls:
                    # Clean up method calls (e.g. self.do_something -> do_something)
                    short_name = call_name.split(".")[-1]
                    target_id = None
                    if call_name in self.symbol_table:
                        target_id = self.symbol_table[call_name]
                    elif short_name in self.symbol_table:
                        target_id = self.symbol_table[short_name]

                    if target_id and target_id != node_id:
                        self.graph.add_edge(node_id, target_id, relation="calls")

        return self.graph
