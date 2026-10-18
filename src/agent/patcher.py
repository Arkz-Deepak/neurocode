"""
Agent prompt formatting, diff extraction, and patch application.
Optimized for local models (Gemma) to avoid JSON tool-calling failures.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List


SYSTEM_PROMPT = """You are an expert autonomous software engineer.
You are given an issue description and a structurally sliced context of the repository.
Your task is to fix the issue by providing a clean SEARCH/REPLACE block.

Formatting Rules:
1. Identify the exact file path and code snippet to modify.
2. Output your fix using this exact format:

FILE: <file_path>
<<<<<<< SEARCH
<exact original code lines>
=======
<fixed replacement code lines>
>>>>>>>

Do not include any conversational filler. Be precise and minimal.
"""


@dataclass
class PatchResult:
    success: bool
    file_path: Optional[str] = None
    search_block: Optional[str] = None
    replace_block: Optional[str] = None
    error_message: Optional[str] = None


class PatchApplier:
    @staticmethod
    def extract_patch(llm_output: str, graph: Optional[Any] = None, repo_dir: Optional[str] = None) -> PatchResult:
        """
        Extracts patch from LLM output.
        Tier 1: Standard SEARCH/REPLACE diff markers.
        Tier 2: AST-Aware Entity Replacement (matches generated function/class against CPG node).
        """
        import os
        # 1. Tier 1: Exact SEARCH/REPLACE Block
        file_match = re.search(r"FILE:\s*([^\n\r]+)", llm_output)
        if file_match:
            file_path = file_match.group(1).strip("`' ")
            pattern = r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>"
            block_match = re.search(pattern, llm_output, re.DOTALL)
            if block_match:
                return PatchResult(
                    success=True,
                    file_path=file_path,
                    search_block=block_match.group(1),
                    replace_block=block_match.group(2),
                )

        # 2. Tier 2: AST Symbol Replacement Fallback
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        if not code_blocks:
            if "def " in llm_output or "class " in llm_output:
                code_blocks = [llm_output]

        for block in code_blocks:
            func_match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", block)
            class_match = re.search(r"class\s+([a-zA-Z0-9_]+)", block)
            symbol_name = func_match.group(1) if func_match else (class_match.group(1) if class_match else None)

            if symbol_name and graph is not None:
                for node, data in graph.nodes(data=True):
                    if data.get("name") == symbol_name and data.get("type") in ("function", "method", "class"):
                        file_path = data.get("file")
                        orig_source = data.get("source") or data.get("code")
                        if file_path and orig_source and orig_source.strip():
                            return PatchResult(
                                success=True,
                                file_path=file_path,
                                search_block=orig_source.strip(),
                                replace_block=block.strip()
                            )
                        elif file_path and repo_dir:
                            full_p = os.path.join(repo_dir, file_path)
                            if os.path.exists(full_p):
                                with open(full_p, "r", encoding="utf-8") as fh:
                                    lines = fh.readlines()
                                s_line = data.get("start_line", 1) - 1
                                e_line = data.get("end_line", len(lines))
                                orig_chunk = "".join(lines[s_line:e_line]).strip()
                                return PatchResult(
                                    success=True,
                                    file_path=file_path,
                                    search_block=orig_chunk,
                                    replace_block=block.strip()
                                )

        return PatchResult(success=False, error_message="No valid SEARCH/REPLACE block or AST symbol match found.")

    @staticmethod
    def apply_patch_to_file(full_file_path: str, search_block: str, replace_block: str) -> Tuple[bool, str]:
        """Applies an exact search-and-replace edit to a target file on disk."""
        try:
            with open(full_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if search_block not in content:
                # Fallback: Normalize line endings
                norm_search = search_block.replace("\r\n", "\n").strip()
                norm_content = content.replace("\r\n", "\n")
                if norm_search not in norm_content:
                    return False, f"Search block not found in {full_file_path}"
                new_content = norm_content.replace(norm_search, replace_block.replace("\r\n", "\n"), 1)
            else:
                new_content = content.replace(search_block, replace_block, 1)

            with open(full_file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, "Patch applied successfully."
        except Exception as e:
            return False, f"Failed to modify {full_file_path}: {str(e)}"
