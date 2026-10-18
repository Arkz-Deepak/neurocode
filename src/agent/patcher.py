"""
Agent prompt formatting, diff extraction, and patch application.
Optimized for local models (Gemma) to avoid JSON tool-calling failures.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple


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
    def extract_patch(llm_output: str) -> PatchResult:
        """Extracts FILE, SEARCH, and REPLACE blocks from the LLM's raw markdown output."""
        file_match = re.search(r"FILE:\s*([^\n\r]+)", llm_output)
        if not file_match:
            return PatchResult(success=False, error_message="Missing 'FILE: <path>' header in output.")

        file_path = file_match.group(1).strip("`' ")

        pattern = r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>"
        block_match = re.search(pattern, llm_output, re.DOTALL)
        if not block_match:
            return PatchResult(success=False, file_path=file_path, error_message="Missing <<<<<<< SEARCH ... ======= ... >>>>>>> block.")

        search_block = block_match.group(1)
        replace_block = block_match.group(2)

        return PatchResult(
            success=True,
            file_path=file_path,
            search_block=search_block,
            replace_block=replace_block,
        )

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
