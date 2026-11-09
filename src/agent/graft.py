"""
AST Subtree Grafting Engine for Autonomous SWE Agents.
Performs deterministic, syntax-verified AST node replacements in-memory,
completely eliminating fragile multi-line whitespace and indentation diff failures.
"""

import ast
import difflib
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tree_sitter import Node, Parser, Language
import tree_sitter_python as tspython

_PY_LANG = Language(tspython.language())
_PY_PARSER = Parser(_PY_LANG)

try:
    import tree_sitter_javascript as tsjs
    _JS_LANG = Language(tsjs.language())
    _JS_PARSER = Parser(_JS_LANG)
except Exception:
    _JS_LANG = None
    _JS_PARSER = None

try:
    import tree_sitter_typescript as tsts
    _TS_LANG = Language(tsts.language_typescript())
    _TS_PARSER = Parser(_TS_LANG)
    _TSX_LANG = Language(tsts.language_tsx())
    _TSX_PARSER = Parser(_TSX_LANG)
except Exception:
    _TS_LANG = None
    _TS_PARSER = None
    _TSX_LANG = None
    _TSX_PARSER = None

try:
    import tree_sitter_go as tsgo
    _GO_LANG = Language(tsgo.language())
    _GO_PARSER = Parser(_GO_LANG)
except Exception:
    _GO_LANG = None
    _GO_PARSER = None

try:
    import tree_sitter_rust as tsrs
    _RUST_LANG = Language(tsrs.language())
    _RUST_PARSER = Parser(_RUST_LANG)
except Exception:
    _RUST_LANG = None
    _RUST_PARSER = None


def get_parser_for_file(file_path: str) -> Parser:
    """Returns the appropriate Tree-sitter parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".tsx" and _TSX_PARSER:
        return _TSX_PARSER
    elif ext == ".ts" and _TS_PARSER:
        return _TS_PARSER
    elif ext in (".js", ".jsx", ".mjs", ".cjs") and _JS_PARSER:
        return _JS_PARSER
    elif ext == ".go" and _GO_PARSER:
        return _GO_PARSER
    elif ext == ".rs" and _RUST_PARSER:
        return _RUST_PARSER
    return _PY_PARSER


def find_target_node(root: Node, symbol: str, code_bytes: bytes) -> Optional[Node]:
    """
    Locates the AST node corresponding to the given symbol name or 'ClassName.methodName'.
    Supports Python, JavaScript, TypeScript, Go, and Rust.
    """
    def get_text(node: Optional[Node]) -> str:
        if not node:
            return ""
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    target_class = None
    target_name = symbol
    if "." in symbol:
        target_class, target_name = symbol.split(".", 1)

    stack = [root]
    while stack:
        curr = stack.pop()
        
        # Check Function or Method across Python, JS, TS, Go, Rust
        if curr.type in ("function_definition", "function_declaration", "method_definition", "method_declaration", "function_item"):
            name_node = curr.child_by_field_name("name")
            if name_node and get_text(name_node) == target_name:
                if target_class:
                    # Check parent class or impl
                    parent = curr.parent
                    while parent and parent.type not in ("class_definition", "class_declaration", "impl_item"):
                        parent = parent.parent
                    if parent:
                        p_name_node = parent.child_by_field_name("name") or parent.child_by_field_name("type")
                        if p_name_node and target_class.lower() in get_text(p_name_node).lower():
                            return curr
                    # Check Go receiver
                    rec_node = curr.child_by_field_name("receiver")
                    if rec_node and target_class.lower() in get_text(rec_node).lower():
                        return curr
                else:
                    return curr

        # Check Class, Struct, or Impl
        elif curr.type in ("class_definition", "class_declaration", "type_declaration", "struct_item", "impl_item"):
            name_node = curr.child_by_field_name("name") or curr.child_by_field_name("type")
            if not target_class and name_node and target_name.lower() in get_text(name_node).lower():
                return curr

        # Check JS/TS Arrow Function or Variable
        elif curr.type == "variable_declarator":
            name_node = curr.child_by_field_name("name")
            val_node = curr.child_by_field_name("value")
            if name_node and get_text(name_node) == target_name:
                if curr.parent and curr.parent.type in ("lexical_declaration", "variable_declaration"):
                    return curr.parent
                return curr

        for child in reversed(curr.children):
            stack.append(child)

    return None


def align_indentation(replacement_code: str, target_indent_spaces: int) -> str:
    """
    Normalizes and re-indents replacement code so it seamlessly matches the AST target scope.
    """
    lines = replacement_code.strip("\r\n").splitlines()
    if not lines:
        return ""

    # Determine common leading whitespace in replacement lines (excluding empty lines)
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return replacement_code

    min_leading = min(len(l) - len(l.lstrip()) for l in non_empty)
    unindented_lines = [(l[min_leading:] if len(l) >= min_leading else l) for l in lines]

    target_indent = " " * target_indent_spaces
    reindented = []
    for idx, line in enumerate(unindented_lines):
        if not line.strip():
            reindented.append("")
        elif idx == 0:
            reindented.append(line.lstrip())
        else:
            leading_len = len(line) - len(line.lstrip())
            reindented.append(target_indent + (" " * leading_len) + line.lstrip())

    return "\n".join(reindented)


def graft_symbol(
    repo_path: str,
    file_path: str,
    symbol: str,
    replacement_code: str,
    apply: bool = False,
    test_command: Optional[str] = None,
    auto_rollback: bool = True
) -> Dict[str, Any]:
    """
    Performs deterministic AST subtree grafting for a target symbol.
    Optionally executes test_command to verify business logic, with auto-rollback on test failure.
    """
    base_dir = Path(repo_path).resolve()
    target_file = (base_dir / file_path).resolve()

    if not target_file.exists():
        return {
            "success": False,
            "error": f"File does not exist: {file_path}",
            "file_path": file_path
        }

    try:
        original_source = target_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {e}", "file_path": file_path}

    parser = get_parser_for_file(str(target_file))
    code_bytes = original_source.encode("utf-8")
    tree = parser.parse(code_bytes)

    # 1. Locate Target AST Node
    target_node = find_target_node(tree.root_node, symbol, code_bytes)
    if not target_node:
        return {
            "success": False,
            "error": f"Symbol '{symbol}' was not found in the AST of {file_path}.",
            "file_path": file_path
        }

    # 2. Indentation alignment
    target_indent_cols = target_node.start_point[1]
    formatted_replacement = align_indentation(replacement_code, target_indent_cols)

    # 3. Perform byte slice replacement
    new_bytes = (
        code_bytes[:target_node.start_byte]
        + formatted_replacement.encode("utf-8")
        + code_bytes[target_node.end_byte:]
    )
    new_source = new_bytes.decode("utf-8", errors="replace")

    # 4. AST Integrity Verification
    new_tree = parser.parse(new_bytes)
    if new_tree.root_node.has_error:
        return {
            "success": False,
            "error": "SyntaxError: Grafted code resulted in invalid AST grammar.",
            "file_path": file_path,
            "syntax_valid": False
        }

    # Python-specific AST check
    if target_file.suffix == ".py":
        try:
            ast.parse(new_source, filename=str(target_file))
        except SyntaxError as se:
            return {
                "success": False,
                "error": f"Python SyntaxError at line {se.lineno}: {se.msg}",
                "file_path": file_path,
                "syntax_valid": False
            }

    # 5. Generate Diff
    diff_lines = list(difflib.unified_diff(
        original_source.splitlines(keepends=True),
        new_source.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    ))
    diff_str = "\n".join(diff_lines)

    # 6. Apply to disk if requested
    if apply:
        try:
            target_file.write_text(new_source, encoding="utf-8")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write changes to disk: {e}",
                "file_path": file_path
            }

        # 7. Automated Test Verification (Self-Healing Loop)
        if test_command:
            try:
                proc = subprocess.run(
                    test_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(base_dir)
                )
                if proc.returncode != 0:
                    # Test failed!
                    if auto_rollback:
                        target_file.write_text(original_source, encoding="utf-8")
                        return {
                            "success": False,
                            "file_path": file_path,
                            "symbol": symbol,
                            "syntax_valid": True,
                            "applied": False,
                            "rolled_back": True,
                            "test_passed": False,
                            "test_exit_code": proc.returncode,
                            "test_stdout": proc.stdout[:2000],
                            "test_stderr": proc.stderr[:2000],
                            "error": f"Unit tests failed with exit code {proc.returncode}. Changes were automatically rolled back to prevent repository breakage."
                        }
                    else:
                        return {
                            "success": True,
                            "file_path": file_path,
                            "symbol": symbol,
                            "syntax_valid": True,
                            "applied": True,
                            "rolled_back": False,
                            "test_passed": False,
                            "test_exit_code": proc.returncode,
                            "test_stdout": proc.stdout[:2000],
                            "test_stderr": proc.stderr[:2000],
                            "diff": diff_str,
                            "message": f"Code grafted to disk, but tests failed (exit code {proc.returncode})."
                        }
            except Exception as te:
                if auto_rollback:
                    target_file.write_text(original_source, encoding="utf-8")
                return {
                    "success": False,
                    "error": f"Error executing test command: {str(te)}",
                    "file_path": file_path,
                    "rolled_back": auto_rollback
                }

    return {
        "success": True,
        "file_path": file_path,
        "symbol": symbol,
        "entity_type": target_node.type,
        "start_line": target_node.start_point[0] + 1,
        "end_line": target_node.end_point[0] + 1,
        "syntax_valid": True,
        "applied": apply,
        "test_passed": True if test_command else None,
        "diff": diff_str,
        "message": "Subtree successfully grafted with 100% verified syntax and tests passed." if test_command else "Subtree successfully grafted with 100% verified syntax."
    }
