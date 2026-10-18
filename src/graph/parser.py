"""
Tree-sitter based AST extractor for Python source files.
Extracts function definitions, class structures, imports, and caller-callee references.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(PY_LANGUAGE)


@dataclass
class FunctionEntity:
    name: str
    class_name: Optional[str]
    start_line: int
    end_line: int
    docstring: Optional[str]
    calls: List[str] = field(default_factory=list)
    source_code: str = ""


@dataclass
class ClassEntity:
    name: str
    bases: List[str]
    start_line: int
    end_line: int
    docstring: Optional[str]
    methods: List[FunctionEntity] = field(default_factory=list)
    source_code: str = ""


@dataclass
class ImportEntity:
    module: str
    names: List[str]
    alias: Optional[str] = None


@dataclass
class ParsedFile:
    file_path: str
    imports: List[ImportEntity] = field(default_factory=list)
    classes: List[ClassEntity] = field(default_factory=list)
    functions: List[FunctionEntity] = field(default_factory=list)


def parse_python_file(file_path: str, code_str: Optional[str] = None) -> ParsedFile:
    """Parses a Python file and returns structured AST entities."""
    if code_str is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code_str = f.read()

    code_bytes = code_str.encode("utf-8")
    tree = _PARSER.parse(code_bytes)
    root = tree.root_node

    parsed = ParsedFile(file_path=file_path)

    def extract_text(node: Node) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def extract_docstring(body_node: Node) -> Optional[str]:
        if not body_node:
            return None
        for child in body_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = extract_text(sub)
                        return raw.strip("\"' \n\t")
            elif child.type != "comment":
                break
        return None

    def find_calls(node: Node) -> List[str]:
        calls = []
        cursor = node.walk()
        visited = set()
        stack = [node]

        while stack:
            curr = stack.pop()
            if curr.id in visited:
                continue
            visited.add(curr.id)

            if curr.type == "call":
                func_node = curr.child_by_field_name("function")
                if func_node:
                    calls.append(extract_text(func_node))

            for child in curr.children:
                stack.append(child)

        return list(dict.fromkeys(calls))

    def parse_function(node: Node, class_name: Optional[str] = None) -> FunctionEntity:
        name_node = node.child_by_field_name("name")
        func_name = extract_text(name_node) if name_node else "<anonymous>"
        body_node = node.child_by_field_name("body")
        docstring = extract_docstring(body_node)
        calls = find_calls(node)
        snippet = extract_text(node)

        return FunctionEntity(
            name=func_name,
            class_name=class_name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            calls=calls,
            source_code=snippet,
        )

    def parse_class(node: Node) -> ClassEntity:
        name_node = node.child_by_field_name("name")
        class_name = extract_text(name_node) if name_node else "<anonymous>"
        superclasses_node = node.child_by_field_name("superclasses")

        bases = []
        if superclasses_node:
            for child in superclasses_node.children:
                if child.type == "identifier" or child.type == "attribute":
                    bases.append(extract_text(child))

        body_node = node.child_by_field_name("body")
        docstring = extract_docstring(body_node)
        methods = []

        if body_node:
            for child in body_node.children:
                if child.type == "function_definition":
                    methods.append(parse_function(child, class_name=class_name))

        snippet = extract_text(node)
        return ClassEntity(
            name=class_name,
            bases=bases,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            methods=methods,
            source_code=snippet,
        )

    # Top-level scan
    for child in root.children:
        if child.type in ("import_statement", "import_from_statement"):
            if child.type == "import_statement":
                for sub in child.children:
                    if sub.type == "dotted_name":
                        parsed.imports.append(ImportEntity(module=extract_text(sub), names=[]))
            elif child.type == "import_from_statement":
                mod_node = child.child_by_field_name("module_name")
                module_name = extract_text(mod_node) if mod_node else ""
                names = []
                for sub in child.children:
                    if sub.type == "dotted_name" and sub != mod_node:
                        names.append(extract_text(sub))
                    elif sub.type == "identifier":
                        names.append(extract_text(sub))
                parsed.imports.append(ImportEntity(module=module_name, names=names))

        elif child.type == "function_definition":
            parsed.functions.append(parse_function(child, class_name=None))

        elif child.type == "class_definition":
            parsed.classes.append(parse_class(child))

    return parsed
