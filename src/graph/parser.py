"""
Tree-sitter based AST extractor for Python, JavaScript, and TypeScript source files.
Extracts function definitions, class structures, imports, and caller-callee references.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(PY_LANGUAGE)

try:
    import tree_sitter_javascript as tsjs
    JS_LANGUAGE = Language(tsjs.language())
    _JS_PARSER = Parser(JS_LANGUAGE)
except Exception:
    JS_LANGUAGE = None
    _JS_PARSER = None

try:
    import tree_sitter_typescript as tsts
    TS_LANGUAGE = Language(tsts.language_typescript())
    _TS_PARSER = Parser(TS_LANGUAGE)
    TSX_LANGUAGE = Language(tsts.language_tsx())
    _TSX_PARSER = Parser(TSX_LANGUAGE)
except Exception:
    TS_LANGUAGE = None
    _TS_PARSER = None
    TSX_LANGUAGE = None
    _TSX_PARSER = None

try:
    import tree_sitter_go as tsgo
    GO_LANGUAGE = Language(tsgo.language())
    _GO_PARSER = Parser(GO_LANGUAGE)
except Exception:
    GO_LANGUAGE = None
    _GO_PARSER = None

try:
    import tree_sitter_rust as tsrs
    RUST_LANGUAGE = Language(tsrs.language())
    _RUST_PARSER = Parser(RUST_LANGUAGE)
except Exception:
    RUST_LANGUAGE = None
    _RUST_PARSER = None


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
        stack = [node]
        visited = set()

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
                if child.type in ("identifier", "attribute"):
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


def parse_js_ts_file(file_path: str, code_str: Optional[str] = None) -> ParsedFile:
    """Parses a JavaScript or TypeScript file (.js, .jsx, .ts, .tsx) and extracts AST entities."""
    if code_str is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code_str = f.read()

    code_bytes = code_str.encode("utf-8")
    ext = Path(file_path).suffix.lower()

    if ext == ".tsx" and _TSX_PARSER is not None:
        parser = _TSX_PARSER
    elif ext == ".ts" and _TS_PARSER is not None:
        parser = _TS_PARSER
    elif _JS_PARSER is not None:
        parser = _JS_PARSER
    else:
        parser = _PARSER

    tree = parser.parse(code_bytes)
    root = tree.root_node
    parsed = ParsedFile(file_path=file_path)

    def extract_text(node: Optional[Node]) -> str:
        if not node:
            return ""
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def find_calls(node: Node) -> List[str]:
        calls = []
        stack = [node]
        visited = set()
        while stack:
            curr = stack.pop()
            if curr.id in visited:
                continue
            visited.add(curr.id)
            if curr.type == "call_expression":
                fn_node = curr.child_by_field_name("function")
                if fn_node:
                    calls.append(extract_text(fn_node))
            for child in curr.children:
                stack.append(child)
        return list(dict.fromkeys(calls))

    def parse_function_node(node: Node, class_name: Optional[str] = None, explicit_name: Optional[str] = None) -> FunctionEntity:
        name_node = node.child_by_field_name("name")
        func_name = explicit_name or (extract_text(name_node) if name_node else "<anonymous>")
        calls = find_calls(node)
        snippet = extract_text(node)
        return FunctionEntity(
            name=func_name,
            class_name=class_name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=None,
            calls=calls,
            source_code=snippet
        )

    def parse_class_node(node: Node) -> ClassEntity:
        name_node = node.child_by_field_name("name")
        class_name = extract_text(name_node) if name_node else "<anonymous>"
        bases = []
        # Check extends / heritage
        for child in node.children:
            if child.type == "class_heritage":
                for h_child in child.children:
                    if h_child.type == "extends_clause":
                        for sub in h_child.children:
                            if sub.type in ("identifier", "type_identifier", "member_expression"):
                                bases.append(extract_text(sub))

        methods: List[FunctionEntity] = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "method_definition":
                    m_name_node = child.child_by_field_name("name")
                    m_name = extract_text(m_name_node) if m_name_node else "<anonymous>"
                    methods.append(parse_function_node(child, class_name=class_name, explicit_name=m_name))

        snippet = extract_text(node)
        return ClassEntity(
            name=class_name,
            bases=bases,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=None,
            methods=methods,
            source_code=snippet
        )

    def process_decl(node: Node):
        if node.type == "import_statement":
            src_node = node.child_by_field_name("source")
            mod_name = extract_text(src_node).strip("\"'") if src_node else ""
            names: List[str] = []
            for child in node.children:
                if child.type == "import_clause":
                    for sub in child.children:
                        if sub.type == "identifier":
                            names.append(extract_text(sub))
                        elif sub.type == "named_imports":
                            for spec in sub.children:
                                if spec.type == "import_specifier":
                                    n = spec.child_by_field_name("name")
                                    if n:
                                        names.append(extract_text(n))
            parsed.imports.append(ImportEntity(module=mod_name, names=names))

        elif node.type == "function_declaration":
            parsed.functions.append(parse_function_node(node))

        elif node.type == "class_declaration":
            parsed.classes.append(parse_class_node(node))

        elif node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.children:
                if declarator.type == "variable_declarator":
                    val = declarator.child_by_field_name("value")
                    name_n = declarator.child_by_field_name("name")
                    var_name = extract_text(name_n)
                    if val and val.type in ("arrow_function", "function_expression"):
                        parsed.functions.append(parse_function_node(val, explicit_name=var_name))

        elif node.type == "export_statement":
            for child in node.children:
                if child.type in ("function_declaration", "class_declaration", "lexical_declaration", "variable_declaration"):
                    process_decl(child)

    for child in root.children:
        process_decl(child)

    return parsed


def parse_go_file(file_path: str, code_str: Optional[str] = None) -> ParsedFile:
    """Parses a Go source file (.go) and extracts functions, methods, structs, and imports."""
    if code_str is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code_str = f.read()

    code_bytes = code_str.encode("utf-8")
    parser = _GO_PARSER or _PARSER
    tree = parser.parse(code_bytes)
    root = tree.root_node
    parsed = ParsedFile(file_path=file_path)

    def extract_text(node: Optional[Node]) -> str:
        if not node:
            return ""
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def find_calls(node: Node) -> List[str]:
        calls = []
        stack = [node]
        visited = set()
        while stack:
            curr = stack.pop()
            if curr.id in visited:
                continue
            visited.add(curr.id)
            if curr.type == "call_expression":
                fn_node = curr.child_by_field_name("function")
                if fn_node:
                    calls.append(extract_text(fn_node))
            for child in curr.children:
                stack.append(child)
        return list(dict.fromkeys(calls))

    for child in root.children:
        if child.type == "import_declaration":
            for spec in child.children:
                if spec.type == "import_spec_list":
                    for s in spec.children:
                        if s.type == "import_spec":
                            p = s.child_by_field_name("path")
                            if p:
                                parsed.imports.append(ImportEntity(module=extract_text(p).strip('"'), names=[]))
                elif spec.type == "import_spec":
                    p = spec.child_by_field_name("path")
                    if p:
                        parsed.imports.append(ImportEntity(module=extract_text(p).strip('"'), names=[]))

        elif child.type == "type_declaration":
            for spec in child.children:
                if spec.type == "type_spec":
                    name_n = spec.child_by_field_name("name")
                    name = extract_text(name_n)
                    parsed.classes.append(ClassEntity(
                        name=name,
                        bases=[],
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        docstring=None,
                        methods=[],
                        source_code=extract_text(child)
                    ))

        elif child.type == "method_declaration":
            name_n = child.child_by_field_name("name")
            name = extract_text(name_n)
            rec_n = child.child_by_field_name("receiver")
            receiver = extract_text(rec_n).strip("()* ") if rec_n else ""
            receiver_type = receiver.split()[-1] if receiver else None
            calls = find_calls(child)
            parsed.functions.append(FunctionEntity(
                name=name,
                class_name=receiver_type,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                docstring=None,
                calls=calls,
                source_code=extract_text(child)
            ))

        elif child.type == "function_declaration":
            name_n = child.child_by_field_name("name")
            name = extract_text(name_n)
            calls = find_calls(child)
            parsed.functions.append(FunctionEntity(
                name=name,
                class_name=None,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                docstring=None,
                calls=calls,
                source_code=extract_text(child)
            ))

    return parsed


def parse_rust_file(file_path: str, code_str: Optional[str] = None) -> ParsedFile:
    """Parses a Rust source file (.rs) and extracts functions, structs, impl blocks, and use declarations."""
    if code_str is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code_str = f.read()

    code_bytes = code_str.encode("utf-8")
    parser = _RUST_PARSER or _PARSER
    tree = parser.parse(code_bytes)
    root = tree.root_node
    parsed = ParsedFile(file_path=file_path)

    def extract_text(node: Optional[Node]) -> str:
        if not node:
            return ""
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def find_calls(node: Node) -> List[str]:
        calls = []
        stack = [node]
        visited = set()
        while stack:
            curr = stack.pop()
            if curr.id in visited:
                continue
            visited.add(curr.id)
            if curr.type == "call_expression":
                fn_node = curr.child_by_field_name("function")
                if fn_node:
                    calls.append(extract_text(fn_node))
            for child in curr.children:
                stack.append(child)
        return list(dict.fromkeys(calls))

    for child in root.children:
        if child.type == "use_declaration":
            arg = child.child_by_field_name("argument")
            if arg:
                parsed.imports.append(ImportEntity(module=extract_text(arg), names=[]))

        elif child.type in ("struct_item", "enum_item"):
            name_n = child.child_by_field_name("name")
            name = extract_text(name_n)
            parsed.classes.append(ClassEntity(
                name=name,
                bases=[],
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                docstring=None,
                methods=[],
                source_code=extract_text(child)
            ))

        elif child.type == "impl_item":
            type_n = child.child_by_field_name("type")
            impl_name = extract_text(type_n) if type_n else "<impl>"
            body = child.child_by_field_name("body")
            if body:
                for sub in body.children:
                    if sub.type == "function_item":
                        name_n = sub.child_by_field_name("name")
                        name = extract_text(name_n)
                        calls = find_calls(sub)
                        parsed.functions.append(FunctionEntity(
                            name=name,
                            class_name=impl_name,
                            start_line=sub.start_point[0] + 1,
                            end_line=sub.end_point[0] + 1,
                            docstring=None,
                            calls=calls,
                            source_code=extract_text(sub)
                        ))

        elif child.type == "function_item":
            name_n = child.child_by_field_name("name")
            name = extract_text(name_n)
            calls = find_calls(child)
            parsed.functions.append(FunctionEntity(
                name=name,
                class_name=None,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                docstring=None,
                calls=calls,
                source_code=extract_text(child)
            ))

    return parsed


def parse_file(file_path: str, code_str: Optional[str] = None) -> ParsedFile:
    """Universal parser dispatch: parses Python, JS/TS, Go, and Rust according to file extension."""
    ext = Path(file_path).suffix.lower()
    if ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
        return parse_js_ts_file(file_path, code_str)
    elif ext == ".go":
        return parse_go_file(file_path, code_str)
    elif ext == ".rs":
        return parse_rust_file(file_path, code_str)
    return parse_python_file(file_path, code_str)
