from __future__ import annotations
import ast
import os
from typing import Dict, List, Any, Tuple

ROOT_FILES_DEFAULT = [
    "hrm.py",
    os.path.join("models", "layers.py"),
    os.path.join("models", "losses.py"),
    os.path.join("models", "common.py"),
    os.path.join("models", "evolve", "pipeline.py"),
    os.path.join("models", "evolve", "planner.py"),
    os.path.join("models", "evolve", "concepts.py"),
    os.path.join("models", "evolve", "literature.py"),
    os.path.join("models", "evolve", "introspect.py"),
    os.path.join("scripts", "test_self_evolve.py"),
]


def _outline_from_ast(tree: ast.AST) -> Dict[str, Any]:
    outline: Dict[str, Any] = {"functions": [], "classes": []}
    for node in tree.body if hasattr(tree, "body") else []:
        if isinstance(node, ast.FunctionDef):
            outline["functions"].append({
                "name": node.name,
                "lineno": node.lineno,
                "doc": ast.get_docstring(node) or "",
                "args": [a.arg for a in node.args.args],
            })
        elif isinstance(node, ast.ClassDef):
            cls = {"name": node.name, "lineno": node.lineno, "doc": ast.get_docstring(node) or "", "methods": []}
            for b in node.body:
                if isinstance(b, ast.FunctionDef):
                    cls["methods"].append({
                        "name": b.name,
                        "lineno": b.lineno,
                        "doc": ast.get_docstring(b) or "",
                        "args": [a.arg for a in b.args.args],
                    })
            outline["classes"].append(cls)
    return outline


def summarize_files(root: str, include: List[str] | None = None, max_bytes_per_file: int = 20000) -> Dict[str, Any]:
    """Create a compact, structured summary of the most relevant code files.

    Returns a dict mapping relpath -> {"outline": ..., "head": "first_lines"}.
    """
    include = include or ROOT_FILES_DEFAULT
    result: Dict[str, Any] = {}
    for rel in include:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read(max_bytes_per_file)
        except Exception:
            continue
        try:
            tree = ast.parse(src)
            outline = _outline_from_ast(tree)
        except SyntaxError:
            outline = {"functions": [], "classes": []}
        head = "\n".join(src.splitlines()[:200])
        result[rel] = {"outline": outline, "head": head}
    return result


def render_summary_text(summary: Dict[str, Any]) -> str:
    """Render the structured summary into a readable text for LLM context."""
    lines: List[str] = []
    for rel, info in summary.items():
        lines.append(f"# File: {rel}")
        outline = info.get("outline", {})
        funcs = outline.get("functions", [])
        classes = outline.get("classes", [])
        if funcs:
            lines.append("Functions:")
            for fn in funcs:
                args = ", ".join(fn.get("args", []))
                lines.append(f" - def {fn['name']}({args})  [line {fn['lineno']}]  doc: {fn.get('doc','')[:120]}")
        if classes:
            lines.append("Classes:")
            for cls in classes:
                lines.append(f" - class {cls['name']}  [line {cls['lineno']}]  doc: {cls.get('doc','')[:120]}")
                for m in cls.get("methods", []):
                    args = ", ".join(m.get("args", []))
                    lines.append(f"   - def {m['name']}({args})  [line {m['lineno']}]  doc: {m.get('doc','')[:120]}")
        lines.append("")
    return "\n".join(lines)
