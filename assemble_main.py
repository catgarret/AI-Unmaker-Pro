"""Maintain the single-file application from the working component files."""
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parent
PARTS = ("runtime_setup.py", "file_publish.py", "metadata_core.py", "visible_bridge.py",
         "model_runtime.py", "qwen_inference.py", "pixel_engine.py", "open_watermark_bridge.py", "worker_runtime.py", "provenance.py", "app_backend.py",
         "ui_assets.py", "bootstrap_ui.py", "desktop_ui.py")


def component(path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    removed = set()
    for index, node in enumerate(tree.body):
        if (index == 0 and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            removed.update(range(node.lineno - 1, node.end_lineno))
        if isinstance(node, ast.ImportFrom) and node.module in {"__future__", "file_publish", "provenance"}:
            removed.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for index, line in enumerate(lines) if index not in removed).strip() + '\n'


def assemble():
    header = (ROOT / "assembly_header.txt").read_text(encoding="utf-8")
    sections = [header.rstrip() + "\n\n"]
    for name in PARTS:
        sections.append(f"# {name}\n# {'-' * 72}\n" + component(ROOT / name) + "\n\n")
    sections.append((ROOT / "assembly_entrypoint.txt").read_text(encoding="utf-8"))
    result = ''.join(sections)
    ast.parse(result)
    (ROOT / "main.py").write_text(result, encoding="utf-8")
    return ROOT / "main.py"


if __name__ == "__main__":
    print(assemble())
