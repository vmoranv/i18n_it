from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


@dataclass
class Replacement:
    start: int
    end: int
    new_literal: str


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def abs_pos(offsets: list[int], lineno: int, col: int) -> int:
    return offsets[lineno - 1] + col


def iter_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if not is_ignored_path(p) and p.is_file()]


def collect_replacements(source: str) -> list[Replacement]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    offsets = line_offsets(source)
    replacements: list[Replacement] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "t":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        old = first.value
        if "." not in old:
            continue
        if not KEY_RE.match(old):
            continue
        new = old.replace(".", "-")
        if new == old:
            continue
        if not (
            hasattr(first, "lineno")
            and hasattr(first, "col_offset")
            and hasattr(first, "end_lineno")
            and hasattr(first, "end_col_offset")
        ):
            continue
        start = abs_pos(offsets, first.lineno, first.col_offset)
        end = abs_pos(offsets, first.end_lineno, first.end_col_offset)
        replacements.append(Replacement(start=start, end=end, new_literal=repr(new)))

    return replacements


def apply_replacements(source: str, replacements: list[Replacement]) -> str:
    if not replacements:
        return source
    out = source
    for rep in sorted(replacements, key=lambda r: r.start, reverse=True):
        out = out[: rep.start] + rep.new_literal + out[rep.end :]
    return out


def process_file(path: Path, dry_run: bool) -> tuple[bool, int]:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return False, 0
    replacements = collect_replacements(source)
    if not replacements:
        return False, 0
    updated = apply_replacements(source, replacements)
    if updated == source:
        return False, 0
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return True, len(replacements)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize dotted t('key.with.dot') literals to t('key-with-dot')."
    )
    parser.add_argument("--root", required=True, help="Project root to scan.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = iter_py_files(root)

    touched = 0
    replaced = 0
    for path in files:
        changed, count = process_file(path, dry_run=args.dry_run)
        if changed:
            touched += 1
            replaced += count
            print(f"[updated] {path} ({count})")

    print(
        f"Scanned {len(files)} file(s); {'would update' if args.dry_run else 'updated'} {touched} file(s); "
        f"replaced literals: {replaced}."
    )


if __name__ == "__main__":
    main()
