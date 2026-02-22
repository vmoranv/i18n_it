from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

CODING_RE = re.compile(r"^#.*coding[:=]\s*([-\w.]+)")
IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def iter_python_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if is_ignored_path(root):
            continue
        if root.is_file() and root.suffix == ".py":
            real = root.resolve()
            if real not in seen:
                files.append(root)
                seen.add(real)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if is_ignored_path(path):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            files.append(path)
    return files


def has_target_symbol_import(tree: ast.Module, target_module: str, symbol: str) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != target_module:
            continue
        for alias in node.names:
            if alias.name in {symbol, "*"}:
                return True
    return False


def detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    return "\n"


def format_import_alias(alias: ast.alias) -> str:
    if alias.asname:
        return f"{alias.name} as {alias.asname}"
    return alias.name


def normalize_legacy_symbol_imports(
    text: str,
    tree: ast.Module,
    symbol: str,
    legacy_modules: set[str],
) -> tuple[str, bool]:
    targets: list[ast.ImportFrom] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in legacy_modules:
            continue
        if any(alias.name == symbol for alias in node.names):
            targets.append(node)

    if not targets:
        return text, False

    newline = detect_newline(text)
    lines = text.splitlines(keepends=True)
    changed = False

    for node in reversed(targets):
        kept_aliases = [alias for alias in node.names if alias.name != symbol]
        start = node.lineno - 1
        end = int(getattr(node, "end_lineno", node.lineno))
        indent_match = re.match(r"\s*", lines[start])
        indent = indent_match.group(0) if indent_match else ""

        replacement: list[str] = []
        if kept_aliases:
            rendered = ", ".join(format_import_alias(alias) for alias in kept_aliases)
            module = node.module or ""
            replacement.append(f"{indent}from {module} import {rendered}{newline}")
        lines[start:end] = replacement
        changed = True

    return "".join(lines), changed


def has_unqualified_symbol_call(tree: ast.Module, symbol: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == symbol:
            return True
    return False


def module_docstring_end_line(tree: ast.Module) -> int:
    if not tree.body:
        return 0
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return int(getattr(first, "end_lineno", first.lineno))
    return 0


def import_block_end_line(tree: ast.Module) -> int:
    end = module_docstring_end_line(tree)
    start_idx = 1 if end else 0
    for node in tree.body[start_idx:]:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            break
        end = int(getattr(node, "end_lineno", node.lineno))
    return end


def preamble_line_count(lines: list[str]) -> int:
    count = 0
    if lines and lines[0].startswith("#!"):
        count = 1
    if len(lines) > count and CODING_RE.match(lines[count]):
        count += 1
    return count


def add_target_import(
    text: str,
    tree: ast.Module,
    target_module: str,
    symbol: str,
) -> str:
    import_line = f"from {target_module} import {symbol}"
    newline = detect_newline(text)
    if not text:
        return f"{import_line}{newline}"
    lines = text.splitlines(keepends=True)
    insert_after = max(import_block_end_line(tree), preamble_line_count(lines))
    insert_at = min(insert_after, len(lines))
    new_lines = lines[:insert_at] + [f"{import_line}{newline}"] + lines[insert_at:]
    return "".join(new_lines)


def process_file(
    path: Path,
    *,
    dry_run: bool,
    symbol: str,
    target_module: str,
    legacy_modules: set[str],
    auto_add_on_call: bool,
) -> tuple[bool, str]:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, "skip: non-utf8"
    except OSError as exc:
        return False, f"skip: read error ({exc})"

    try:
        tree = ast.parse(original)
    except SyntaxError as exc:
        return False, f"skip: syntax error ({exc.msg})"

    updated = original
    changed = False

    if legacy_modules:
        updated, normalized = normalize_legacy_symbol_imports(
            updated,
            tree,
            symbol=symbol,
            legacy_modules=legacy_modules,
        )
        if normalized:
            changed = True
            tree = ast.parse(updated)

    needs_target_import = changed or (
        auto_add_on_call and has_unqualified_symbol_call(tree, symbol)
    )
    if needs_target_import and not has_target_symbol_import(
        tree, target_module, symbol
    ):
        updated = add_target_import(
            updated,
            tree,
            target_module=target_module,
            symbol=symbol,
        )
        changed = True

    if not changed or updated == original:
        return False, "skip: no change"

    if not dry_run:
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return False, f"skip: write error ({exc})"
    return True, "updated"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generic import normalizer/checker for a symbol imported via "
            "`from <module> import <symbol>`."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Root directory or Python file to scan. Repeatable.",
    )
    parser.add_argument(
        "--target-module",
        required=True,
        help="Target module for canonical import, e.g. `foo.bar`.",
    )
    parser.add_argument(
        "--symbol",
        default="t",
        help="Imported symbol name to normalize (default: t).",
    )
    parser.add_argument(
        "--legacy-module",
        action="append",
        default=[],
        help=(
            "Legacy module to strip the symbol from, repeatable. "
            "Example: --legacy-module old.module"
        ),
    )
    parser.add_argument(
        "--auto-add-on-call",
        action="store_true",
        help="If enabled, add target import when unqualified `<symbol>(...)` call exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report files that would be updated.",
    )
    parser.add_argument(
        "--write-list",
        default="",
        help="Optional path to write updated file list (one per line).",
    )
    parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Return exit code 1 if any file would be/was updated.",
    )
    args = parser.parse_args()

    roots = [Path(r).resolve() for r in args.root]
    py_files = iter_python_files(roots)
    legacy_modules = set(args.legacy_module)

    updated_files: list[Path] = []
    skipped = 0

    for path in py_files:
        changed, message = process_file(
            path,
            dry_run=args.dry_run,
            symbol=args.symbol,
            target_module=args.target_module,
            legacy_modules=legacy_modules,
            auto_add_on_call=args.auto_add_on_call,
        )
        if changed:
            updated_files.append(path)
            print(f"[updated] {path}")
        else:
            skipped += 1
            if message.startswith("skip: syntax error") or message.startswith(
                "skip: read error"
            ):
                print(f"[{message}] {path}")

    print(
        f"Scanned {len(py_files)} file(s); "
        f"{'would update' if args.dry_run else 'updated'} {len(updated_files)} file(s); "
        f"skipped {skipped} file(s)."
    )

    if args.write_list:
        out_path = Path(args.write_list)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(f"{path}\n" for path in updated_files),
            encoding="utf-8",
        )
        print(f"Wrote updated file list: {out_path}")

    if args.fail_on_change and updated_files:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
