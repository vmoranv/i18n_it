#!/usr/bin/env python3
"""Extract non-comment Chinese hardcoded text into a markdown TODO checklist.

Usage:
    python extract_i18n_todo.py
    python extract_i18n_todo.py --root .
    python extract_i18n_todo.py --root . --output custom.todo.md
    python extract_i18n_todo.py --root . --key-prefix hardcoded
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")

PYTHON_EXTENSIONS = {".py", ".pyi"}
JS_LIKE_EXTENSIONS = {
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".vue",
    ".svelte",
    ".astro",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".php",
    ".cs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".m",
    ".mm",
    ".swift",
    ".scala",
    ".dart",
    ".html",
    ".htm",
    ".xml",
    ".svg",
    ".xhtml",
    ".mdx",
}
HASH_COMMENT_EXTENSIONS = {
    ".rb",
    ".sh",
    ".bash",
    ".zsh",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
}
DEFAULT_EXTENSIONS = PYTHON_EXTENSIONS | JS_LIKE_EXTENSIONS | HASH_COMMENT_EXTENSIONS
HTML_COMMENT_EXTENSIONS = {
    ".vue",
    ".svelte",
    ".astro",
    ".html",
    ".htm",
    ".xml",
    ".svg",
    ".xhtml",
    ".mdx",
}
DEFAULT_EXCLUDE_DIR_NAMES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
DEFAULT_EXCLUDE_PATH_FRAGMENTS = {
    "i18n/locales/",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    snippet: str


def has_han(text: str) -> bool:
    return bool(HAN_RE.search(text))


def should_exclude(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    rel_posix = rel.as_posix().lower()

    if any(part.lower() in DEFAULT_EXCLUDE_DIR_NAMES for part in rel.parts):
        return True
    if any(fragment in rel_posix for fragment in DEFAULT_EXCLUDE_PATH_FRAGMENTS):
        return True
    if path.name.lower().endswith(".min.js"):
        return True
    return False


def parse_extensions(raw: str) -> set[str]:
    extensions: set[str] = set()
    for part in re.split(r"[,;\s]+", raw.strip()):
        if not part:
            continue
        ext = part.strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.add(ext)
    return extensions


def collect_files(root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if should_exclude(path, root):
            continue
        files.append(path)
    files.sort()
    return files


def default_output_path(root: Path) -> Path:
    root_name = root.name or "project"
    return root / f"i18n.todo.scan.{root_name}.md"


def collect_python_ignored_string_lines(source: str) -> set[int]:
    """Collect line numbers of standalone string expression statements.

    This includes:
    - canonical docstrings (module/class/function first string statement)
    - no-op string literals used as comment-like markers inside code blocks
    """
    ignored_lines: set[int] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ignored_lines

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue

        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if not isinstance(start, int):
            continue
        if not isinstance(end, int):
            end = start
        for line_no in range(start, end + 1):
            ignored_lines.add(line_no)
    return ignored_lines


def extract_from_python(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    ignored_string_lines = collect_python_ignored_string_lines(text)
    line_to_snippet: dict[int, str] = {}

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in tokens:
            tok_type = tok.type
            tok_str = tok.string
            line_no = tok.start[0]

            if tok_type == tokenize.COMMENT:
                continue
            if tok_type == tokenize.STRING and line_no in ignored_string_lines:
                continue
            if not has_han(tok_str):
                continue

            if 1 <= line_no <= len(lines):
                snippet = lines[line_no - 1].strip()
            else:
                snippet = tok_str.strip()
            if snippet:
                line_to_snippet.setdefault(line_no, snippet)
    except tokenize.TokenError:
        # Fallback when tokenization fails: use plain line scan.
        for idx, line in enumerate(lines, start=1):
            if has_han(line):
                line_to_snippet.setdefault(idx, line.strip())

    return [
        Finding(path=path, line=k, snippet=v)
        for k, v in sorted(line_to_snippet.items())
    ]


def strip_js_like_comments(text: str, include_html_comments: bool) -> str:
    normal = 0
    line_comment = 1
    block_comment = 2
    single_quote = 3
    double_quote = 4
    template_quote = 5
    html_comment = 6

    state = normal
    out: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if state == normal:
            if include_html_comments and text.startswith("<!--", i):
                out.append(" " * 4)
                i += 4
                state = html_comment
                continue
            if ch == "/" and nxt == "/":
                out.append("  ")
                i += 2
                state = line_comment
                continue
            if ch == "/" and nxt == "*":
                out.append("  ")
                i += 2
                state = block_comment
                continue
            if ch == "'":
                out.append(ch)
                i += 1
                state = single_quote
                continue
            if ch == '"':
                out.append(ch)
                i += 1
                state = double_quote
                continue
            if ch == "`":
                out.append(ch)
                i += 1
                state = template_quote
                continue

            out.append(ch)
            i += 1
            continue

        if state == line_comment:
            if ch in "\r\n":
                out.append(ch)
                state = normal
            else:
                out.append(" ")
            i += 1
            continue

        if state == block_comment:
            if ch == "*" and nxt == "/":
                out.append("  ")
                i += 2
                state = normal
            else:
                out.append(ch if ch in "\r\n" else " ")
                i += 1
            continue

        if state == html_comment:
            if text.startswith("-->", i):
                out.append(" " * 3)
                i += 3
                state = normal
            else:
                out.append(ch if ch in "\r\n" else " ")
                i += 1
            continue

        if state == single_quote:
            if ch == "\\" and i + 1 < n:
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            out.append(ch)
            i += 1
            if ch == "'":
                state = normal
            continue

        if state == double_quote:
            if ch == "\\" and i + 1 < n:
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            out.append(ch)
            i += 1
            if ch == '"':
                state = normal
            continue

        if state == template_quote:
            if ch == "\\" and i + 1 < n:
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            out.append(ch)
            i += 1
            if ch == "`":
                state = normal
            continue

    return "".join(out)


def extract_from_js_like(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    include_html_comments = path.suffix.lower() in HTML_COMMENT_EXTENSIONS
    original_lines = text.splitlines()
    line_to_snippet: dict[int, str] = {}

    for idx, raw_line in enumerate(original_lines, start=1):
        snippet = raw_line.strip()
        if not snippet:
            continue
        # Defensive filter: do not emit pure comment lines for JS/TS/Vue.
        # This also protects against parser state drift in mixed .vue files.
        if is_js_or_html_comment_line(snippet):
            continue
        # Re-strip comments on the current line and only keep lines whose
        # non-comment part still contains Chinese.
        line_without_comments = strip_js_like_comments(
            raw_line,
            include_html_comments=include_html_comments,
        ).strip()
        if not has_han(line_without_comments):
            continue
        line_to_snippet.setdefault(idx, snippet)

    return [
        Finding(path=path, line=k, snippet=v)
        for k, v in sorted(line_to_snippet.items())
    ]


def strip_hash_line_comments(line: str) -> str:
    out: list[str] = []
    in_single = False
    in_double = False
    escaped = False

    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            continue
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out)


def extract_from_hash_like(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    original_lines = text.splitlines()
    line_to_snippet: dict[int, str] = {}

    for idx, raw_line in enumerate(original_lines, start=1):
        snippet = raw_line.strip()
        if not snippet:
            continue
        if snippet.startswith("#"):
            continue
        line_without_comments = strip_hash_line_comments(raw_line).strip()
        if not has_han(line_without_comments):
            continue
        line_to_snippet.setdefault(idx, snippet)

    return [
        Finding(path=path, line=k, snippet=v)
        for k, v in sorted(line_to_snippet.items())
    ]


def is_js_or_html_comment_line(snippet: str) -> bool:
    s = snippet.lstrip()
    return (
        s.startswith("//")
        or s.startswith("/*")
        or s.startswith("*/")
        or s.startswith("*")
        or s.startswith("<!--")
        or s.startswith("-->")
    )


def normalize_snippet(text: str, max_len: int = 140) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_len:
        return collapsed
    return f"{collapsed[: max_len - 3]}..."


def sanitize_key_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token.lower() or "item"


def normalize_key_prefix(raw: str) -> str:
    base = raw.strip().strip(".")
    if not base:
        return "hardcoded"
    parts = [sanitize_key_token(part) for part in base.split(".") if part.strip()]
    return ".".join(parts) if parts else "hardcoded"


def build_i18n_key(rel_path: str, line_no: int, key_prefix: str) -> str:
    no_suffix = Path(rel_path).with_suffix("")
    parts = [sanitize_key_token(part) for part in no_suffix.parts if part]
    return ".".join([key_prefix, *parts, f"line_{line_no}"])


def to_markdown(findings: Iterable[Finding], root: Path, key_prefix: str) -> str:
    findings_sorted = sorted(findings, key=lambda x: (x.path.as_posix(), x.line))
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings_sorted:
        grouped[finding.path.relative_to(root).as_posix()].append(finding)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = [
        "# i18n TODO",
        "",
        f"- Generated at (UTC): `{now}`",
        f"- Root: `{root.as_posix()}`",
        f"- Files with candidates: `{len(grouped)}`",
        f"- Total candidate lines: `{len(findings_sorted)}`",
        "",
        "> Comment/docstring lines are excluded as much as possible. Review before replacement.",
        "",
    ]

    if not grouped:
        lines.append("No Chinese hardcoded text candidates found.")
        lines.append("")
        return "\n".join(lines)

    for rel_path in sorted(grouped):
        lines.append(f"## `{rel_path}`")
        for item in grouped[rel_path]:
            snippet = normalize_snippet(item.snippet).replace("`", "\\`")
            key = build_i18n_key(rel_path, item.line, key_prefix)
            lines.append(f"- [ ] `{rel_path}:{item.line}` [key:`{key}`] {snippet}")
        lines.append("")

    return "\n".join(lines)


def extract_findings(root: Path, files: list[Path]) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    errors: list[str] = []
    for path in files:
        try:
            suffix = path.suffix.lower()
            if suffix in PYTHON_EXTENSIONS:
                findings.extend(extract_from_python(path))
            elif suffix in HASH_COMMENT_EXTENSIONS:
                findings.extend(extract_from_hash_like(path))
            else:
                findings.extend(extract_from_js_like(path))
        except Exception as exc:  # noqa: BLE001
            rel = path.relative_to(root).as_posix()
            errors.append(f"{rel}: {exc}")
    return findings, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract non-comment Chinese hardcoded text into i18n TODO markdown."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root path to scan. Default: current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output markdown file path. Default: <root>/i18n.todo.scan.<root_name>.md"
        ),
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help=(
            "Comma/space/semicolon-separated file extensions to scan, "
            "for example: .py,.ts,.tsx,.vue,.java"
        ),
    )
    parser.add_argument(
        "--key-prefix",
        default="hardcoded",
        help="Prefix for generated i18n keys written into TODO items.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    key_prefix = normalize_key_prefix(args.key_prefix)
    extensions = parse_extensions(args.extensions)
    if not extensions:
        raise ValueError("No valid extensions parsed from --extensions")
    files = collect_files(root, extensions)
    findings, errors = extract_findings(root, files)
    content = to_markdown(findings, root, key_prefix)

    output_path = args.output.resolve() if args.output else default_output_path(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    print(f"Scanned files: {len(files)}")
    print(f"Scanned extensions: {len(extensions)}")
    print(f"Found candidate lines: {len(findings)}")
    print(f"Output: {output_path}")
    if errors:
        print(f"Warnings: {len(errors)} file(s) failed to parse")
        for err in errors[:20]:
            print(f"- {err}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
