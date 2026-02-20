#!/usr/bin/env python3
"""Apply translated i18n draft markdown back to source files.

Input format is produced by translate_i18n_todo.py:
  - section header: ## `relative/path`
  - table row: | `line` | Source | Translation |

The script updates each target line with translated text while preserving
existing leading indentation and line ending.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
from dataclasses import dataclass
from pathlib import Path

HEADER_RE = re.compile(r"^## `(?P<path>[^`]+)`$")
LINE_CELL_RE = re.compile(r"^`(?P<line>\d+)`$")
HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
ATTR_RE = re.compile(
    r"(?P<name>[A-Za-z_][\w:-]*)=(?P<quote>['\"])(?P<value>[^'\"]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^'\"]*)(?P=quote)"
)
TAG_TEXT_RE = re.compile(
    r">(?P<text>[^<{]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^<{]*)<"
)
STRING_LITERAL_RE = re.compile(
    r"(?P<quote>['\"])(?P<text>[^'\"]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^'\"]*)(?P=quote)"
)


@dataclass(frozen=True)
class Edit:
    path: Path
    line: int
    key: str | None
    source: str
    translated: str


@dataclass(frozen=True)
class ApplyResult:
    path: Path
    changed: int
    skipped: int
    missing: bool


def split_md_row(line: str) -> list[str] | None:
    if not line.startswith("|") or not line.endswith("|"):
        return None
    body = line[1:-1]
    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in body:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if ch == "|":
            cells.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


def unescape_md_cell(text: str) -> str:
    return (
        text.replace("<br>", "\n")
        .replace("\\\\", "\\")
        .replace("\\|", "|")
        .replace("\r", "")
        .strip()
    )


def normalize_key_cell(text: str) -> str | None:
    key = unescape_md_cell(text).strip()
    if len(key) >= 2 and key.startswith("`") and key.endswith("`"):
        key = key[1:-1].strip()
    return key or None


def parse_translated_markdown(path: Path) -> list[Edit]:
    edits: list[Edit] = []
    current_path: Path | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        header_match = HEADER_RE.match(raw_line.strip())
        if header_match:
            current_path = Path(header_match.group("path"))
            continue

        if current_path is None or not raw_line.startswith("|"):
            continue
        if raw_line.startswith("|---|"):
            continue

        cells = split_md_row(raw_line)
        if not cells or len(cells) not in {3, 4}:
            continue

        line_match = LINE_CELL_RE.match(cells[0])
        if not line_match:
            continue

        line_no = int(line_match.group("line"))
        if len(cells) == 4:
            key = normalize_key_cell(cells[1])
            source = unescape_md_cell(cells[2])
            translated = unescape_md_cell(cells[3])
        else:
            key = None
            source = unescape_md_cell(cells[1])
            translated = unescape_md_cell(cells[2])
        if line_no <= 0 or not translated:
            continue

        edits.append(
            Edit(
                path=current_path,
                line=line_no,
                key=key,
                source=source,
                translated=translated,
            )
        )
    return edits


def line_ending_of(text: str) -> str:
    if text.endswith("\r\n"):
        return "\r\n"
    if text.endswith("\n"):
        return "\n"
    return ""


def build_template_line(
    old_noeol: str, key: str | None, template_func: str
) -> str | None:
    if not key:
        return None
    if not HAN_RE.search(old_noeol):
        return old_noeol

    attr_match = ATTR_RE.search(old_noeol)
    if attr_match:
        name = attr_match.group("name")
        bound_name = (
            name if (name.startswith(":") or name.startswith("v-bind:")) else f":{name}"
        )
        replacement = f"{bound_name}=\"{template_func}('{key}')\""
        return f"{old_noeol[: attr_match.start()]}{replacement}{old_noeol[attr_match.end() :]}"

    tag_text_match = TAG_TEXT_RE.search(old_noeol)
    if tag_text_match:
        replacement = f">{{{{ {template_func}('{key}') }}}}<"
        return (
            f"{old_noeol[: tag_text_match.start()]}"
            f"{replacement}"
            f"{old_noeol[tag_text_match.end() :]}"
        )

    str_match = STRING_LITERAL_RE.search(old_noeol)
    if str_match:
        replacement = f"{template_func}('{key}')"
        return f"{old_noeol[: str_match.start()]}{replacement}{old_noeol[str_match.end() :]}"

    return None


def apply_edits_to_file(
    project_root: Path,
    file_path: Path,
    file_edits: list[Edit],
    *,
    dry_run: bool,
    column: str,
    check_source: bool,
    template_func: str,
) -> ApplyResult:
    full_path = (project_root / file_path).resolve()
    if not full_path.is_file():
        return ApplyResult(
            path=file_path, changed=0, skipped=len(file_edits), missing=True
        )

    lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )
    changed = 0
    skipped = 0

    for edit in sorted(file_edits, key=lambda x: x.line):
        idx = edit.line - 1
        if idx < 0 or idx >= len(lines):
            skipped += 1
            continue

        old_line = lines[idx]
        old_noeol = old_line.rstrip("\r\n")
        if check_source and old_noeol.strip() != edit.source:
            skipped += 1
            continue
        if column == "template":
            template_line = build_template_line(old_noeol, edit.key, template_func)
            if template_line is None:
                skipped += 1
                continue
            new_noeol = template_line
        else:
            chosen_text = edit.translated if column == "translated" else edit.source
            if not chosen_text:
                skipped += 1
                continue
            if "\n" in chosen_text:
                skipped += 1
                continue
            leading = old_noeol[: len(old_noeol) - len(old_noeol.lstrip())]
            new_noeol = f"{leading}{chosen_text.lstrip()}"
        if new_noeol == old_noeol:
            continue

        lines[idx] = new_noeol + line_ending_of(old_line)
        changed += 1

    if changed > 0 and not dry_run:
        full_path.write_text("".join(lines), encoding="utf-8")

    return ApplyResult(path=file_path, changed=changed, skipped=skipped, missing=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply translated markdown draft lines back to source files."
    )
    parser.add_argument("--translated", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--column",
        choices=["template", "translated", "source"],
        default="template",
        help="Apply mode: template uses t('key'); translated/source write literal text.",
    )
    parser.add_argument(
        "--template-func",
        default="t",
        help="Function name used for template mode, e.g. t -> t('key').",
    )
    parser.add_argument(
        "--ignore-source-check",
        action="store_true",
        help="Apply by line number only, even when current line content differs from source column.",
    )
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    translated_path = args.translated.resolve()
    if not translated_path.is_file():
        raise FileNotFoundError(f"Translated markdown not found: {translated_path}")

    project_root = args.project_root.resolve()
    edits = parse_translated_markdown(translated_path)
    if not edits:
        print("No edits parsed from translated markdown.")
        return 0

    grouped: dict[Path, list[Edit]] = {}
    for edit in edits:
        grouped.setdefault(edit.path, []).append(edit)

    results: list[ApplyResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                apply_edits_to_file,
                project_root,
                rel_path,
                rel_edits,
                dry_run=args.dry_run,
                column=args.column,
                check_source=not args.ignore_source_check,
                template_func=args.template_func,
            )
            for rel_path, rel_edits in grouped.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_changed = sum(r.changed for r in results)
    total_skipped = sum(r.skipped for r in results)
    missing_files = [r.path.as_posix() for r in results if r.missing]

    print(f"Parsed edits: {len(edits)}")
    print(f"Target files: {len(grouped)}")
    print(f"Changed lines: {total_changed}")
    print(f"Skipped lines: {total_skipped}")
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        for p in sorted(missing_files)[:20]:
            print(f"- {p}")
        if len(missing_files) > 20:
            print(f"... and {len(missing_files) - 20} more")
    if args.dry_run:
        print("Dry run only. No files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
