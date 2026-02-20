#!/usr/bin/env python3
"""Sync translated draft markdown into locale files.

Input markdown format is produced by translate_i18n_todo.py:
  - section header: ## `relative/path`
  - table rows:
    - | `line` | Source | Translation |
    - or | `line` | `key` | Source | Translation |

User must provide a mapping file to route source files to target locale files.
Supported target formats: JSON (`.json`) and Fluent (`.ftl`).
This script only updates locale files; it does not modify source code.

Mapping file formats (JSON):
1) Simple object:
{
  "src/views/**": "features/ui.json",
  "src/components/chat/**": { "file": "features/chat.json", "prefix": "chat" }
}

2) Explicit list:
{
  "mappings": [
    { "source": "src/views/**", "file": "features/ui.json" },
    { "source": "src/components/chat/**", "file": "features/chat.json", "prefix": "chat" }
  ]
}

Key generation:
- If mapping has "prefix", key becomes "<prefix>.line_<line_no>".
- Otherwise, prefix is auto-derived from source path (without suffix), then:
  "<auto_prefix>.line_<line_no>".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HEADER_RE = re.compile(r"^## `(?P<path>[^`]+)`$")
LINE_CELL_RE = re.compile(r"^`(?P<line>\d+)`$")
SOURCE_TODO_RE = re.compile(r"^- Source TODO:\s*`(?P<path>[^`]+)`\s*$")
TODO_LINE_RE = re.compile(
    r"^- \[(?P<mark>[ xX])\] `(?P<location>[^`]+)`(?:\s+\[key:`(?P<key>[^`]+)`\])?\s*(?P<snippet>.*)$"
)
HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
ASSIGNMENT_RE = re.compile(r"^\s*(?P<lhs>[A-Za-z0-9_.-]+)\s*=\s*(?P<rhs>.+?)\s*$")
TAG_TEXT_RE = re.compile(r">(?P<text>[^<>]+?)<")
I18N_CALL_KEY_RE = re.compile(
    r"(?:\b(?:\$?t|i18n\.t)\s*\(\s*['\"](?P<k1>[A-Za-z0-9_.-]+)['\"]\s*\)|"
    r"['\"](?P<k2>[A-Za-z0-9_.-]+)['\"]\s*\|\|\s*['\"][^'\"]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^'\"]*['\"])"
)
GLOB_CHARS = set("*?[]")


@dataclass(frozen=True)
class DraftItem:
    rel_path: Path
    line_no: int
    key: str | None
    source_text: str
    translated_text: str


@dataclass(frozen=True)
class TargetSpec:
    file: Path
    prefix: str | None


@dataclass(frozen=True)
class MappingRule:
    pattern: str
    spec: TargetSpec
    is_glob: bool


@dataclass(frozen=True)
class LocaleEntry:
    key: str
    value: str
    source_rel_path: str
    line_no: int


@dataclass(frozen=True)
class FileSyncResult:
    file: Path
    total: int
    added: int
    updated: int
    kept: int
    errors: list[str]


FTL_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


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


def parse_translated_markdown(
    path: Path, column: Literal["translated", "source"]
) -> list[DraftItem]:
    items: list[DraftItem] = []
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
        if line_no <= 0:
            continue

        if len(cells) == 4:
            key = normalize_key_cell(cells[1])
            source = unescape_md_cell(cells[2])
            translated = unescape_md_cell(cells[3])
        else:
            key = None
            source = unescape_md_cell(cells[1])
            translated = unescape_md_cell(cells[2])
        value = translated if column == "translated" else source
        if not value:
            continue

        items.append(
            DraftItem(
                rel_path=current_path,
                line_no=line_no,
                key=key,
                source_text=source,
                translated_text=translated,
            )
        )
    return items


def has_han(text: str) -> bool:
    return bool(HAN_RE.search(text))


def extract_quoted_texts(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        quote = text[i]
        if quote not in {"'", '"', "`"}:
            i += 1
            continue
        i += 1
        buf: list[str] = []
        while i < n:
            ch = text[i]
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                i += 1
                break
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return out


def extract_tag_texts(text: str) -> list[str]:
    return [
        m.group("text").strip()
        for m in TAG_TEXT_RE.finditer(text)
        if m.group("text").strip()
    ]


def try_extract_assignment_rhs(source_text: str, target_text: str) -> str | None:
    source_match = ASSIGNMENT_RE.match(source_text.strip())
    if not source_match:
        return None
    source_rhs = source_match.group("rhs").strip()
    if not has_han(source_rhs):
        return None

    target_match = ASSIGNMENT_RE.match(target_text.strip())
    if target_match and target_match.group("lhs") == source_match.group("lhs"):
        rhs = target_match.group("rhs").strip()
        if rhs:
            return rhs

    lhs, sep, rhs = target_text.partition("=")
    if sep and lhs.strip() == source_match.group("lhs"):
        rhs = rhs.strip()
        if rhs:
            return rhs
    return None


def take_by_han_positions(
    source_values: list[str], target_values: list[str]
) -> str | None:
    han_positions = [idx for idx, text in enumerate(source_values) if has_han(text)]
    if not han_positions:
        return None
    selected = [
        target_values[idx].strip()
        for idx in han_positions
        if idx < len(target_values) and target_values[idx].strip()
    ]
    if not selected:
        return None
    return " ".join(selected)


def try_extract_from_quotes(source_text: str, target_text: str) -> str | None:
    source_values = extract_quoted_texts(source_text)
    target_values = extract_quoted_texts(target_text)
    if not source_values or not target_values:
        return None
    return take_by_han_positions(source_values, target_values)


def try_extract_from_tag_text(source_text: str, target_text: str) -> str | None:
    source_values = extract_tag_texts(source_text)
    target_values = extract_tag_texts(target_text)
    if not source_values or not target_values:
        return None
    return take_by_han_positions(source_values, target_values)


def extract_locale_value(
    source_text: str,
    target_text: str,
    mode: Literal["smart", "raw"],
) -> str:
    value = target_text.strip()
    if mode == "raw":
        return value

    for extractor in (
        try_extract_assignment_rhs,
        try_extract_from_quotes,
        try_extract_from_tag_text,
    ):
        extracted = extractor(source_text, target_text)
        if extracted:
            return extracted
    return value


def parse_source_todo_path(path: Path) -> Path | None:
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = SOURCE_TODO_RE.match(raw_line.strip())
        if not match:
            continue
        raw_path = match.group("path").strip()
        if not raw_path:
            return None
        return Path(raw_path)
    return None


def flatten_json_leaf_keys(node: object, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(k, str) or not k:
                continue
            next_prefix = f"{prefix}.{k}" if prefix else k
            keys |= flatten_json_leaf_keys(v, next_prefix)
    elif isinstance(node, list):
        return keys
    elif isinstance(node, str):
        if prefix:
            keys.add(prefix)
    return keys


def build_json_key_file_index(i18n_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in i18n_root.rglob("*.json"):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw:
                continue
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        rel = path.relative_to(i18n_root)
        for key in flatten_json_leaf_keys(payload):
            index.setdefault(key, []).append(rel)
    return index


def extract_existing_i18n_key(source_text: str) -> str | None:
    match = I18N_CALL_KEY_RE.search(source_text)
    if not match:
        return None
    key = (match.group("k1") or match.group("k2") or "").strip()
    return key or None


def split_namespaced_key(key: str) -> tuple[str, str] | None:
    parts = [p for p in key.split(".") if p]
    if len(parts) < 3:
        return None
    return ".".join(parts[:2]), ".".join(parts[2:])


def resolve_namespaced_json_target(key: str) -> tuple[Path, str] | None:
    split = split_namespaced_key(key)
    if not split:
        return None
    namespace, inner = split
    ns_to_file = {
        "core.common": Path("core/common.json"),
        "core.header": Path("core/header.json"),
        "core.navigation": Path("core/navigation.json"),
        "core.shared": Path("core/shared.json"),
        "messages.success": Path("messages/success.json"),
        "messages.errors": Path("messages/errors.json"),
    }
    if namespace.startswith("features."):
        feature = namespace.split(".", 1)[1]
        return Path(f"features/{feature}.json"), inner
    target = ns_to_file.get(namespace)
    if not target:
        return None
    return target, inner


def update_todo_checkmarks(todo_path: Path, completed: dict[str, str]) -> int:
    raw = todo_path.read_text(encoding="utf-8", errors="ignore")
    had_trailing_newline = raw.endswith("\n")
    changed = 0
    new_lines: list[str] = []

    for line in raw.splitlines():
        match = TODO_LINE_RE.match(line)
        if not match:
            new_lines.append(line)
            continue

        location = match.group("location")
        snippet = (match.group("snippet") or "").strip()
        existing_key = (match.group("key") or "").strip()
        if location not in completed:
            new_lines.append(line)
            continue

        key = completed[location] or existing_key
        rebuilt = f"- [x] `{location}`"
        if key:
            rebuilt += f" [key:`{key}`]"
        if snippet:
            rebuilt += f" {snippet}"

        if rebuilt != line:
            changed += 1
        new_lines.append(rebuilt)

    out = "\n".join(new_lines)
    if had_trailing_newline:
        out += "\n"
    if out != raw:
        todo_path.write_text(out, encoding="utf-8")
    return changed


def normalize_rel_path(path_text: str) -> str:
    return path_text.replace("\\", "/").strip()


def has_glob(pattern: str) -> bool:
    return any(ch in GLOB_CHARS for ch in pattern)


def parse_target_spec(raw: object) -> TargetSpec:
    if isinstance(raw, str):
        return TargetSpec(file=Path(normalize_rel_path(raw)), prefix=None)
    if isinstance(raw, dict):
        file_raw = raw.get("file")
        if not isinstance(file_raw, str) or not file_raw.strip():
            raise ValueError("Mapping object must contain non-empty 'file'")
        prefix_raw = raw.get("prefix")
        prefix = (
            prefix_raw.strip()
            if isinstance(prefix_raw, str) and prefix_raw.strip()
            else None
        )
        return TargetSpec(file=Path(normalize_rel_path(file_raw)), prefix=prefix)
    raise ValueError("Mapping target must be string or object")


def load_mapping_rules(mapping_path: Path) -> list[MappingRule]:
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    rules: list[MappingRule] = []

    if isinstance(payload, dict) and isinstance(payload.get("mappings"), list):
        for row in payload["mappings"]:
            if not isinstance(row, dict):
                raise ValueError("Each item in 'mappings' must be an object")
            source = row.get("source")
            file_raw = row.get("file")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Each mapping item must include non-empty 'source'")
            target = {"file": file_raw, "prefix": row.get("prefix")}
            spec = parse_target_spec(target)
            pattern = normalize_rel_path(source)
            rules.append(
                MappingRule(pattern=pattern, spec=spec, is_glob=has_glob(pattern))
            )
    elif isinstance(payload, dict):
        for source, target in payload.items():
            if not isinstance(source, str) or source.startswith("$"):
                continue
            pattern = normalize_rel_path(source)
            spec = parse_target_spec(target)
            rules.append(
                MappingRule(pattern=pattern, spec=spec, is_glob=has_glob(pattern))
            )
    else:
        raise ValueError("Mapping JSON must be an object")

    if not rules:
        raise ValueError("No valid mapping rules found")
    return rules


def resolve_target_spec(
    source_rel_path: str, rules: list[MappingRule]
) -> TargetSpec | None:
    # Exact matches first
    for rule in rules:
        if not rule.is_glob and source_rel_path == rule.pattern:
            return rule.spec
    # Then glob matches in declaration order
    for rule in rules:
        if rule.is_glob and fnmatch.fnmatch(source_rel_path, rule.pattern):
            return rule.spec
    return None


def sanitize_key_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token.lower() or "item"


def auto_prefix_for_path(rel_path: Path) -> str:
    no_suffix = rel_path.with_suffix("")
    parts = [sanitize_key_token(part) for part in no_suffix.parts if part]
    return ".".join(parts) if parts else "text"


def make_key(prefix: str, line_no: int, used: dict[str, str], value: str) -> str:
    base = f"{prefix}.line_{line_no}"
    key = base
    dup = 2
    while key in used and used[key] != value:
        key = f"{base}_dup{dup}"
        dup += 1
    used[key] = value
    return key


def ensure_in_root(root: Path, rel: Path) -> Path:
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Target path escapes i18n root: {rel.as_posix()}")
    return candidate


def set_nested_value(
    data: dict,
    key: str,
    value: str,
    if_exists: Literal["overwrite", "keep", "error"],
) -> tuple[Literal["added", "updated", "kept"], str | None]:
    parts = [p for p in key.split(".") if p]
    if not parts:
        return "kept", "Empty key"

    node = data
    for part in parts[:-1]:
        current = node.get(part)
        if current is None:
            node[part] = {}
            current = node[part]
        if not isinstance(current, dict):
            return "kept", f"Cannot set nested key under non-object segment: {part}"
        node = current

    leaf = parts[-1]
    if leaf not in node:
        node[leaf] = value
        return "added", None

    old = node[leaf]
    if old == value:
        return "kept", None

    if if_exists == "keep":
        return "kept", None
    if if_exists == "error":
        return "kept", f"Key already exists with different value: {key}"

    node[leaf] = value
    return "updated", None


def sync_one_json_file(
    i18n_root: Path,
    rel_file: Path,
    entries: list[LocaleEntry],
    *,
    if_exists: Literal["overwrite", "keep", "error"],
    dry_run: bool,
) -> FileSyncResult:
    errors: list[str] = []
    full_path = ensure_in_root(i18n_root, rel_file)

    if full_path.is_file():
        raw = full_path.read_text(encoding="utf-8", errors="ignore").strip()
        data = json.loads(raw) if raw else {}
    else:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"Locale file must be a JSON object: {rel_file.as_posix()}")

    added = 0
    updated = 0
    kept = 0
    for entry in entries:
        action, err = set_nested_value(data, entry.key, entry.value, if_exists)
        if err:
            errors.append(
                f"{entry.source_rel_path}:{entry.line_no} -> {entry.key}: {err}"
            )
        if action == "added":
            added += 1
        elif action == "updated":
            updated += 1
        else:
            kept += 1

    if (added > 0 or updated > 0) and not dry_run:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return FileSyncResult(
        file=rel_file,
        total=len(entries),
        added=added,
        updated=updated,
        kept=kept,
        errors=errors,
    )


def to_ftl_key(key: str) -> str:
    converted = key.replace(".", "-")
    converted = re.sub(r"[^A-Za-z0-9_-]+", "-", converted)
    converted = re.sub(r"-+", "-", converted).strip("-")
    if not converted:
        converted = "item"
    if not re.match(r"^[A-Za-z]", converted):
        converted = f"k-{converted}"
    if not FTL_KEY_RE.match(converted):
        converted = "item"
    return converted


def normalize_ftl_value(value: str) -> str:
    compact = value.replace("\r", "").replace("\n", " ").strip()
    compact = re.sub(r"\s+", " ", compact)
    return compact


def parse_ftl_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, _, val = line.partition("=")
    key = key.strip()
    if not key:
        return None
    return key, val.strip()


def sync_one_ftl_file(
    i18n_root: Path,
    rel_file: Path,
    entries: list[LocaleEntry],
    *,
    if_exists: Literal["overwrite", "keep", "error"],
    dry_run: bool,
) -> FileSyncResult:
    errors: list[str] = []
    full_path = ensure_in_root(i18n_root, rel_file)

    lines: list[str]
    if full_path.is_file():
        lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        lines = []

    key_to_index: dict[str, int] = {}
    key_to_value: dict[str, str] = {}
    for idx, line in enumerate(lines):
        parsed = parse_ftl_line(line)
        if not parsed:
            continue
        key, val = parsed
        key_to_index[key] = idx
        key_to_value[key] = val

    added = 0
    updated = 0
    kept = 0
    for entry in entries:
        ftl_key = to_ftl_key(entry.key)
        ftl_val = normalize_ftl_value(entry.value)
        if not ftl_val:
            kept += 1
            continue

        if ftl_key not in key_to_index:
            lines.append(f"{ftl_key} = {ftl_val}")
            key_to_index[ftl_key] = len(lines) - 1
            key_to_value[ftl_key] = ftl_val
            added += 1
            continue

        old_val = key_to_value.get(ftl_key, "")
        if old_val == ftl_val:
            kept += 1
            continue

        if if_exists == "keep":
            kept += 1
            continue
        if if_exists == "error":
            errors.append(
                f"{entry.source_rel_path}:{entry.line_no} -> {ftl_key}: key exists with different value"
            )
            kept += 1
            continue

        idx = key_to_index[ftl_key]
        lines[idx] = f"{ftl_key} = {ftl_val}"
        key_to_value[ftl_key] = ftl_val
        updated += 1

    if (added > 0 or updated > 0) and not dry_run:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return FileSyncResult(
        file=rel_file,
        total=len(entries),
        added=added,
        updated=updated,
        kept=kept,
        errors=errors,
    )


def sync_one_locale_file(
    i18n_root: Path,
    rel_file: Path,
    entries: list[LocaleEntry],
    *,
    if_exists: Literal["overwrite", "keep", "error"],
    dry_run: bool,
) -> FileSyncResult:
    suffix = rel_file.suffix.lower()
    if suffix == ".json":
        return sync_one_json_file(
            i18n_root,
            rel_file,
            entries,
            if_exists=if_exists,
            dry_run=dry_run,
        )
    if suffix == ".ftl":
        return sync_one_ftl_file(
            i18n_root,
            rel_file,
            entries,
            if_exists=if_exists,
            dry_run=dry_run,
        )
    raise ValueError(f"Unsupported locale file type: {rel_file.as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync translated markdown draft into locale files (.json/.ftl)."
    )
    parser.add_argument("--translated", type=Path, required=True)
    parser.add_argument(
        "--i18n-root",
        type=Path,
        required=True,
        help="Root directory containing locale files (.json/.ftl).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="Mapping JSON file from source paths to locale target files.",
    )
    parser.add_argument(
        "--column",
        choices=["translated", "source"],
        default="translated",
        help="Use which column from draft markdown as locale value.",
    )
    parser.add_argument(
        "--value-mode",
        choices=["smart", "raw"],
        default="smart",
        help=(
            "How to build locale value from draft row. "
            "'smart' extracts only localizable content from code lines; "
            "'raw' keeps original whole-row text."
        ),
    )
    parser.add_argument(
        "--if-exists",
        choices=["overwrite", "keep", "error"],
        default="overwrite",
        help="Behavior when locale key already exists with different value.",
    )
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="Skip draft entries without mapping instead of failing.",
    )
    parser.add_argument(
        "--require-existing-target-files",
        action="store_true",
        help=(
            "Fail if any mapped locale target file does not already exist under --i18n-root. "
            "Useful to prevent accidental writes to wrong locale paths."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mark-checked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update source TODO checklist to [x] for successfully applied entries.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional report output path.",
    )
    args = parser.parse_args()

    translated_path = args.translated.resolve()
    i18n_root = args.i18n_root.resolve()
    mapping_path = args.mapping.resolve()

    if not translated_path.is_file():
        raise FileNotFoundError(f"Translated draft not found: {translated_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    if not i18n_root.is_dir():
        raise NotADirectoryError(f"i18n root not found: {i18n_root}")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    draft_items = parse_translated_markdown(translated_path, args.column)
    source_todo_path = parse_source_todo_path(translated_path)
    if not draft_items:
        print("No draft items found.")
        return 0

    rules = load_mapping_rules(mapping_path)
    json_key_index = build_json_key_file_index(i18n_root)

    grouped_entries: dict[Path, list[LocaleEntry]] = {}
    used_keys_per_file: dict[str, dict[str, str]] = {}
    completed_locations: dict[str, str] = {}
    unmapped: list[str] = []

    for item in draft_items:
        src_rel = normalize_rel_path(item.rel_path.as_posix())
        spec = resolve_target_spec(src_rel, rules)
        if spec is None:
            unmapped.append(f"{src_rel}:{item.line_no}")
            continue

        existing_key = extract_existing_i18n_key(item.source_text)
        if existing_key:
            namespaced = resolve_namespaced_json_target(existing_key)
            if namespaced is not None:
                spec = TargetSpec(file=namespaced[0], prefix=None)
                existing_key = namespaced[1]
            candidates = json_key_index.get(existing_key, [])
            if len(candidates) == 1:
                spec = TargetSpec(file=candidates[0], prefix=None)

        prefix = spec.prefix or auto_prefix_for_path(item.rel_path)
        file_key = spec.file.as_posix()
        used = used_keys_per_file.setdefault(file_key, {})
        row_text = (
            item.translated_text if args.column == "translated" else item.source_text
        )
        value = extract_locale_value(
            source_text=item.source_text,
            target_text=row_text,
            mode=args.value_mode,
        )
        key = (
            existing_key
            or item.key
            or make_key(prefix=prefix, line_no=item.line_no, used=used, value=value)
        )

        grouped_entries.setdefault(spec.file, []).append(
            LocaleEntry(
                key=key,
                value=value,
                source_rel_path=src_rel,
                line_no=item.line_no,
            )
        )
        completed_locations[f"{src_rel}:{item.line_no}"] = key

    if unmapped and not args.allow_unmapped:
        preview = "\n".join(f"- {x}" for x in unmapped[:50])
        more = f"\n... and {len(unmapped) - 50} more" if len(unmapped) > 50 else ""
        raise ValueError(
            "Unmapped draft entries found. Add mapping rules or use --allow-unmapped.\n"
            f"{preview}{more}"
        )

    if args.require_existing_target_files:
        missing_targets: list[str] = []
        for rel_file in grouped_entries:
            full = (i18n_root / rel_file).resolve()
            if not full.is_relative_to(i18n_root):
                raise ValueError(
                    f"Target path escapes i18n root: {rel_file.as_posix()}"
                )
            if not full.exists():
                missing_targets.append(rel_file.as_posix())
        if missing_targets:
            preview = "\n".join(f"- {p}" for p in sorted(missing_targets)[:50])
            more = (
                f"\n... and {len(missing_targets) - 50} more"
                if len(missing_targets) > 50
                else ""
            )
            raise FileNotFoundError(
                "Mapped locale target files do not exist (use corrected mapping or remove "
                "--require-existing-target-files):\n"
                f"{preview}{more}"
            )

    results: list[FileSyncResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                sync_one_locale_file,
                i18n_root,
                rel_file,
                entries,
                if_exists=args.if_exists,
                dry_run=args.dry_run,
            )
            for rel_file, entries in grouped_entries.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_entries = sum(r.total for r in results)
    total_added = sum(r.added for r in results)
    total_updated = sum(r.updated for r in results)
    total_kept = sum(r.kept for r in results)
    all_errors = [err for r in results for err in r.errors]

    print(f"Draft items: {len(draft_items)}")
    print(f"Mapped entries: {total_entries}")
    print(f"Target locale files: {len(results)}")
    print(f"Added keys: {total_added}")
    print(f"Updated keys: {total_updated}")
    print(f"Kept keys: {total_kept}")
    print(f"Unmapped entries: {len(unmapped)}")
    print(f"Errors: {len(all_errors)}")
    if args.dry_run:
        print("Dry run only. No files were modified.")

    if all_errors:
        print("First errors:")
        for err in all_errors[:20]:
            print(f"- {err}")
        if len(all_errors) > 20:
            print(f"... and {len(all_errors) - 20} more")

    if args.report_json:
        report = {
            "draft_items": len(draft_items),
            "mapped_entries": total_entries,
            "target_files": len(results),
            "added": total_added,
            "updated": total_updated,
            "kept": total_kept,
            "unmapped": unmapped,
            "errors": all_errors,
            "files": [
                {
                    "file": r.file.as_posix(),
                    "total": r.total,
                    "added": r.added,
                    "updated": r.updated,
                    "kept": r.kept,
                    "errors": r.errors,
                }
                for r in sorted(results, key=lambda x: x.file.as_posix())
            ],
        }
        report_path = args.report_json.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Report: {report_path}")

    if args.mark_checked and not args.dry_run and source_todo_path:
        todo_path = source_todo_path
        if not todo_path.is_absolute():
            todo_path = (translated_path.parent / todo_path).resolve()
        if todo_path.is_file():
            changed = update_todo_checkmarks(todo_path, completed_locations)
            print(f"TODO checked: {changed} ({todo_path})")
        else:
            print(f"TODO check skipped, file not found: {todo_path}")

    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
