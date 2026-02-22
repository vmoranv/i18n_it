#!/usr/bin/env python3
"""Normalize i18n translation files: terms, punctuation, and dirty values.

Supported formats:
- JSON locale files
- FTL locale files

By default only zh locale files are processed to avoid cross-language pollution.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TERM_MAP_ZH: list[tuple[str, str]] = [
    ("启动 Live Mode 失败:", "启动实时模式失败："),
    ("启动 Live Mode 失败：", "启动实时模式失败："),
    ("Live Mode", "实时模式"),
    ("直播模式", "实时模式"),
    ("新的聊天", "新对话"),
    ("夜间模式", "深色模式"),
    ("日间模式", "浅色模式"),
]

FTL_EMPTY_VALUE_FALLBACKS: dict[str, dict[str, str]] = {
    "cli-commands-cmd_plug-list_header": {
        "zh-cn": "插件列表：",
        "en-us": "Plugin list:",
        "ja-jp": "プラグイン一覧：",
        "fr-fr": "Liste des plugins :",
        "ru-ru": "Список плагинов:",
    },
    "cli-commands-cmd_plug-table_header": {
        "zh-cn": "插件信息：",
        "en-us": "Plugin info:",
        "ja-jp": "プラグイン情報：",
        "fr-fr": "Informations du plugin :",
        "ru-ru": "Информация о плагине:",
    },
    "cli-commands-cmd_plug-header_format": {
        "zh-cn": "插件：",
        "en-us": "Plugin:",
        "ja-jp": "プラグイン：",
        "fr-fr": "Plugin :",
        "ru-ru": "Плагин:",
    },
}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
FTL_ENTRY_RE = re.compile(r"^(\s*[-A-Za-z0-9_.]+)(\s*=\s*)(.*)$")
ZH_LOCALE_PATTERN = re.compile(r"^zh([_-].*)?$", re.IGNORECASE)
LOCALE_SEGMENT_PATTERN = re.compile(r"^[a-z]{2,3}(?:[_-][A-Za-z0-9]{2,8})*$")
FTL_PLACEHOLDER_RE = re.compile(r"\{\s*\$([A-Za-z_][A-Za-z0-9_-]*)\s*\}")
FTL_FULLWIDTH_BRACE_RE = re.compile(r"｛([^｝]*)｝")
FTL_NEWLINE_TOKEN = '{"\\u000A"}'
FTL_ANY_NEWLINE_TOKEN_RE = re.compile(
    r"(?:\{\s*['\"]\\u000A['\"]\s*\}|｛\s*['\"]\\u000A['\"]\s*｝)",
)
FTL_ESCAPED_NEWLINE_RE = re.compile(r"(?<!\\)\\n")
FTL_DOUBLE_FULLWIDTH_BRACE_RE = re.compile(r"｛｛\s*([^｝]+?)\s*｝｝")
FTL_BROKEN_FULLWIDTH_EXPR_RE = re.compile(
    r"｛[^｝\n]*?(\{\s*\$([A-Za-z_][A-Za-z0-9_-]*)\s*\})",
)


@dataclass
class NormalizeStats:
    files_scanned: int = 0
    files_changed: int = 0
    values_changed: int = 0
    term_replacements: int = 0
    punctuation_fixes: int = 0
    dirty_value_fixes: int = 0
    ftl_artifact_fixes: int = 0


def clean_dirty_value(value: str) -> tuple[str, int]:
    """Clean common accidental wrappers like "'text';" or "\"text\";"."""
    changed = 0
    original = value

    # "'text';" or "\"text\";" => "text"
    wrapped_with_semicolon = re.match(
        r"^\s*(['\"])(.*?)\1;\s*$", value, flags=re.DOTALL
    )
    if wrapped_with_semicolon:
        value = wrapped_with_semicolon.group(2)
        changed += 1

    # "'text'" or "\"text\"" => "text"
    wrapped_plain = re.match(r"^\s*(['\"])(.*?)\1\s*$", value, flags=re.DOTALL)
    if wrapped_plain and value == wrapped_plain.group(0):
        value = wrapped_plain.group(2)
        changed += 1

    if value != original:
        return value, changed
    return value, 0


def normalize_punctuation(value: str) -> tuple[str, int]:
    """Normalize ASCII punctuation in Chinese strings."""
    if not CJK_RE.search(value):
        return value, 0

    changes = 0
    # Chinese context colon: "错误:" => "错误：", "模型: {x}" => "模型：{x}"
    value, n = re.subn(r"([\u4e00-\u9fff])\s*:\s*", r"\1：", value)
    changes += n

    # Chinese context exclamation/question mark
    value, n = re.subn(r"([\u4e00-\u9fff])!", r"\1！", value)
    changes += n
    value, n = re.subn(r"([\u4e00-\u9fff])\?", r"\1？", value)
    changes += n

    return value, changes


def apply_term_map(value: str, term_map: list[tuple[str, str]]) -> tuple[str, int]:
    changes = 0
    for old, new in term_map:
        count = value.count(old)
        if count:
            value = value.replace(old, new)
            changes += count
    return value, changes


def normalize_term_spacing(value: str) -> tuple[str, int]:
    """Remove accidental spaces introduced when English terms are replaced by Chinese terms."""
    changes = 0
    value, n = re.subn(r"启动\s+实时模式\s+失败", "启动实时模式失败", value)
    changes += n
    return value, changes


def normalize_ftl_artifacts(value: str) -> tuple[str, int]:
    """Normalize artifacts like full-width braces and FTL newline token style."""
    changes = 0
    original = value

    # Recover malformed fragments like "｛json.dumps(..., {$var}" -> "{ $var}".
    def _replace_broken_fullwidth(match: re.Match[str]) -> str:
        var_name = match.group(2)
        return f"{{${var_name}}}"

    value, n = FTL_BROKEN_FULLWIDTH_EXPR_RE.subn(_replace_broken_fullwidth, value)
    changes += n

    # Convert full-width double braces to literal moustache braces.
    value, n = FTL_DOUBLE_FULLWIDTH_BRACE_RE.subn(r"{{\1}}", value)
    changes += n

    # Canonicalize newline tokens to Fluent-compatible {"\u000A"}.
    value, n = FTL_ANY_NEWLINE_TOKEN_RE.subn(lambda _m: FTL_NEWLINE_TOKEN, value)
    changes += n

    # Convert escaped newlines from extracted strings: "\n" => {"\u000A"}.
    value, n = FTL_ESCAPED_NEWLINE_RE.subn(lambda _m: FTL_NEWLINE_TOKEN, value)
    changes += n

    placeholder_vars = FTL_PLACEHOLDER_RE.findall(value)
    placeholder_set = set(placeholder_vars)

    def _replace_fullwidth(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not inner:
            return ""
        # Keep already-correct Fluent placeholders unchanged.
        if inner.startswith("$"):
            return "{" + inner + "}"
        # Prefer variables that already exist as placeholders in this message.
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", inner)
        for token in reversed(tokens):
            if token in placeholder_set:
                return "{ $" + token + " }".replace(" ", "")
        # If only one placeholder exists, map unknown expression to that variable.
        if len(placeholder_set) == 1:
            only = next(iter(placeholder_set))
            return "{ $" + only + " }".replace(" ", "")
        # Otherwise drop non-renderable expression text.
        return ""

    value_new, n = FTL_FULLWIDTH_BRACE_RE.subn(_replace_fullwidth, value)
    value = value_new
    changes += n

    # Canonicalize Fluent variable spacing to compact style: "{$x}".
    value_new, n = FTL_PLACEHOLDER_RE.subn(lambda m: f"{{${m.group(1)}}}", value)
    value = value_new
    changes += n

    # Deduplicate directly repeated placeholders: "{$x} {$x}" -> "{$x}".
    value_new, n = re.subn(
        r"(\{\s*\$([A-Za-z_][A-Za-z0-9_-]*)\s*\})(\s+\{\s*\$\2\s*\})+",
        r"{$\2}",
        value,
    )
    value = value_new
    changes += n

    # Remove trailing placeholder tails when all tail vars already appeared in the main text.
    tail_match = re.search(r"(?:\s+\{\s*\$[A-Za-z_][A-Za-z0-9_-]*\s*\})+\s*$", value)
    if tail_match:
        head = value[: tail_match.start()]
        tail = tail_match.group(0)
        head_vars = set(FTL_PLACEHOLDER_RE.findall(head))
        tail_vars = FTL_PLACEHOLDER_RE.findall(tail)
        if head_vars and tail_vars and all(v in head_vars for v in tail_vars):
            value = head.rstrip()
            changes += 1

    # Cleanup punctuation leftovers after placeholder cleanup.
    value_new, n = re.subn(r"([:：])\s*,", r"\1", value)
    value = value_new
    changes += n

    # Collapse excessive spaces introduced by cleanup.
    value_new, n = re.subn(r" {2,}", " ", value)
    value = value_new
    changes += n

    if value != original:
        return value.strip(), max(changes, 1)
    return value, 0


def normalize_string(
    value: str, stats: NormalizeStats, term_map: list[tuple[str, str]]
) -> str:
    original = value

    value, n = clean_dirty_value(value)
    if n:
        stats.dirty_value_fixes += n

    value, n = apply_term_map(value, term_map)
    if n:
        stats.term_replacements += n

    value, n = normalize_term_spacing(value)
    if n:
        stats.term_replacements += n

    value, n = normalize_punctuation(value)
    if n:
        stats.punctuation_fixes += n

    if value != original:
        stats.values_changed += 1
    return value


def normalize_object(
    obj: Any, stats: NormalizeStats, term_map: list[tuple[str, str]]
) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_object(v, stats, term_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_object(v, stats, term_map) for v in obj]
    if isinstance(obj, str):
        return normalize_string(obj, stats, term_map)
    return obj


def collect_i18n_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        p = Path(target).resolve()
        if p.is_file() and p.suffix.lower() in {".json", ".ftl"}:
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(x for x in p.rglob("*.json") if x.is_file()))
            files.extend(sorted(x for x in p.rglob("*.ftl") if x.is_file()))
        else:
            print(f"[skip] Not a supported file or directory: {p}")
    # De-duplicate while preserving order
    seen: set[Path] = set()
    ordered: list[Path] = []
    for f in files:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered


def is_zh_locale_file(file_path: Path) -> bool:
    """Best-effort locale detection from path parts (e.g., zh-CN, zh-cn)."""
    parts = [p.lower() for p in file_path.parts]
    for part in parts:
        if ZH_LOCALE_PATTERN.match(part):
            return True
    return False


def detect_locale(file_path: Path) -> str | None:
    """Best-effort locale code detection from path segments."""
    for part in file_path.parts:
        p = part.strip()
        if LOCALE_SEGMENT_PATTERN.match(p):
            return p.lower()
    return None


def process_json_file(
    file_path: Path, apply: bool, stats: NormalizeStats, term_map: list[tuple[str, str]]
) -> bool:
    stats.files_scanned += 1
    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="utf-8-sig")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    data = json.loads(raw)

    before_values_changed = stats.values_changed
    normalized = normalize_object(data, stats, term_map)

    if stats.values_changed == before_values_changed:
        return False

    stats.files_changed += 1
    if apply:
        serialized = json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"
        file_path.write_text(serialized, encoding="utf-8")
    return True


def _normalize_ftl_line(
    line: str,
    stats: NormalizeStats,
    term_map: list[tuple[str, str]],
    locale: str | None,
) -> str:
    # Keep comments and blank lines untouched
    if not line.strip() or line.lstrip().startswith("#"):
        return line

    m = FTL_ENTRY_RE.match(line)
    if not m:
        return line

    key, sep, value = m.groups()
    key_name = key.strip()
    if not value.strip():
        fallback = FTL_EMPTY_VALUE_FALLBACKS.get(key_name, {}).get(
            (locale or "").lower()
        )
        if fallback:
            stats.values_changed += 1
            stats.dirty_value_fixes += 1
            return f"{key}{sep}{fallback}"
        return line

    normalized_value = normalize_string(value, stats, term_map)
    value_before_artifacts = normalized_value
    normalized_value, n = normalize_ftl_artifacts(normalized_value)
    if n:
        stats.ftl_artifact_fixes += n
    if normalized_value != value_before_artifacts:
        stats.values_changed += 1
    return f"{key}{sep}{normalized_value}"


def process_ftl_file(
    file_path: Path, apply: bool, stats: NormalizeStats, term_map: list[tuple[str, str]]
) -> bool:
    stats.files_scanned += 1
    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="utf-8-sig")
    has_trailing_newline = raw.endswith("\n")
    lines = raw.splitlines()
    before_values_changed = stats.values_changed
    locale = detect_locale(file_path)
    normalized_lines = [
        _normalize_ftl_line(line, stats, term_map, locale) for line in lines
    ]

    if stats.values_changed == before_values_changed:
        return False

    stats.files_changed += 1
    if apply:
        serialized = "\n".join(normalized_lines)
        if has_trailing_newline:
            serialized += "\n"
        file_path.write_text(serialized, encoding="utf-8")
    return True


def process_file(
    file_path: Path, apply: bool, stats: NormalizeStats, term_map: list[tuple[str, str]]
) -> bool:
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return process_json_file(file_path, apply, stats, term_map)
    if suffix == ".ftl":
        return process_ftl_file(file_path, apply, stats, term_map)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize i18n files (.json/.ftl): terms, punctuation, dirty value cleanup.",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="JSON/FTL file or directory (repeatable).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to files. Without this flag, only report.",
    )
    parser.add_argument(
        "--include-non-zh",
        action="store_true",
        help="Also process non-zh locale files (off by default).",
    )
    args = parser.parse_args()

    files = collect_i18n_files(args.target)
    if not args.include_non_zh:
        files = [f for f in files if is_zh_locale_file(f)]
    if not files:
        print("No matching i18n files found.")
        return 0

    stats = NormalizeStats()
    changed_files: list[Path] = []

    for file_path in files:
        try:
            locale = detect_locale(file_path)
            term_map = (
                DEFAULT_TERM_MAP_ZH if (locale and locale.startswith("zh")) else []
            )
            changed = process_file(file_path, args.apply, stats, term_map)
            if changed:
                changed_files.append(file_path)
        except Exception as exc:  # pragma: no cover - defensive report
            print(f"[error] {file_path}: {exc}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"[{mode}] scanned={stats.files_scanned} changed_files={stats.files_changed} changed_values={stats.values_changed}"
    )
    print(
        f"[{mode}] term_replacements={stats.term_replacements} "
        f"punctuation_fixes={stats.punctuation_fixes} "
        f"dirty_value_fixes={stats.dirty_value_fixes} "
        f"ftl_artifact_fixes={stats.ftl_artifact_fixes}",
    )
    for path in changed_files:
        print(f"[changed] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
