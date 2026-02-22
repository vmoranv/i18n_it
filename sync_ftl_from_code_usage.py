from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

KEY_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)\s*=\s*(.*)$")
PLACEHOLDER_RE = re.compile(r"\{\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*\}")
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
class Entry:
    key: str
    start: int
    end: int


class FTLFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        self.entries: dict[str, Entry] = {}
        self.modified = False
        self._parse_entries()

    def _parse_entries(self) -> None:
        self.entries.clear()
        starts: list[tuple[int, str]] = []
        for idx, line in enumerate(self.lines):
            if not line or line[0].isspace() or line.lstrip().startswith("#"):
                continue
            m = KEY_LINE_RE.match(line.rstrip("\r\n"))
            if m:
                starts.append((idx, m.group(1)))

        for i, (start, key) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(self.lines)
            self.entries[key] = Entry(key=key, start=start, end=end)

    @property
    def newline(self) -> str:
        for line in self.lines:
            if line.endswith("\r\n"):
                return "\r\n"
            if line.endswith("\n"):
                return "\n"
        return "\n"

    def save(self) -> None:
        if not self.modified:
            return
        self.path.write_text("".join(self.lines), encoding="utf-8")

    def entry_placeholders(self, key: str) -> set[str]:
        entry = self.entries[key]
        text = "".join(self.lines[entry.start : entry.end])
        return set(PLACEHOLDER_RE.findall(text))

    def entry_value_first_line(self, key: str) -> str:
        line = self.lines[self.entries[key].start].rstrip("\r\n")
        _, _, rhs = line.partition("=")
        return rhs.strip()

    def add_key_with_value(self, key: str, value: str) -> bool:
        if key in self.entries:
            return False
        nl = self.newline
        if self.lines and not self.lines[-1].endswith(("\n", "\r\n")):
            self.lines[-1] = self.lines[-1] + nl
        self.lines.append(f"{key} = {value}{nl}")
        self.modified = True
        self._parse_entries()
        return True

    def ensure_missing_placeholders(self, key: str, expected: set[str]) -> bool:
        if key not in self.entries or not expected:
            return False
        existing = self.entry_placeholders(key)
        missing = sorted(expected - existing)
        if not missing:
            return False
        entry = self.entries[key]
        first = self.lines[entry.start]
        raw = first.rstrip("\r\n")
        nl = first[len(raw) :]
        if not nl:
            nl = self.newline
        left, sep, right = raw.partition("=")
        if not sep:
            return False
        rhs = right.strip()
        suffix = " ".join("{$%s}" % p for p in missing)
        if rhs:
            new_line = f"{left.rstrip()} = {rhs} {suffix}{nl}"
        else:
            new_line = f"{left.rstrip()} = {suffix}{nl}"
        self.lines[entry.start] = new_line
        self.modified = True
        self._parse_entries()
        return True


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if is_ignored_path(path):
            continue
        files.append(path)
    return files


def parse_t_key_usage(root: Path) -> dict[str, set[str]]:
    usage: dict[str, set[str]] = defaultdict(set)
    for path in iter_python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name) or func.id != "t":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            key = first.value.strip()
            if not key:
                continue
            for kw in node.keywords:
                if kw.arg:
                    usage[key].add(kw.arg)
    return usage


def load_locale_files(locale_root: Path) -> dict[str, dict[str, FTLFile]]:
    data: dict[str, dict[str, FTLFile]] = {}
    for locale_dir in locale_root.iterdir():
        if not locale_dir.is_dir():
            continue
        files: dict[str, FTLFile] = {}
        for ftl_path in locale_dir.glob("*.ftl"):
            files[ftl_path.name] = FTLFile(ftl_path)
        if files:
            data[locale_dir.name] = files
    return data


def dot_suffix_aliases(key: str) -> list[str]:
    aliases: list[str] = []
    if key.endswith(".true"):
        aliases.append(key[:-5] + "-true")
    if key.endswith(".false"):
        aliases.append(key[:-6] + "-false")
    return aliases


def key_file_index(files: dict[str, FTLFile]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for rel_file, ftl in files.items():
        for key in ftl.entries:
            idx[key] = rel_file
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sync FTL keys/placeholders from Python t(...) usage and reference locale."
        )
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root containing Python source files.",
    )
    parser.add_argument(
        "--locale-root",
        default="i18n/locales",
        help="Locale root relative to --project-root or absolute path.",
    )
    parser.add_argument(
        "--reference-locale",
        default="zh-cn",
        help="Reference locale used for missing key fill.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    locale_root = Path(args.locale_root)
    if not locale_root.is_absolute():
        locale_root = (project_root / locale_root).resolve()

    usage = parse_t_key_usage(project_root)
    locale_data = load_locale_files(locale_root)
    if args.reference_locale not in locale_data:
        raise SystemExit(f"Reference locale not found: {args.reference_locale}")

    ref_files = locale_data[args.reference_locale]
    if "main.ftl" not in ref_files:
        raise SystemExit("Reference locale missing main.ftl")

    ref_index = key_file_index(ref_files)

    added_missing_in_ref = 0
    added_missing_in_other = 0
    added_missing_from_ref = 0
    placeholders_added = 0

    # Ensure every used key exists in reference locale.
    for key in sorted(usage):
        if key in ref_index:
            continue
        aliased_value = None
        target_file = "main.ftl"
        for alias in dot_suffix_aliases(key):
            if alias in ref_index:
                alias_file = ref_index[alias]
                aliased_value = ref_files[alias_file].entry_value_first_line(alias)
                target_file = alias_file
                break
        if aliased_value is None:
            aliased_value = key
        if ref_files[target_file].add_key_with_value(key, aliased_value):
            added_missing_in_ref += 1
            ref_index[key] = target_file

    # Ensure all non-reference locales contain all reference keys.
    ref_index = key_file_index(ref_files)
    for locale, files in locale_data.items():
        if locale == args.reference_locale:
            continue
        for rel_file, ref_ftl in ref_files.items():
            if rel_file not in files:
                continue
            dst = files[rel_file]
            for key in ref_ftl.entries:
                if key in dst.entries:
                    continue
                value = ref_ftl.entry_value_first_line(key)
                if dst.add_key_with_value(key, value):
                    added_missing_from_ref += 1

    # Ensure used-key placeholders exist in every locale.
    for key, params in usage.items():
        target_file = ref_index.get(key, "main.ftl")
        for locale, files in locale_data.items():
            if target_file not in files:
                continue
            ftl = files[target_file]
            if key not in ftl.entries:
                ref_value = ref_files[target_file].entry_value_first_line(key)
                if ftl.add_key_with_value(key, ref_value):
                    added_missing_in_other += 1
            if ftl.ensure_missing_placeholders(key, params):
                placeholders_added += 1

    touched_files = 0
    for files in locale_data.values():
        for ftl in files.values():
            if ftl.modified:
                touched_files += 1
                if not args.dry_run:
                    ftl.save()

    report = {
        "project_root": str(project_root),
        "locale_root": str(locale_root),
        "reference_locale": args.reference_locale,
        "used_keys": len(usage),
        "added_missing_in_reference": added_missing_in_ref,
        "added_missing_from_reference": added_missing_from_ref,
        "added_missing_during_placeholder_sync": added_missing_in_other,
        "placeholder_updates": placeholders_added,
        "touched_files": touched_files,
        "dry_run": bool(args.dry_run),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report_json:
        out = Path(args.report_json)
        if not out.is_absolute():
            out = (Path.cwd() / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote report: {out}")


if __name__ == "__main__":
    main()
