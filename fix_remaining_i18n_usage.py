#!/usr/bin/env python3
"""Generic regex patcher for text-based refactoring tasks.

This utility applies one or more regex replacement rules to one or more target files.
It is intentionally project-agnostic.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReplaceRule:
    name: str
    pattern: str
    replacement: str


def load_rules(path: Path) -> list[ReplaceRule]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Rules file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Rules file is not valid JSON: {path} ({exc})") from exc

    items: list[dict[str, str]]
    if isinstance(raw, dict):
        node = raw.get("rules")
        if not isinstance(node, list):
            raise SystemExit(
                "Rules JSON must be a list or an object containing `rules` list."
            )
        items = node
    elif isinstance(raw, list):
        items = raw
    else:
        raise SystemExit(
            "Rules JSON must be a list or an object containing `rules` list."
        )

    rules: list[ReplaceRule] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"Invalid rule at index {idx}: expected object.")
        pattern = item.get("pattern")
        replacement = item.get("replacement")
        if not isinstance(pattern, str) or not isinstance(replacement, str):
            raise SystemExit(
                f"Invalid rule at index {idx}: `pattern` and `replacement` must be strings."
            )
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"rule_{idx}"

        # Fail fast for invalid regex.
        try:
            re.compile(pattern)
        except re.error as exc:
            raise SystemExit(f"Invalid regex in rule `{name}`: {exc}") from exc

        rules.append(ReplaceRule(name=name, pattern=pattern, replacement=replacement))

    if not rules:
        raise SystemExit("No rules provided.")
    return rules


def apply_rules(text: str, rules: list[ReplaceRule]) -> tuple[str, list[str]]:
    updated = text
    applied: list[str] = []

    for rule in rules:
        replaced, count = re.subn(rule.pattern, rule.replacement, updated)
        if count > 0:
            updated = replaced
            applied.append(f"{rule.name}:{count}")

    return updated, applied


def process_file(
    path: Path,
    *,
    rules: list[ReplaceRule],
    dry_run: bool,
    encoding: str,
) -> tuple[bool, list[str]]:
    text = path.read_text(encoding=encoding)
    updated, applied = apply_rules(text, rules)

    if updated == text:
        return False, []

    if not dry_run:
        path.write_text(updated, encoding=encoding)
    return True, applied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply regex replacement rules to target files."
    )
    parser.add_argument(
        "--rules-json",
        type=Path,
        required=True,
        help="Path to rules JSON (list or object with `rules`).",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Target file path (relative to repo root if not absolute). Repeatable.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve relative target paths (default: current directory).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for read/write (default: utf-8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report files that would change.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    rules = load_rules(args.rules_json.resolve())

    changed = 0
    for target in args.target:
        path = Path(target)
        if not path.is_absolute():
            path = repo_root / path

        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        if not path.is_file():
            print(f"[skip] not a file {path}")
            continue

        try:
            is_changed, details = process_file(
                path,
                rules=rules,
                dry_run=args.dry_run,
                encoding=args.encoding,
            )
        except UnicodeDecodeError as exc:
            print(f"[skip] decode failed {path}: {exc}")
            continue
        except OSError as exc:
            print(f"[skip] io failed {path}: {exc}")
            continue

        if is_changed:
            changed += 1
            action = "would update" if args.dry_run else "updated"
            print(f"[{action}] {path}")
            for detail in details:
                print(f"  - {detail}")
        else:
            print(f"[ok] {path} (no changes)")

    print(f"done: changed_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
