#!/usr/bin/env python3
"""Restore FTL newline tokens based on a git baseline commit.

This script compares current FTL files against a baseline commit and restores
`{"\\u000A"}` tokens for keys where:
1) baseline value contains the token, and
2) current value does not contain the token.

Typical usage:
  python restore_ftl_newline_tokens.py --repo-root ../target-repo --base-commit 350a11be --dry-run
  python restore_ftl_newline_tokens.py --repo-root ../target-repo --base-commit 350a11be --apply
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ENTRY_RE = re.compile(r"^(\s*[-A-Za-z0-9_.]+)(\s*=\s*)(.*)$")
PLACEHOLDER_RE = re.compile(r"\{\s*\$([A-Za-z_][A-Za-z0-9_-]*)\s*\}")
DEFAULT_TOKEN = '{"\\u000A"}'
DEFAULT_GLOB = "i18n/locales/**/*.ftl"


@dataclass
class Stats:
    scanned_files: int = 0
    compared_files: int = 0
    changed_files: int = 0
    changed_keys: int = 0
    skipped_missing_in_baseline: int = 0


def run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def assert_git_repo(repo_root: Path) -> None:
    proc = run_git(repo_root, "rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise RuntimeError(f"Not a git repository: {repo_root}")


def assert_commit_exists(repo_root: Path, commit: str) -> None:
    proc = run_git(repo_root, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "unknown git error"
        raise RuntimeError(f"Invalid commit '{commit}': {msg}")


def get_file_from_commit(repo_root: Path, commit: str, rel_path: str) -> str | None:
    proc = run_git(repo_root, "show", f"{commit}:{rel_path}")
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_ftl_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        m = ENTRY_RE.match(line)
        if not m:
            continue
        key_raw, _, value = m.groups()
        mapping[key_raw.strip()] = value
    return mapping


def normalize_placeholder_style(value: str) -> str:
    return PLACEHOLDER_RE.sub(lambda m: f"{{${m.group(1)}}}", value)


def restore_file(
    file_path: Path,
    baseline_map: dict[str, str],
    token: str,
    apply: bool,
) -> tuple[int, list[str]]:
    text = file_path.read_text(encoding="utf-8")
    has_trailing_newline = text.endswith("\n")

    changed = 0
    changed_keys: list[str] = []
    out_lines: list[str] = []

    for line in text.splitlines():
        m = ENTRY_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        key_raw, sep, current_value = m.groups()
        key = key_raw.strip()
        baseline_value = baseline_map.get(key)
        if baseline_value and token in baseline_value and token not in current_value:
            restored_value = normalize_placeholder_style(baseline_value)
            out_lines.append(f"{key_raw}{sep}{restored_value}")
            changed += 1
            changed_keys.append(key)
        else:
            out_lines.append(line)

    if changed > 0 and apply:
        updated = "\n".join(out_lines)
        if has_trailing_newline:
            updated += "\n"
        file_path.write_text(updated, encoding="utf-8")

    return changed, changed_keys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore FTL newline tokens from a baseline git commit.",
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Path to target git repository (e.g., ../target-repo).",
    )
    parser.add_argument(
        "--base-commit",
        required=True,
        help="Baseline commit to compare against (e.g., 350a11be).",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help=f"Glob under repo root for target FTL files (default: {DEFAULT_GLOB}).",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN,
        help='Newline token to restore (default: {"\\\\u000A"}).',
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to files. Without this, only report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias of report mode (same as not passing --apply).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    try:
        assert_git_repo(repo_root)
        assert_commit_exists(repo_root, args.base_commit)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2

    do_apply = args.apply and not args.dry_run
    stats = Stats()

    files = sorted(repo_root.glob(args.glob))
    if not files:
        print(f"No files matched glob: {args.glob}")
        return 0

    for file_path in files:
        if not file_path.is_file():
            continue
        stats.scanned_files += 1
        rel = file_path.relative_to(repo_root).as_posix()
        baseline_text = get_file_from_commit(repo_root, args.base_commit, rel)
        if baseline_text is None:
            stats.skipped_missing_in_baseline += 1
            continue

        stats.compared_files += 1
        baseline_map = parse_ftl_map(baseline_text)
        changed_count, changed_keys = restore_file(
            file_path=file_path,
            baseline_map=baseline_map,
            token=args.token,
            apply=do_apply,
        )

        if changed_count:
            stats.changed_files += 1
            stats.changed_keys += changed_count
            print(f"[changed] {rel} -> {changed_count}")
            for key in changed_keys:
                print(f"  - {key}")

    mode = "APPLY" if do_apply else "DRY-RUN"
    print(
        f"[{mode}] scanned={stats.scanned_files} compared={stats.compared_files} "
        f"changed_files={stats.changed_files} changed_keys={stats.changed_keys} "
        f"skipped_missing_in_baseline={stats.skipped_missing_in_baseline}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
