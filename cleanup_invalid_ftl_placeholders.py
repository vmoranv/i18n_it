from __future__ import annotations

import argparse
import re
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"\{\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*\}")


def cleanup_file(path: Path, dry_run: bool) -> tuple[bool, int]:
    text = path.read_text(encoding="utf-8")
    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed
        name = match.group(1)
        if name and name[0].isalpha():
            return match.group(0)
        removed += 1
        return ""

    updated = PLACEHOLDER_RE.sub(repl, text)
    # Normalize double spaces introduced by removals.
    updated = re.sub(r"[ \t]{2,}", " ", updated)
    updated = re.sub(r" = \n", " =\n", updated)

    if removed and not dry_run:
        path.write_text(updated, encoding="utf-8")
    return (removed > 0), removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove invalid FTL placeholders whose variable name does not start with a letter."
    )
    parser.add_argument(
        "--locale-root",
        required=True,
        help="Locale root directory, e.g. path/to/i18n/locales",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    locale_root = Path(args.locale_root).resolve()
    ftl_files = sorted(locale_root.glob("*/*.ftl"))

    touched = 0
    removed_total = 0
    for path in ftl_files:
        changed, removed = cleanup_file(path, dry_run=args.dry_run)
        if changed:
            touched += 1
            removed_total += removed
            print(f"[updated] {path} (removed {removed})")

    print(
        f"Scanned {len(ftl_files)} file(s); {'would update' if args.dry_run else 'updated'} {touched} file(s); "
        f"removed placeholders: {removed_total}."
    )


if __name__ == "__main__":
    main()
