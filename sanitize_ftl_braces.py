from __future__ import annotations

import argparse
import re
from pathlib import Path

KEY_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)\s*=\s*(.*)$")
PLACEHOLDER_RE = re.compile(r"\{\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*\}")


def protect_placeholders(text: str) -> tuple[str, list[str]]:
    saved: list[str] = []

    def repl(match: re.Match[str]) -> str:
        saved.append(match.group(0))
        return f"__PH_{len(saved) - 1}__"

    return PLACEHOLDER_RE.sub(repl, text), saved


def restore_placeholders(text: str, saved: list[str]) -> str:
    for idx, placeholder in enumerate(saved):
        text = text.replace(f"__PH_{idx}__", placeholder)
    return text


def sanitize_value_text(text: str) -> str:
    protected, saved = protect_placeholders(text)
    protected = protected.replace("{", "｛").replace("}", "｝")
    return restore_placeholders(protected, saved)


def sanitize_ftl_file(path: Path, dry_run: bool) -> tuple[bool, int]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    changed = 0

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.rstrip("\r\n")
        if not stripped or stripped[0].isspace() or stripped.lstrip().startswith("#"):
            i += 1
            continue

        m = KEY_LINE_RE.match(stripped)
        if not m:
            i += 1
            continue

        key = m.group(1)
        first_rhs = m.group(2)
        nl = raw[len(stripped) :]
        new_rhs = sanitize_value_text(first_rhs)
        if new_rhs != first_rhs:
            lines[i] = f"{key} = {new_rhs}{nl}"
            changed += 1

        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            nxt_stripped = nxt.rstrip("\r\n")
            if not nxt_stripped:
                j += 1
                continue
            if not nxt_stripped[0].isspace():
                if KEY_LINE_RE.match(nxt_stripped):
                    break
            nln = nxt[len(nxt_stripped) :]
            new_line = sanitize_value_text(nxt_stripped)
            if new_line != nxt_stripped:
                lines[j] = new_line + nln
                changed += 1
            j += 1

        i = j

    if changed and not dry_run:
        path.write_text("".join(lines), encoding="utf-8")
    return (changed > 0), changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize FTL values by keeping {$param} placeholders and turning other braces into literal full-width braces."
        )
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
    changed_lines = 0
    for path in ftl_files:
        changed, count = sanitize_ftl_file(path, dry_run=args.dry_run)
        if changed:
            touched += 1
            changed_lines += count
            print(f"[updated] {path} ({count} line(s))")

    print(
        f"Scanned {len(ftl_files)} file(s); {'would update' if args.dry_run else 'updated'} {touched} file(s); "
        f"changed lines: {changed_lines}."
    )


if __name__ == "__main__":
    main()
