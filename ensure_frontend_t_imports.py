#!/usr/bin/env python3
r"""
Ensure frontend files that call t(...) have an import for t from i18n composables.

Usage:
  python ensure_frontend_t_imports.py --frontend-root ./frontend
"""

from __future__ import annotations

import argparse
import pathlib
import re


I18N_IMPORT_RE = re.compile(
    r"import\s*{\s*([^}]+)\s*}\s*from\s*(['\"])([^'\"]*i18n/composables)\2;?"
)

IMPORT_LINE_RE = re.compile(r"^import .+?;?$", re.MULTILINE)
SCRIPT_BLOCK_RE = re.compile(r"(<script\b[^>]*>)([\s\S]*?)(</script>)", re.IGNORECASE)
T_USAGE_RE = re.compile(r"\bt\s*\(")
T_DEFINED_RE = re.compile(
    r"\b(?:const|let|var|function)\s+t\b|{\s*t\s*}|\bt\s*:",
    re.MULTILINE,
)
LOCAL_T_BINDING_RE = re.compile(
    r"\b(?:const|let|var)\s*{\s*[^}]*\bt\b[^}]*}\s*=\s*use(?:Module)?I18n\s*\(",
    re.MULTILINE,
)


def _ensure_t_import(code: str, preferred_path: str) -> tuple[str, bool]:
    match = I18N_IMPORT_RE.search(code)
    if match:
        names = [n.strip() for n in match.group(1).split(",") if n.strip()]
        if "t" in names:
            return code, False
        names.append("t")
        new_import = f"import {{ {', '.join(names)} }} from {match.group(2)}{match.group(3)}{match.group(2)};"
        return code[: match.start()] + new_import + code[match.end() :], True

    import_matches = list(IMPORT_LINE_RE.finditer(code))
    new_line = f"import {{ t }} from '{preferred_path}';"
    if import_matches:
        insert_at = import_matches[-1].end()
        patched = code[:insert_at] + "\n" + new_line + code[insert_at:]
        return patched, True

    return new_line + "\n" + code, True


def _remove_t_from_i18n_import(code: str) -> tuple[str, bool]:
    match = I18N_IMPORT_RE.search(code)
    if not match:
        return code, False

    names = [n.strip() for n in match.group(1).split(",") if n.strip()]
    if "t" not in names:
        return code, False

    kept = [n for n in names if n != "t"]
    if kept:
        new_import = (
            f"import {{ {', '.join(kept)} }} from "
            f"{match.group(2)}{match.group(3)}{match.group(2)};"
        )
    else:
        new_import = ""

    patched = code[: match.start()] + new_import + code[match.end() :]
    patched = re.sub(r"\n{3,}", "\n\n", patched)
    return patched, True


def patch_file(path: pathlib.Path, frontend_root: pathlib.Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not T_USAGE_RE.search(text):
        return False

    if path.as_posix().endswith("src/i18n/composables.ts"):
        return False

    relative = path.relative_to(frontend_root).as_posix()
    preferred_path = (
        "./i18n/composables" if relative == "src/main.ts" else "@/i18n/composables"
    )

    changed = False
    patched = text

    if path.suffix == ".vue":
        script_match = SCRIPT_BLOCK_RE.search(text)
        if not script_match:
            return False
        script_open, script_body, script_close = script_match.groups()
        if LOCAL_T_BINDING_RE.search(script_body):
            new_body, changed = _remove_t_from_i18n_import(script_body)
        else:
            new_body, changed = _ensure_t_import(script_body, preferred_path)
        if changed:
            patched = (
                text[: script_match.start()]
                + script_open
                + new_body
                + script_close
                + text[script_match.end() :]
            )
    else:
        if T_DEFINED_RE.search(text) and "i18n/composables" not in text:
            return False
        patched, changed = _ensure_t_import(text, preferred_path)

    if changed:
        path.write_text(patched, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure frontend files that use t(...) import t from i18n composables."
    )
    parser.add_argument(
        "--frontend-root",
        type=pathlib.Path,
        required=True,
        help="Frontend project root path.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optional explicit file list (relative to frontend root).",
    )
    args = parser.parse_args()

    frontend_root = args.frontend_root.resolve()
    if not frontend_root.exists():
        raise SystemExit(f"Frontend root not found: {frontend_root}")

    if args.files:
        targets = [frontend_root / f for f in args.files]
    else:
        targets = [
            p for p in (frontend_root / "src").rglob("*") if p.suffix in {".ts", ".vue"}
        ]

    changed_files: list[pathlib.Path] = []
    for target in targets:
        if target.exists() and patch_file(target, frontend_root):
            changed_files.append(target)

    print(f"Updated {len(changed_files)} files.")
    for path in changed_files:
        print(path.relative_to(frontend_root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
