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
import io
import keyword
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

HEADER_RE = re.compile(r"^## `(?P<path>[^`]+)`$")
LINE_CELL_RE = re.compile(r"^`(?P<line>\d+)`$")
HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
ATTR_RE = re.compile(
    r"(?P<name>[A-Za-z_][\w:-]*)=(?P<quote>['\"])(?P<value>[^'\"]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^'\"]*)(?P=quote)"
)
TAG_TEXT_RE = re.compile(
    r">(?P<text>[^<]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^<]*)<"
)
VUE_VAR_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
MUSTACHE_BLOCK_RE = re.compile(r"{{\s*(?P<expr>[^{}]+)\s*}}")
TERNARY_STRING_EXPR_RE = re.compile(
    r"^(?P<cond>.+?)\?\s*(?P<q1>['\"])(?P<true>(?:\\.|(?!\2).)*)\2\s*:\s*(?P<q2>['\"])(?P<false>(?:\\.|(?!\4).)*)\4$"
)
PY_TERNARY_SINGLE_QUOTE_RE = re.compile(
    r"^'(?P<true>(?:\\.|[^'])*)'\s+if\s+(?P<cond>.+?)\s+else\s+'(?P<false>(?:\\.|[^'])*)'$"
)
PY_TERNARY_DOUBLE_QUOTE_RE = re.compile(
    r'^"(?P<true>(?:\\.|[^"])*)"\s+if\s+(?P<cond>.+?)\s+else\s+"(?P<false>(?:\\.|[^"])*)"$'
)
STRING_LITERAL_RE = re.compile(
    r"(?P<prefix>(?:[rRuUbBfF]{1,2})?)"
    r"(?P<quote>['\"])(?P<text>[^'\"]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^'\"]*)(?P=quote)"
)
JS_TEMPLATE_LITERAL_RE = re.compile(
    r"`(?P<text>(?:[^`\\]|\\.)*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF](?:[^`\\]|\\.)*)`"
)
JS_PLACEHOLDER_RE = re.compile(r"\$\{(?P<expr>[^{}]+)\}")
JS_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FTL_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
PY_STANDALONE_STRING_RE = re.compile(
    r"^\s*(?:[rRuUbBfF]{0,3})?(?:'[^'\n]*'|\"[^\"\n]*\")\s*,?\s*(?:#.*)?$"
)
PY_STRING_LITERAL_ANY_RE = re.compile(
    r"(?P<prefix>(?:[rRuUbBfF]{0,3}))"
    r"(?P<quote>['\"])(?P<text>[^'\"\n]*)(?P=quote)"
)
FSTRING_START_TOK = getattr(tokenize, "FSTRING_START", None)
FSTRING_END_TOK = getattr(tokenize, "FSTRING_END", None)


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
    current_headers: list[str] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        header_match = HEADER_RE.match(raw_line.strip())
        if header_match:
            current_path = Path(header_match.group("path"))
            current_headers = []
            continue

        if current_path is None or not raw_line.startswith("|"):
            continue
        if raw_line.startswith("|---|"):
            continue

        cells = split_md_row(raw_line)
        if not cells:
            continue

        lowered = [unescape_md_cell(cell).strip().strip("`").lower() for cell in cells]
        if lowered and lowered[0] == "line":
            current_headers = lowered
            continue

        line_match = LINE_CELL_RE.match(cells[0])
        if not line_match:
            continue

        line_no = int(line_match.group("line"))
        headers = (
            current_headers
            if current_headers and len(current_headers) == len(cells)
            else []
        )
        if headers:
            key_idx = headers.index("key") if "key" in headers else None
            source_idx = headers.index("source") if "source" in headers else None
            translated_idx: int | None = None
            for idx, header in enumerate(headers):
                if header in {"translation", "translated"}:
                    translated_idx = idx
                    break
            if translated_idx is None:
                for idx, header in enumerate(headers):
                    if header.startswith("translated_"):
                        translated_idx = idx
                        break
        else:
            key_idx = 1 if len(cells) >= 4 else None
            source_idx = 2 if len(cells) >= 4 else (1 if len(cells) >= 3 else None)
            translated_idx = 3 if len(cells) >= 4 else (2 if len(cells) >= 3 else None)

        if source_idx is None or translated_idx is None:
            continue
        if source_idx >= len(cells) or translated_idx >= len(cells):
            continue

        key = (
            normalize_key_cell(cells[key_idx])
            if key_idx is not None and key_idx < len(cells)
            else None
        )
        source = unescape_md_cell(cells[source_idx])
        translated = unescape_md_cell(cells[translated_idx])
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


def is_markup_path(file_path: Path) -> bool:
    return file_path.suffix.lower() in {
        ".vue",
        ".html",
        ".xml",
        ".svg",
        ".svelte",
        ".astro",
        ".mdx",
    }


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


def normalize_template_key(
    key: str | None,
    *,
    file_path: Path,
    template_func: str,
    key_style: str,
) -> str | None:
    if not key:
        return None
    if key_style == "raw":
        return key
    if key_style == "ftl":
        return to_ftl_key(key)

    # auto mode: backend Python t(...) keys are Fluent IDs.
    fn_name = template_func.strip().split(".")[-1]
    if file_path.suffix.lower() == ".py" and fn_name in {"t", "_t"}:
        return to_ftl_key(key)
    return key


def is_python_standalone_string_line(line: str) -> bool:
    return bool(PY_STANDALONE_STRING_RE.match(line.rstrip("\r\n")))


def append_plus_before_comment(line: str) -> str:
    comment_idx = line.find("#")
    if comment_idx == -1:
        return f"{line.rstrip()} +"
    prefix = line[:comment_idx].rstrip()
    comment = line[comment_idx:]
    return f"{prefix} + {comment}"


def choose_template_func_for_file(file_text: str, template_func: str) -> str:
    if template_func != "tm":
        return template_func
    # Prefer tm only when it is actually bound/used in the file.
    if (
        re.search(r"\bconst\s*\{[^}]*\btm\b", file_text)
        or re.search(r"\b(?:let|const|var)\s+tm\s*=", file_text)
        or re.search(r"\bfunction\s+tm\b", file_text)
        or re.search(r"\btm\s*\(", file_text)
    ):
        return "tm"
    # Fallback to t when tm is not present but t is available.
    if (
        re.search(r"\bconst\s*\{[^}]*\bt\b", file_text)
        or re.search(r"\b(?:let|const|var)\s+t\s*=", file_text)
        or re.search(r"\bfunction\s+t\b", file_text)
        or re.search(r"\bt\s*\(", file_text)
    ):
        return "t"
    return template_func


def extract_mustache_params(expr: str) -> tuple[str | None, str]:
    cleaned = expr.strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\b", cleaned)
    if not m:
        return None, cleaned
    return m.group(1), cleaned


def parse_ternary_string_expr(expr: str) -> tuple[str, str, str] | None:
    cleaned = expr.strip()
    match = TERNARY_STRING_EXPR_RE.match(cleaned)
    if not match:
        return None
    cond = match.group("cond").strip()
    if not cond:
        return None
    true_text = match.group("true")
    false_text = match.group("false")
    return cond, true_text, false_text


def parse_python_ternary_string_expr(expr: str) -> tuple[str, str, str] | None:
    cleaned = expr.strip()
    for pattern, quote in (
        (PY_TERNARY_SINGLE_QUOTE_RE, "'"),
        (PY_TERNARY_DOUBLE_QUOTE_RE, '"'),
    ):
        match = pattern.match(cleaned)
        if not match:
            continue
        cond = match.group("cond").strip()
        if not cond:
            return None
        true_text = (
            match.group("true").replace(f"\\{quote}", quote).replace("\\\\", "\\")
        )
        false_text = (
            match.group("false").replace(f"\\{quote}", quote).replace("\\\\", "\\")
        )
        return cond, true_text.strip(), false_text.strip()
    return None


def build_markup_v_html_line(old_noeol: str, key: str, template_func: str) -> str:
    leading = old_noeol[: len(old_noeol) - len(old_noeol.lstrip())]
    vars_found: list[str] = []
    for match in VUE_VAR_RE.finditer(old_noeol):
        name = match.group(1)
        if name not in vars_found:
            vars_found.append(name)

    if vars_found:
        params = ", ".join(f"{name}: {name}" for name in vars_found)
        call = f"{template_func}('{key}', {{ {params} }})"
    else:
        call = f"{template_func}('{key}')"
    return f'{leading}<span v-html="{call}"></span>'


def build_markup_mustache_text_line(
    old_noeol: str, key: str, template_func: str
) -> str | None:
    stripped = old_noeol.strip()
    if not stripped or "<" in stripped or ">" in stripped:
        return None

    blocks = list(MUSTACHE_BLOCK_RE.finditer(stripped))
    if not blocks:
        return None

    # Ensure Han text exists outside mustache expressions.
    outside: list[str] = []
    cursor = 0
    for blk in blocks:
        outside.append(stripped[cursor : blk.start()])
        cursor = blk.end()
    outside.append(stripped[cursor:])
    outside_text = "".join(outside)
    if (
        len(blocks) == 1
        and stripped.startswith("{{")
        and stripped.endswith("}}")
        and not HAN_RE.search(outside_text)
    ):
        ternary = parse_ternary_string_expr(blocks[0].group("expr"))
        if ternary:
            cond, true_text, false_text = ternary
            if HAN_RE.search(true_text) or HAN_RE.search(false_text):
                leading = old_noeol[: len(old_noeol) - len(old_noeol.lstrip())]
                return (
                    f"{leading}{{{{ {cond} ? {template_func}('{key}.true') : "
                    f"{template_func}('{key}.false') }}}}"
                )

    if not HAN_RE.search(outside_text):
        return None

    params: list[tuple[str, str]] = []
    seen: set[str] = set()
    for blk in blocks:
        name, expr = extract_mustache_params(blk.group("expr"))
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        params.append((name, expr))

    if params:
        param_obj = ", ".join(f"{name}: {expr}" for name, expr in params)
        call = f"{template_func}('{key}', {{ {param_obj} }})"
    else:
        call = f"{template_func}('{key}')"

    leading = old_noeol[: len(old_noeol) - len(old_noeol.lstrip())]
    return f"{leading}{{{{ {call} }}}}"


def strip_fstring_expr(expr: str) -> str | None:
    cleaned = expr.strip()
    if not cleaned:
        return None

    # Strip top-level conversion/format suffixes, e.g.:
    #   e!s -> e
    #   value:.2f -> value
    #   obj.attr!r -> obj.attr
    level = 0
    in_quote: str | None = None
    escaped = False
    cut = len(cleaned)
    for i, ch in enumerate(cleaned):
        if in_quote:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == in_quote:
                in_quote = None
            continue
        if ch in {"'", '"'}:
            in_quote = ch
            continue
        if ch in "([{":
            level += 1
            continue
        if ch in ")]}":
            level = max(level - 1, 0)
            continue
        if level == 0 and ch in {"!", ":"}:
            cut = i
            break

    expr_core = cleaned[:cut].strip()
    return expr_core or None


def derive_py_placeholder_name(expr_core: str, used_names: set[str]) -> str:
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_core)
    blocked = {
        "True",
        "False",
        "None",
        "len",
        "str",
        "repr",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "set",
        "tuple",
        "type",
        "min",
        "max",
        "sum",
        "any",
        "all",
        "sorted",
        "map",
        "filter",
        "zip",
        "enumerate",
        "range",
        "print",
        "format",
    }

    base: str | None = None
    for candidate in reversed(identifiers):
        if keyword.iskeyword(candidate):
            continue
        if candidate in blocked:
            continue
        base = candidate
        break
    if not base:
        base = "value"

    name = base
    idx = 2
    while name in used_names or keyword.iskeyword(name):
        name = f"{base}_{idx}"
        idx += 1
    return name


def extract_fstring_arguments_from_expressions(
    expressions: list[str],
) -> list[tuple[str, str]] | None:
    ordered_args: list[tuple[str, str]] = []
    seen_exprs: set[str] = set()
    used_names: set[str] = set()

    for expr in expressions:
        expr_core = strip_fstring_expr(expr)
        if not expr_core:
            return None

        # Pure literals do not need runtime parameters.
        if re.fullmatch(r"""(?s)(['"]).*?\1""", expr_core):
            continue
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", expr_core):
            continue

        if expr_core in seen_exprs:
            continue
        seen_exprs.add(expr_core)

        arg_name = derive_py_placeholder_name(expr_core, used_names)
        used_names.add(arg_name)
        ordered_args.append((arg_name, expr_core))

    return ordered_args


def extract_fstring_arguments(body: str) -> list[tuple[str, str]] | None:
    expressions = re.findall(r"\{([^{}]+)\}", body)
    return extract_fstring_arguments_from_expressions(expressions)


def build_python_template_line(
    old_noeol: str,
    key: str | None,
    template_func: str,
    *,
    allow_non_han: bool = False,
) -> str | None:
    if not key:
        return None
    if not HAN_RE.search(old_noeol) and not allow_non_han:
        return old_noeol
    # Keep Python replacement strict to avoid breaking syntax/semantics:
    # - one Han-containing STRING token only
    # - no f-string / raw-string token
    # - no interpolation braces in token text
    # - skip common regex pattern lines
    if "re.compile(" in old_noeol:
        return None

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(old_noeol).readline))
    except tokenize.TokenError:
        # Fallback for partial statement lines (e.g. dict entries ending with "{")
        # that cannot be tokenized as standalone code snippets.
        str_matches = list(STRING_LITERAL_RE.finditer(old_noeol))
        han_str_matches = [m for m in str_matches if HAN_RE.search(m.group("text"))]
        if not han_str_matches:
            return None
        if len(han_str_matches) == 1:
            str_match = han_str_matches[0]
            prefix = (str_match.group("prefix") or "").lower()
            if "f" in prefix:
                return None
            replacement = f"{template_func}('{key}')"
            return (
                f"{old_noeol[: str_match.start()]}"
                f"{replacement}"
                f"{old_noeol[str_match.end() :]}"
            )

        out = old_noeol
        indexed_matches = list(enumerate(han_str_matches, start=1))
        for idx, str_match in reversed(indexed_matches):
            prefix = (str_match.group("prefix") or "").lower()
            if "f" in prefix:
                return None
            replacement = f"{template_func}('{key}.part{idx}')"
            out = f"{out[: str_match.start()]}{replacement}{out[str_match.end() :]}"
        return out

    # Python 3.12+/3.13: f-strings are tokenized as FSTRING_START/MIDDLE/END.
    if FSTRING_START_TOK is not None and FSTRING_END_TOK is not None:
        indexed = list(enumerate(tokens))
        f_starts = [x for x in indexed if x[1].type == FSTRING_START_TOK]
        f_ends = [x for x in indexed if x[1].type == FSTRING_END_TOK]
        if len(f_starts) > 1 or len(f_ends) > 1:
            return None
        if len(f_starts) == 1 and len(f_ends) == 1:
            start_idx, f_start = f_starts[0]
            end_idx, f_end = f_ends[0]
            if start_idx < end_idx and f_start.start[0] == f_end.end[0]:
                prefix = (f_start.string or "").lower()
                if "f" in prefix and "r" not in prefix:
                    exprs: list[str] = []
                    brace_depth = 0
                    expr_start_col: int | None = None
                    for tok in tokens[start_idx + 1 : end_idx]:
                        if tok.type == tokenize.OP and tok.string == "{":
                            if brace_depth == 0:
                                expr_start_col = tok.end[1]
                            brace_depth += 1
                            continue
                        if tok.type == tokenize.OP and tok.string == "}":
                            if brace_depth <= 0:
                                return None
                            brace_depth -= 1
                            if brace_depth == 0 and expr_start_col is not None:
                                exprs.append(old_noeol[expr_start_col : tok.start[1]])
                                expr_start_col = None
                            continue
                    if brace_depth != 0:
                        return None

                    middle_text = old_noeol[f_start.end[1] : f_end.start[1]]
                    if (
                        allow_non_han
                        or HAN_RE.search(middle_text)
                        or any(HAN_RE.search(expr) for expr in exprs)
                    ):
                        ternary_exprs: list[tuple[str, tuple[str, str, str]]] = []
                        for expr in exprs:
                            parsed = parse_python_ternary_string_expr(expr)
                            if not parsed:
                                continue
                            if HAN_RE.search(parsed[1]) or HAN_RE.search(parsed[2]):
                                ternary_exprs.append((expr, parsed))

                        if len(ternary_exprs) > 1:
                            return None
                        if len(ternary_exprs) == 1:
                            ternary_expr, (cond, _true_text, _false_text) = (
                                ternary_exprs[0]
                            )
                            non_ternary_exprs = [e for e in exprs if e != ternary_expr]
                            ordered_args = extract_fstring_arguments_from_expressions(
                                non_ternary_exprs
                            )
                            if ordered_args is None:
                                return None
                            if ordered_args:
                                args = ", ".join(
                                    f"{name}={expr}" for name, expr in ordered_args
                                )
                                true_call = f"{template_func}('{key}.true', {args})"
                                false_call = f"{template_func}('{key}.false', {args})"
                            else:
                                true_call = f"{template_func}('{key}.true')"
                                false_call = f"{template_func}('{key}.false')"
                            replacement = f"{true_call} if {cond} else {false_call}"
                        else:
                            ordered_args = extract_fstring_arguments_from_expressions(
                                exprs
                            )
                            if ordered_args is None:
                                return None
                            if ordered_args:
                                args = ", ".join(
                                    f"{name}={expr}" for name, expr in ordered_args
                                )
                                replacement = f"{template_func}('{key}', {args})"
                            else:
                                replacement = f"{template_func}('{key}')"
                        return (
                            f"{old_noeol[: f_start.start[1]]}"
                            f"{replacement}"
                            f"{old_noeol[f_end.end[1] :]}"
                        )

    matches: list[tokenize.TokenInfo] = []
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        if not allow_non_han and not HAN_RE.search(tok.string):
            continue
        matches.append(tok)

    if not matches:
        return None

    if len(matches) > 1:
        out = old_noeol
        indexed_matches = list(enumerate(matches, start=1))
        for idx, tok in reversed(indexed_matches):
            if tok.start[0] != tok.end[0]:
                return None
            m = re.match(r"(?i)^([rubf]{1,3})", tok.string)
            prefix = (m.group(1) if m else "").lower()
            if "f" in prefix:
                return None
            if "${" in tok.string:
                return None
            start_col, end_col = tok.start[1], tok.end[1]
            replacement = f"{template_func}('{key}.part{idx}')"
            out = f"{out[:start_col]}{replacement}{out[end_col:]}"
        return out

    tok = matches[0]
    if tok.start[0] != tok.end[0]:
        return None

    # Prefix parsing: r/u/b/f combos before quote.
    m = re.match(r"(?i)^([rubf]{1,3})", tok.string)
    prefix = (m.group(1) if m else "").lower()
    if "r" in prefix and "f" in prefix:
        return None
    if "f" in prefix:
        expressions = re.findall(r"\{([^{}]+)\}", tok.string)
        ternary_exprs: list[tuple[str, tuple[str, str, str]]] = []
        for expr in expressions:
            parsed = parse_python_ternary_string_expr(expr)
            if not parsed:
                continue
            if HAN_RE.search(parsed[1]) or HAN_RE.search(parsed[2]):
                ternary_exprs.append((expr, parsed))
        if len(ternary_exprs) > 1:
            return None

        if len(ternary_exprs) == 1:
            ternary_expr, (cond, _true_text, _false_text) = ternary_exprs[0]
            non_ternary_exprs = [e for e in expressions if e != ternary_expr]
            ordered_args = extract_fstring_arguments_from_expressions(non_ternary_exprs)
            if ordered_args is None:
                return None
            if ordered_args:
                args = ", ".join(f"{name}={expr}" for name, expr in ordered_args)
                replacement = (
                    f"{template_func}('{key}.true', {args}) if {cond} else "
                    f"{template_func}('{key}.false', {args})"
                )
            else:
                replacement = (
                    f"{template_func}('{key}.true') if {cond} else "
                    f"{template_func}('{key}.false')"
                )
        else:
            ordered_args = extract_fstring_arguments(tok.string)
            if ordered_args is None:
                return None
            if ordered_args:
                args = ", ".join(f"{name}={expr}" for name, expr in ordered_args)
                replacement = f"{template_func}('{key}', {args})"
            else:
                replacement = f"{template_func}('{key}')"
        scrubbed = re.sub(r"\{[^{}]+\}", "", tok.string)
        if "{" in scrubbed or "}" in scrubbed or "${" in tok.string:
            return None
        start_col, end_col = tok.start[1], tok.end[1]
        return f"{old_noeol[:start_col]}{replacement}{old_noeol[end_col:]}"
    if "${" in tok.string:
        return None

    tok_idx = next(
        (
            i
            for i, tinfo in enumerate(tokens)
            if tinfo.start == tok.start
            and tinfo.end == tok.end
            and tinfo.string == tok.string
            and tinfo.type == tok.type
        ),
        -1,
    )
    start_col, end_col = tok.start[1], tok.end[1]

    replacement = f"{template_func}('{key}')"
    prev_idx = -1
    next_idx = len(tokens)
    if tok_idx != -1:
        ignore_types = {
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.ENDMARKER,
            tokenize.COMMENT,
        }
        prev_idx = tok_idx - 1
        while prev_idx >= 0 and tokens[prev_idx].type in ignore_types:
            prev_idx -= 1
        prev_prev_idx = prev_idx - 1
        while prev_prev_idx >= 0 and tokens[prev_prev_idx].type in ignore_types:
            prev_prev_idx -= 1
        next_idx = tok_idx + 1
        while next_idx < len(tokens) and tokens[next_idx].type in ignore_types:
            next_idx += 1

        template_name = template_func.strip().split(".")[-1]
        if (
            prev_idx >= 0
            and prev_prev_idx >= 0
            and next_idx < len(tokens)
            and tokens[prev_idx].type == tokenize.OP
            and tokens[prev_idx].string == "("
            and tokens[prev_prev_idx].type == tokenize.NAME
            and tokens[prev_prev_idx].string == template_name
            and tokens[next_idx].type == tokenize.OP
            and tokens[next_idx].string == ")"
        ):
            # Already inside template_func(...): replace string literal with key literal.
            replacement = f"'{key}'"

    # Python allows implicit concatenation for adjacent string literals.
    # If one side stops being a string literal after replacement, inject '+'
    # to keep the expression valid.
    prev_is_string = (
        0 <= prev_idx < len(tokens) and tokens[prev_idx].type == tokenize.STRING
    )
    next_is_string = (
        0 <= next_idx < len(tokens) and tokens[next_idx].type == tokenize.STRING
    )
    if replacement.startswith(template_func + "("):
        if prev_is_string and next_is_string:
            replacement = f" + {replacement} + "
        elif prev_is_string:
            replacement = f" + {replacement}"
        elif next_is_string:
            replacement = f"{replacement} + "

    return f"{old_noeol[:start_col]}{replacement}{old_noeol[end_col:]}"


def simplify_js_placeholder_expr(expr: str) -> tuple[str | None, str | None]:
    cleaned = expr.strip()
    if not cleaned:
        return None, None

    for sep in ("||", "??"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
            break
    if "?" in cleaned and ":" in cleaned:
        cleaned = cleaned.split("?", 1)[0].strip()

    cleaned = cleaned.replace("?.", ".")
    identifiers = JS_IDENTIFIER_RE.findall(cleaned)
    if not identifiers:
        return None, None
    return identifiers[-1], cleaned


def extract_js_template_params(body: str) -> list[tuple[str, str]] | None:
    params: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in JS_PLACEHOLDER_RE.finditer(body):
        name, expr = simplify_js_placeholder_expr(match.group("expr"))
        if not name or not expr:
            return None
        if name in seen:
            continue
        seen.add(name)
        params.append((name, expr))
    return params


def build_js_template_line(
    old_noeol: str,
    key: str | None,
    template_func: str,
) -> str | None:
    if not key:
        return None
    js_matches = list(JS_TEMPLATE_LITERAL_RE.finditer(old_noeol))
    if len(js_matches) != 1:
        return None

    js_match = js_matches[0]
    body = js_match.group("text")
    params = extract_js_template_params(body)
    if params is None:
        return None

    if params:
        param_obj = ", ".join(f"{name}: {expr}" for name, expr in params)
        replacement = f"{template_func}('{key}', {{ {param_obj} }})"
    else:
        replacement = f"{template_func}('{key}')"

    return f"{old_noeol[: js_match.start()]}{replacement}{old_noeol[js_match.end() :]}"


def build_template_line(
    old_noeol: str,
    key: str | None,
    template_func: str,
    *,
    file_path: Path,
    key_style: str,
    markup_mode: bool,
    python_mode: bool,
    allow_non_han: bool = False,
) -> str | None:
    normalized_key = normalize_template_key(
        key,
        file_path=file_path,
        template_func=template_func,
        key_style=key_style,
    )
    if not normalized_key:
        return None
    if not HAN_RE.search(old_noeol) and not allow_non_han:
        return old_noeol

    if python_mode:
        return build_python_template_line(
            old_noeol,
            normalized_key,
            template_func,
            allow_non_han=allow_non_han,
        )

    # Markup-specific handling first: attribute/text interpolation.
    if markup_mode:
        mustache_text_line = build_markup_mustache_text_line(
            old_noeol, normalized_key, template_func
        )
        if mustache_text_line is not None:
            return mustache_text_line

        stripped = old_noeol.strip()
        if (
            HAN_RE.search(stripped)
            and "<" not in stripped
            and ">" not in stripped
            and "'" not in stripped
            and '"' not in stripped
            and not stripped.startswith("//")
            and not stripped.startswith("/*")
            and not stripped.startswith("*")
        ):
            leading = old_noeol[: len(old_noeol) - len(old_noeol.lstrip())]
            return f"{leading}{{{{ {template_func}('{normalized_key}') }}}}"

        attr_matches = list(ATTR_RE.finditer(old_noeol))
        if len(attr_matches) == 1:
            attr_match = attr_matches[0]
            name = attr_match.group("name")
            bound_name = (
                name
                if (name.startswith(":") or name.startswith("v-bind:"))
                else f":{name}"
            )
            replacement = f"{bound_name}=\"{template_func}('{normalized_key}')\""
            return (
                f"{old_noeol[: attr_match.start()]}"
                f"{replacement}"
                f"{old_noeol[attr_match.end() :]}"
            )
        if len(attr_matches) > 1:
            return None

        tag_text_matches = list(TAG_TEXT_RE.finditer(old_noeol))
        if len(tag_text_matches) == 1:
            tag_text_match = tag_text_matches[0]
            tag_text = tag_text_match.group("text")
            blocks = list(MUSTACHE_BLOCK_RE.finditer(tag_text))
            if blocks and HAN_RE.search(tag_text):
                params: list[tuple[str, str]] = []
                seen: set[str] = set()
                for blk in blocks:
                    name, expr = extract_mustache_params(blk.group("expr"))
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    params.append((name, expr))

                if params:
                    param_obj = ", ".join(f"{name}: {expr}" for name, expr in params)
                    call = f"{template_func}('{normalized_key}', {{ {param_obj} }})"
                else:
                    call = f"{template_func}('{normalized_key}')"
                replacement = f">{{{{ {call} }}}}<"
                return (
                    f"{old_noeol[: tag_text_match.start()]}"
                    f"{replacement}"
                    f"{old_noeol[tag_text_match.end() :]}"
                )

            remaining = (
                f"{old_noeol[: tag_text_match.start()]}"
                f"{old_noeol[tag_text_match.end() :]}"
            )
            if HAN_RE.search(remaining):
                return build_markup_v_html_line(
                    old_noeol, normalized_key, template_func
                )
            replacement = f">{{{{ {template_func}('{normalized_key}') }}}}<"
            return (
                f"{old_noeol[: tag_text_match.start()]}"
                f"{replacement}"
                f"{old_noeol[tag_text_match.end() :]}"
            )
        if len(tag_text_matches) > 1:
            return build_markup_v_html_line(old_noeol, normalized_key, template_func)

    # Generic quoted literal replacement for code/template expressions.
    # Keep it conservative to avoid breaking syntax:
    # - only one Han-containing literal in this line
    # - no f-string / template interpolation markers
    js_template_line = build_js_template_line(old_noeol, normalized_key, template_func)
    if js_template_line is not None:
        return js_template_line

    str_matches = list(STRING_LITERAL_RE.finditer(old_noeol))
    han_str_matches = [m for m in str_matches if HAN_RE.search(m.group("text"))]
    if len(han_str_matches) == 1:
        str_match = han_str_matches[0]
        prefix = (str_match.group("prefix") or "").lower()
        body = str_match.group("text")
        if "f" in prefix:
            return None
        if "{" in body or "}" in body or "${" in body:
            return None
        replacement = f"{template_func}('{normalized_key}')"
        return f"{old_noeol[: str_match.start()]}{replacement}{old_noeol[str_match.end() :]}"
    if len(han_str_matches) > 1:
        unique_values = {m.group("text") for m in han_str_matches}
        if len(unique_values) == 1:
            for m in han_str_matches:
                prefix = (m.group("prefix") or "").lower()
                body = m.group("text")
                if "f" in prefix:
                    return None
                if "{" in body or "}" in body or "${" in body:
                    return None

            replacement = f"{template_func}('{normalized_key}')"
            out = old_noeol
            for m in reversed(han_str_matches):
                out = f"{out[: m.start()]}{replacement}{out[m.end() :]}"
            return out
        return None

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
    key_style: str,
) -> ApplyResult:
    full_path = (project_root / file_path).resolve()
    if not full_path.is_file():
        return ApplyResult(
            path=file_path, changed=0, skipped=len(file_edits), missing=True
        )

    lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )
    effective_template_func = choose_template_func_for_file(
        "".join(lines), template_func
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
        old_stripped = old_noeol.strip()
        matches_source = old_stripped == edit.source
        matches_translated = old_stripped == edit.translated
        if check_source and not (matches_source or matches_translated):
            skipped += 1
            continue
        if column == "template":
            allow_non_han = not HAN_RE.search(old_noeol) and (
                matches_translated or not check_source
            )
            template_line = build_template_line(
                old_noeol,
                edit.key,
                effective_template_func,
                file_path=file_path,
                key_style=key_style,
                markup_mode=is_markup_path(file_path),
                python_mode=file_path.suffix.lower() == ".py",
                allow_non_han=allow_non_han,
            )
            if template_line is None:
                skipped += 1
                continue
            new_noeol = template_line
            if (
                file_path.suffix.lower() == ".py"
                and idx + 1 < len(lines)
                and not old_noeol.rstrip().endswith(",")
                and is_python_standalone_string_line(old_noeol)
                and is_python_standalone_string_line(lines[idx + 1].rstrip("\r\n"))
                and not new_noeol.rstrip().endswith("+")
            ):
                new_noeol = append_plus_before_comment(new_noeol)
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
        "--key-style",
        choices=["auto", "raw", "ftl"],
        default="auto",
        help="Key normalization in template mode. auto maps Python t(...) keys to FTL format.",
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
                key_style=args.key_style,
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
