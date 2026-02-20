#!/usr/bin/env python3
"""Translate i18n TODO candidates via OpenAI-compatible chat completions API.

Examples:
    python translate_i18n_todo.py \
      --project-root . \
      --env-file .env \
      --model grok-2-latest \
      --target-locale en-US
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
TODO_LINE_RE = re.compile(
    r"^- \[(?P<mark>[ xX])\] `(?P<location>[^`]+)`(?:\s+\[key:`(?P<key>[^`]+)`\])?\s*(?P<snippet>.*)$"
)
I18N_CALL_KEY_RE = re.compile(
    r"(?:\b(?:\$?t|i18n\.t)\s*\(\s*['\"](?P<k1>[A-Za-z0-9_.-]+)['\"]\s*\)|"
    r"['\"](?P<k2>[A-Za-z0-9_.-]+)['\"]\s*\|\|\s*['\"][^'\"]*[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][^'\"]*['\"])"
)
LINE_KEY_SUFFIX_RE = re.compile(r"\.line_\d+(?:_dup\d+)?$")
HARDCODED_PREFIX = "hardcoded."
T = TypeVar("T")

# Keep common placeholders unchanged between source and translation.
PLACEHOLDER_RE = re.compile(
    r"(\{[A-Za-z_][A-Za-z0-9_]*(?::[^{}]+)?\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}|%\(\w+\)[#0\- +]?\d*(?:\.\d+)?[a-zA-Z]|%[#0\- +]?\d*(?:\.\d+)?[a-zA-Z])"
)
ENV_ASSIGN_RE = re.compile(
    r"^(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)


@dataclass(frozen=True)
class TodoItem:
    id: int
    checked: bool
    location: str
    key: str
    rel_path: Path
    line_no: int
    todo_snippet: str
    source_text: str


@dataclass(frozen=True)
class TranslationResult:
    id: int
    location: str
    key: str
    rel_path: Path
    line_no: int
    source_text: str
    translated_text: str


@dataclass(frozen=True)
class ParsedTranslation:
    translated_text: str
    key: str | None = None


class OpenAICompatClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout_sec: int = 120,
        retries: int = 3,
    ) -> None:
        self.api_base = normalize_api_base(api_base)
        self.api_keys = parse_api_keys(api_key)
        if not self.api_keys:
            raise ValueError("At least one API key is required")
        self.model = model
        self.timeout_sec = timeout_sec
        self.retries = retries

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        key_offset: int = 0,
    ) -> dict:
        url = f"{self.api_base}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        max_attempts = (self.retries + 1) * len(self.api_keys)
        for attempt in range(max_attempts):
            api_key = self.api_keys[(key_offset + attempt) % len(self.api_keys)]
            request = urllib.request.Request(
                url=url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            try:
                with urllib.request.urlopen(  # noqa: S310 - user-provided endpoint by design
                    request,
                    timeout=self.timeout_sec,
                ) as response:
                    data = response.read().decode("utf-8")
                return json.loads(data)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    break

                # Switch key first; only backoff for transient errors.
                transient_http_codes = {408, 425, 429, 500, 502, 503, 504}
                should_sleep = True
                if isinstance(exc, urllib.error.HTTPError):
                    if exc.code in {401, 403}:
                        should_sleep = False
                    elif exc.code not in transient_http_codes:
                        should_sleep = False

                if not should_sleep:
                    continue
                sleep_sec = min(8, 2 ** min(attempt, self.retries))
                time.sleep(sleep_sec)

        if last_error is None:
            raise RuntimeError("Chat request failed without explicit exception")
        raise RuntimeError(f"Chat request failed: {last_error}") from last_error


def has_han(text: str) -> bool:
    return bool(HAN_RE.search(text))


def parse_api_keys(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [key.strip() for key in raw.split(",") if key.strip()]


def parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = (
                value.replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        else:
            value = value.replace("\\'", "'").replace("\\\\", "\\")
        return value

    # Drop inline comments for unquoted values: KEY=abc # comment
    for idx, ch in enumerate(value):
        if ch == "#" and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value


def load_env_file(env_file: Path) -> int:
    if not env_file.is_file():
        return 0

    loaded = 0
    for raw_line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = ENV_ASSIGN_RE.match(line)
        if not match:
            continue

        key = match.group("key")
        value = parse_env_value(match.group("value"))
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def normalize_api_base(raw: str) -> str:
    api_base = raw.strip().rstrip("/")
    suffix = "/v1/chat/completions"
    if api_base.endswith(suffix):
        return api_base[: -len(suffix)]
    return api_base


def parse_location(location: str) -> tuple[Path, int] | None:
    if ":" not in location:
        return None
    path_part, _, line_part = location.rpartition(":")
    if not path_part:
        return None
    try:
        line_no = int(line_part)
    except ValueError:
        return None
    if line_no <= 0:
        return None
    return Path(path_part), line_no


def default_todo_path(project_root: Path) -> Path:
    root_name = project_root.name or "project"
    return project_root / f"i18n.todo.scan.{root_name}.md"


def default_output_path(todo_path: Path) -> Path:
    suffix = todo_path.suffix or ".md"
    return todo_path.with_name(f"{todo_path.stem}.translated{suffix}")


def sanitize_key_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token.lower() or "item"


def normalize_key_prefix(raw: str) -> str:
    base = raw.strip().strip(".")
    if not base:
        return "hardcoded"
    parts = [sanitize_key_token(part) for part in base.split(".") if part.strip()]
    return ".".join(parts) if parts else "hardcoded"


def build_default_key(rel_path: Path, line_no: int, key_prefix: str) -> str:
    no_suffix = rel_path.with_suffix("")
    parts = [sanitize_key_token(part) for part in no_suffix.parts if part]
    return ".".join([key_prefix, *parts, f"line_{line_no}"])


def extract_existing_i18n_key(text: str) -> str | None:
    match = I18N_CALL_KEY_RE.search(text)
    if not match:
        return None
    key = (match.group("k1") or match.group("k2") or "").strip()
    return key or None


def key_prefix_from_key(key: str) -> str:
    match = LINE_KEY_SUFFIX_RE.search(key)
    if match:
        prefix = key[: match.start()]
    elif "." in key:
        prefix = key.rpartition(".")[0]
    else:
        prefix = key
    if prefix.startswith(HARDCODED_PREFIX):
        prefix = prefix[len(HARDCODED_PREFIX) :]
    return prefix


def should_regenerate_key(key: str) -> bool:
    return bool(LINE_KEY_SUFFIX_RE.search(key) or key.startswith("hardcoded."))


def sanitize_key_candidate(candidate: str) -> str:
    parts = [sanitize_key_token(part) for part in candidate.split(".") if part.strip()]
    return ".".join(parts)


def normalize_generated_key(candidate: str, fallback_key: str) -> str:
    candidate = sanitize_key_candidate(candidate)
    if not candidate:
        return fallback_key

    if candidate.startswith(HARDCODED_PREFIX):
        candidate = candidate[len(HARDCODED_PREFIX) :]

    prefix = sanitize_key_candidate(key_prefix_from_key(fallback_key))
    if not prefix:
        return fallback_key

    if candidate == prefix:
        return fallback_key

    if candidate.startswith(f"{prefix}."):
        normalized = candidate
    elif "." in candidate:
        # Model returned a full key with another prefix; keep only suffix.
        normalized = f"{prefix}.{candidate.rsplit('.', 1)[-1]}"
    else:
        normalized = f"{prefix}.{candidate}"

    if normalized == prefix:
        return fallback_key
    return normalized


def dedupe_result_keys(results: list[TranslationResult]) -> list[TranslationResult]:
    used: dict[str, str] = {}
    deduped: list[TranslationResult] = []
    for item in results:
        base_key = item.key
        key = base_key
        dup = 2
        while key in used and used[key] != item.source_text:
            key = f"{base_key}_dup{dup}"
            dup += 1
        used[key] = item.source_text
        if key == item.key:
            deduped.append(item)
        else:
            deduped.append(replace(item, key=key))
    return deduped


def read_source_line(project_root: Path, rel_path: Path, line_no: int) -> str:
    full_path = project_root / rel_path
    if not full_path.is_file():
        return ""
    try:
        lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def parse_todo_items(
    todo_path: Path,
    project_root: Path,
    include_checked: bool,
    key_prefix: str,
) -> list[TodoItem]:
    items: list[TodoItem] = []
    lines = todo_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    next_id = 0
    for raw_line in lines:
        match = TODO_LINE_RE.match(raw_line)
        if not match:
            continue

        checked = match.group("mark").lower() == "x"
        if checked and not include_checked:
            continue

        location = match.group("location")
        key = (match.group("key") or "").strip()
        todo_snippet = match.group("snippet").strip()
        parsed = parse_location(location)
        if parsed is None:
            continue
        rel_path, line_no = parsed
        source_line = read_source_line(project_root, rel_path, line_no)
        source_text = source_line if source_line else todo_snippet
        if not has_han(source_text):
            continue
        existing_i18n_key = extract_existing_i18n_key(source_text)
        if existing_i18n_key:
            key = existing_i18n_key
        elif not key:
            key = build_default_key(rel_path, line_no, key_prefix)

        next_id += 1
        items.append(
            TodoItem(
                id=next_id,
                checked=checked,
                location=location,
                key=key,
                rel_path=rel_path,
                line_no=line_no,
                todo_snippet=todo_snippet,
                source_text=source_text,
            )
        )
    return items


def chunked(items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_messages(
    target_locale: str,
    items: list[TodoItem],
    *,
    key_mode: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a software localization assistant. "
        "Translate each text into the requested target locale. "
        "Return JSON only. Do not add commentary.\n"
        "Rules:\n"
        "1) Keep all placeholders unchanged, e.g. {name}, ${value}, %s, %(name)s.\n"
        "2) Keep URLs, code symbols, file paths, and markdown syntax unchanged.\n"
        "3) If text contains Chinese (Han) inside code, translate only the Chinese fragments while preserving code syntax exactly.\n"
        "4) If text is already in target language and has no Chinese, keep it unchanged.\n"
        "5) For any input that contains Chinese, output must not contain Chinese characters unless absolutely unavoidable.\n"
        "6) Translate UI labels/messages even when they are inside quotes in code.\n"
    )
    if key_mode == "llm":
        system_prompt += (
            "7) For each item, suggest a semantic i18n key in `key`.\n"
            "8) Keep `key_prefix` unchanged and append a concise, snake_case semantic suffix.\n"
            "9) Do not use line numbers in generated key suffixes.\n"
            '10) Output format: {"translations": [{"id": <int>, "translated": <string>, "key": <string>}, ...]}.\n'
        )
    else:
        system_prompt += (
            '7) Output format: {"translations": [{"id": <int>, "translated": <string>}, ...]}.\n'
        )

    user_payload = {
        "target_locale": target_locale,
        "key_mode": key_mode,
        "items": [
            {
                "id": item.id,
                "text": item.source_text,
                "key": item.key,
                "key_prefix": key_prefix_from_key(item.key),
            }
            for item in items
        ],
    }
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]


def extract_json_text(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

    # Fallback: locate first '{' and last '}'.
    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last != -1 and first < last:
        return content[first : last + 1]
    return content


def parse_translations(
    content: str,
    expected_ids: set[int],
) -> dict[int, ParsedTranslation]:
    json_text = extract_json_text(content)
    payload = json.loads(json_text)

    rows = payload.get("translations", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise ValueError("Invalid translation payload: translations must be a list")

    result: dict[int, ParsedTranslation] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = row.get("id")
        translated = row.get("translated")
        if not isinstance(item_id, int):
            continue
        if item_id not in expected_ids:
            continue
        if not isinstance(translated, str):
            continue
        key = row.get("key")
        result[item_id] = ParsedTranslation(
            translated_text=translated,
            key=key if isinstance(key, str) and key.strip() else None,
        )
    return result


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(text))


def keep_placeholder_safe(source: str, translated: str) -> str:
    src_placeholders = placeholders(source)
    if not src_placeholders:
        return translated

    dst_placeholders = placeholders(translated)
    if src_placeholders.issubset(dst_placeholders):
        return translated
    return source


def chat_content_from_response(data: dict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Invalid response: choices is empty")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Invalid response: choices[0] is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Invalid response: choices[0].message is missing")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Invalid response: message.content is not string")
    return content


def escape_md(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "<br>")
        .replace("\r", "")
        .strip()
    )


def to_markdown(
    results: list[TranslationResult],
    *,
    source_todo: Path,
    project_root: Path,
    target_locale: str,
    model: str,
) -> str:
    grouped: dict[str, list[TranslationResult]] = defaultdict(list)
    for item in results:
        grouped[item.rel_path.as_posix()].append(item)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = [
        "# i18n Translation Draft",
        "",
        f"- Generated at (UTC): `{now}`",
        f"- Source TODO: `{source_todo.as_posix()}`",
        f"- Project root: `{project_root.as_posix()}`",
        f"- Target locale: `{target_locale}`",
        f"- Model: `{model}`",
        f"- Total items: `{len(results)}`",
        "",
        "> This is a draft. Review translations before applying them to locale files.",
        "",
    ]

    for rel_path in sorted(grouped):
        lines.append(f"## `{rel_path}`")
        lines.append("")
        lines.append("| Line | Key | Source | Translation |")
        lines.append("|---|---|---|---|")
        for item in sorted(grouped[rel_path], key=lambda x: x.line_no):
            lines.append(
                f"| `{item.line_no}` | `{item.key}` | {escape_md(item.source_text)} | {escape_md(item.translated_text)} |"
            )
        lines.append("")
    return "\n".join(lines)


def to_json(results: list[TranslationResult]) -> list[dict[str, str | int]]:
    return [
        {
            "id": item.id,
            "location": item.location,
            "key": item.key,
            "path": item.rel_path.as_posix(),
            "line": item.line_no,
            "source": item.source_text,
            "translated": item.translated_text,
        }
        for item in results
    ]


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate i18n.todo candidates using OpenAI-compatible API."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to dotenv file used for API/model defaults. Default: .env",
    )
    parser.add_argument(
        "--todo",
        type=Path,
        default=None,
        help=(
            "Input TODO markdown path. "
            "Default: <project-root>/i18n.todo.scan.<project_root_name>.md"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file path (.md or .json). "
            "Default: <todo_stem>.translated<todo_suffix>"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve `path:line` entries.",
    )
    parser.add_argument(
        "--key-prefix",
        default="hardcoded",
        help="Fallback key prefix when TODO entries do not contain [key:`...`].",
    )
    parser.add_argument(
        "--target-locale",
        default="en-US",
        help="Target locale label passed to the model.",
    )
    parser.add_argument(
        "--key-mode",
        choices=["llm", "keep"],
        default="llm",
        help=(
            "Key strategy for translated draft. "
            "'llm' asks model to generate semantic key suffixes, "
            "'keep' keeps original TODO keys."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible API base URL (without /v1/chat/completions suffix).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Supports comma-separated keys for rotation.",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--include-checked",
        action="store_true",
        help="Include already checked lines from TODO.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call API. Output source text as translation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue when a batch fails and fallback to source text for that batch.",
    )
    parser.add_argument(
        "--mark-checked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update source TODO checklist to [x] for translated entries.",
    )
    args = parser.parse_args()

    env_file = args.env_file.resolve()
    loaded_vars = load_env_file(env_file)
    if env_file.is_file():
        print(f"Loaded {loaded_vars} env var(s) from {env_file}")

    model = args.model or os.getenv("GROK_MODEL") or os.getenv("OPENAI_MODEL") or "grok-2-latest"
    api_base = args.api_base or os.getenv("GROK_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("GROK_API_KEY") or os.getenv("OPENAI_API_KEY")

    project_root = args.project_root.resolve()
    key_prefix = normalize_key_prefix(args.key_prefix)
    todo_path = args.todo.resolve() if args.todo else default_todo_path(project_root)
    output_path = args.output.resolve() if args.output else default_output_path(todo_path)

    if not todo_path.is_file():
        raise FileNotFoundError(
            f"TODO file not found: {todo_path}. "
            "Use --todo to set a custom input file."
        )

    items = parse_todo_items(
        todo_path=todo_path,
        project_root=project_root,
        include_checked=args.include_checked,
        key_prefix=key_prefix,
    )
    if args.start_index > 0:
        items = items[args.start_index :]
    if args.max_items > 0:
        items = items[: args.max_items]

    if not items:
        print("No translatable TODO items found.")
        return 0

    api_keys = parse_api_keys(api_key)

    if not args.dry_run and (not api_base or not api_keys):
        raise ValueError(
            "Missing API config. Provide --api-base/--api-key, or set "
            "GROK_API_BASE/GROK_API_KEY (or OPENAI_BASE_URL/OPENAI_API_KEY) in .env."
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    results: list[TranslationResult] = []
    if args.dry_run:
        for item in items:
            results.append(
                TranslationResult(
                    id=item.id,
                    location=item.location,
                    key=item.key,
                    rel_path=item.rel_path,
                    line_no=item.line_no,
                    source_text=item.source_text,
                    translated_text=item.source_text,
                )
            )
    else:
        client = OpenAICompatClient(
            api_base=api_base,
            api_key=api_key,
            model=model,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
        )
        print(f"Using {len(api_keys)} API key(s)")
        batches = chunked(items, args.batch_size)
        total = len(items)
        processed = 0
        processed_lock = threading.Lock()
        ordered_results: dict[int, list[TranslationResult]] = {}

        def fallback_results(batch: list[TodoItem]) -> list[TranslationResult]:
            return [
                TranslationResult(
                    id=item.id,
                    location=item.location,
                    key=item.key,
                    rel_path=item.rel_path,
                    line_no=item.line_no,
                    source_text=item.source_text,
                    translated_text=item.source_text,
                )
                for item in batch
            ]

        def translate_with_split(
            batch: list[TodoItem],
            *,
            key_offset: int,
            depth: int = 0,
        ) -> list[TranslationResult]:
            if not batch:
                return []

            try:
                messages = build_messages(
                    args.target_locale,
                    batch,
                    key_mode=args.key_mode,
                )
                response = client.chat_json(messages, key_offset=key_offset)
                content = chat_content_from_response(response)
                expected_ids = {item.id for item in batch}
                mapping = parse_translations(content, expected_ids)

                unresolved = [item for item in batch if item.id not in mapping]
                resolved_map: dict[int, ParsedTranslation] = dict(mapping)

                if unresolved:
                    if len(unresolved) == len(batch):
                        raise ValueError("Model returned no valid translations for this batch")
                    if not args.continue_on_error:
                        raise ValueError(
                            f"Model returned incomplete translations: missing {len(unresolved)} of {len(batch)}"
                        )
                    retry_results = translate_with_split(
                        unresolved,
                        key_offset=key_offset + 1,
                        depth=depth + 1,
                    )
                    for rr in retry_results:
                        resolved_map[rr.id] = ParsedTranslation(
                            translated_text=rr.translated_text,
                            key=rr.key,
                        )

                out: list[TranslationResult] = []
                for item in batch:
                    parsed = resolved_map.get(
                        item.id,
                        ParsedTranslation(translated_text=item.source_text, key=item.key),
                    )
                    translated = keep_placeholder_safe(item.source_text, parsed.translated_text)
                    new_key = item.key
                    if args.key_mode == "llm" and should_regenerate_key(item.key):
                        if parsed.key:
                            new_key = normalize_generated_key(parsed.key, item.key)
                    out.append(
                        TranslationResult(
                            id=item.id,
                            location=item.location,
                            key=new_key,
                            rel_path=item.rel_path,
                            line_no=item.line_no,
                            source_text=item.source_text,
                            translated_text=translated,
                        )
                    )
                return out
            except Exception as exc:
                if not args.continue_on_error:
                    raise

                first = batch[0].location if batch else "unknown"
                if len(batch) == 1:
                    print(f"[WARN] Item failed near {first}: {exc}")
                    print("[WARN] Fallback to source text for this item.")
                    return fallback_results(batch)

                indent = "  " * min(depth, 6)
                print(
                    f"[WARN] {indent}Batch failed near {first}: {exc}. "
                    f"Split {len(batch)} -> {len(batch) // 2}+{len(batch) - len(batch) // 2}"
                )
                mid = len(batch) // 2
                left = translate_with_split(
                    batch[:mid],
                    key_offset=key_offset,
                    depth=depth + 1,
                )
                right = translate_with_split(
                    batch[mid:],
                    key_offset=key_offset + mid,
                    depth=depth + 1,
                )
                return left + right

        def process_batch(
            batch_index: int,
            batch: list[TodoItem],
        ) -> tuple[int, list[TranslationResult]]:
            batch_results = translate_with_split(
                batch,
                key_offset=batch_index,
                depth=0,
            )

            if args.sleep_between > 0:
                time.sleep(args.sleep_between)
            with processed_lock:
                nonlocal processed
                processed += len(batch)
                print(f"Processed {processed}/{total}")
            return batch_index, batch_results

        if args.concurrency == 1:
            for batch_index, batch in enumerate(batches):
                idx, batch_results = process_batch(batch_index, batch)
                ordered_results[idx] = batch_results
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as executor:
                future_map = {
                    executor.submit(process_batch, idx, batch): idx
                    for idx, batch in enumerate(batches)
                }
                for future in concurrent.futures.as_completed(future_map):
                    idx, batch_results = future.result()
                    ordered_results[idx] = batch_results

        for idx in range(len(batches)):
            results.extend(ordered_results.get(idx, []))

    results = dedupe_result_keys(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps(to_json(results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        output_path.write_text(
            to_markdown(
                results,
                source_todo=todo_path,
                project_root=project_root,
                target_locale=args.target_locale,
                model=model,
            ),
            encoding="utf-8",
        )

    if args.mark_checked and not args.dry_run:
        completed = {
            item.location: item.key
            for item in results
            if not has_han(item.translated_text)
        }
        updated = update_todo_checkmarks(todo_path, completed)
        print(f"TODO checked: {updated}")

    print(f"Input items: {len(items)}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
