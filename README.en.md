# i18n Tools

[中文](README.md)

Utilities to scan hardcoded Han text, translate with an OpenAI-compatible API, and sync results into locale JSON files.

## 0) Configure `.env`

Copy template and fill credentials:

```bash
Copy-Item .env.example .env
```

Or create `.env` directly:

```dotenv
GROK_API_BASE=https://api.x.ai
GROK_API_KEY=your_grok_api_key
GROK_MODEL=grok-4.1-fast
```

`translate_i18n_todo.py` auto-loads `.env`; CLI args override env values.

Optional: maintain `translate_i18n_todo.locale_map.json` (includes `zh-CN` examples)
to provide clearer human-readable locale hints for the model.

`locale_map` supports two useful patterns:

```json
{
  "ja-JP": "Japanese (Japan)",
  "jp": "ja-JP"
}
```

- `ja-JP: "Japanese (Japan)"`: human-readable hint for the locale.
- `jp: "ja-JP"`: shorthand alias to canonical locale (you can pass `jp` later).

## 1) Extract hardcoded Han text

```bash
python extract_i18n_todo.py --root <project_root>
```

Each TODO item now includes a stable translation key, for example:

```text
- [ ] `src/App.vue:11` [key:`hardcoded.src.app.line_11`] <v-btn ...>关闭</v-btn>
```

Auto-derived default output path:

```text
<artifacts_dir>/<project_root_name>/i18n.todo.scan.<project_root_name>.md
```

By default, `artifacts_dir` is `./output` (override with `--artifacts-dir`).

Optional explicit output:

```bash
python extract_i18n_todo.py --root <project_root> --output <todo_md>
```

Optional extensions override:

```bash
python extract_i18n_todo.py --root <project_root> --extensions ".py,.ts,.tsx,.vue,.svelte,.astro,.java,.kt"
```

Optional custom key prefix:

```bash
python extract_i18n_todo.py --root <project_root> --key-prefix hardcoded
```

Optional centralized artifacts directory (recommended):

```bash
python extract_i18n_todo.py --root <project_root> --artifacts-dir output
```

Default extension coverage includes examples like `py/pyi`, `js/ts/jsx/tsx/mjs/cjs`, `vue/svelte/astro/mdx`, `java/kt/go/rs/php/cs/c/cpp/swift/dart`, `rb/sh`, `yaml/toml/ini/conf/properties`, `html/xml/svg`.

## 2) Translate TODO with OpenAI-compatible API (e.g., Grok)

```bash
python translate_i18n_todo.py --project-root <project_root> --target-locale en-US --concurrency 8 --continue-on-error
```

The translated markdown keeps the `Key` column.  
By default, translated TODO items are auto-marked as checked (`- [x]`) in the source TODO.  
Disable this via `--no-mark-checked`.

Example translating into Simplified Chinese:

```bash
python translate_i18n_todo.py --project-root <project_root> --target-locale zh-CN
```

Optional custom locale map:

```bash
python translate_i18n_todo.py --project-root <project_root> --target-locale zh-CN --locale-map translate_i18n_todo.locale_map.json
```

Parallel multi-locale translation (with shorthand like `jp`):

```bash
python translate_i18n_todo.py \
  --project-root <project_root> \
  --target-locales "en-US,jp,ko-KR" \
  --locale-map translate_i18n_todo.locale_map.json \
  --concurrency 8 \
  --locale-concurrency 2 \
  --continue-on-error
```

In multi-locale mode, the draft contains one column per locale (for example `translated_en-us`, `translated_jp`, `translated_ko-kr`) plus `Locale columns` metadata for automatic sync routing.

Auto-derived default input path:

```text
<artifacts_dir>/<project_root_name>/i18n.todo.scan.<project_root_name>.md
```

Auto-derived default output path:

```text
<todo_stem>.translated<todo_suffix>
```

When `--todo` is omitted and default paths are used, translated drafts are written to the same `output/<project_root_name>/` folder.

Optional explicit input/output:

```bash
python translate_i18n_todo.py --todo <todo_md> --output <translated_md> --env-file .env --target-locale en-US
```

Optional centralized artifacts directory (recommended):

```bash
python translate_i18n_todo.py --project-root <project_root> --artifacts-dir output --target-locale en-US
```

## 3) Main workflow: sync into locale files (JSON/FTL, manual mapping, concurrent)

`sync_i18n_locale.py` is a generic syncer. You must provide a `--mapping` file that routes source paths to locale target files.

Minimal mapping example:

```json
{
  "mappings": [
    { "source": "src/components/**", "file": "{locale}/features/common.json", "prefix": "components" },
    { "source": "src/views/**", "file": "{locale}/features/views.json" },
    { "source": "**", "file": "{locale}/core/common.json" }
  ]
}
```

Supported placeholders: `{locale}` / `{locale_lower}` (auto-inferred from draft `Target locale`, overridable with `--locale`).

Generic example mappings in this repo (optional):

- `examples/mappings/generic.mapping.json`

If the draft is multi-locale (multiple `translated_*` columns), use `--column all-translated` to apply all locale columns in one pass:

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping <mapping_json> \
  --column all-translated \
  --concurrency 8 \
  --dry-run
```

Run sync first in dry-run mode:

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping <mapping_json> \
  --locale zh-CN \
  --concurrency 8 \
  --dry-run
```

Remove `--dry-run` to write locale JSON.  
Successful sync also marks source TODO items as checked by default (disable with `--no-mark-checked`).  
Optional report output:

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping <mapping_json> \
  --report-json sync.report.json
```

## 4) Optional: apply draft back to source lines (not the main i18n flow)

Default mode now applies `t('key')` template replacements (preferred for i18n):

```bash
python apply_i18n_translation_draft.py --translated <translated_md> --project-root <project_root> --dry-run
```

Remove `--dry-run` to apply changes.

If you explicitly want direct text replacement, use:

```bash
python apply_i18n_translation_draft.py --translated <translated_md> --project-root <project_root> --column translated
```

## 5) TODO

See `TODO.md` for remaining generic enhancements.

## 6) Restore FTL Newline Tokens from Baseline (`{"\u000A"}`)

If `{"\u000A"}` tokens were accidentally removed from FTL values by cleanup scripts,
restore them from a baseline commit:

```bash
python restore_ftl_newline_tokens.py --repo-root ../target-repo --base-commit 350a11be --dry-run
python restore_ftl_newline_tokens.py --repo-root ../target-repo --base-commit 350a11be --apply
```

Default scan target:

```text
i18n/locales/**/*.ftl
```

Optional args:

- `--glob`: override file scope (e.g., only `builtin_stars.ftl`)
- `--token`: override token to restore (default `{"\u000A"}`)

## 7) Optional: Parameterized Import Check/Fix

Generic Python import normalizer (project-agnostic):

```bash
python ensure_t_imports.py \
  --root <project_root> \
  --target-module <canonical_module> \
  --symbol t \
  --legacy-module <legacy_module> \
  --auto-add-on-call \
  --dry-run
```

CI check mode (non-zero exit code when fixes are needed):

```bash
python ensure_t_imports.py \
  --root <project_root> \
  --target-module <canonical_module> \
  --symbol t \
  --legacy-module <legacy_module> \
  --auto-add-on-call \
  --dry-run \
  --fail-on-change
```

Frontend `t(...)` import patcher:

```bash
python ensure_frontend_t_imports.py --frontend-root <frontend_root>
```
