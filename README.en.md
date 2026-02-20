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

## 3) Main workflow: sync into i18n locale JSON (manual mapping, concurrent)

Use and maintain these mapping files in this repo (recommended to keep them tracked in Git):

- `sync_i18n_locale.astrbot.mapping.json`
- `sync_i18n_locale.dashboard.mapping.json`

Run sync first in dry-run mode:

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping sync_i18n_locale.mapping.json \
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
  --mapping sync_i18n_locale.mapping.json \
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
