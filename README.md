# i18n 工具

[English](README.en.md)

用于一键扫描硬编码中文、调用 OpenAI 兼容接口翻译、并同步到 i18n locale JSON 的实用脚本。

## 0) `.env` 配置

先复制模板并填写密钥：

```bash
Copy-Item .env.example .env
```

或直接创建 `.env`：

```dotenv
GROK_API_BASE=https://api.x.ai
GROK_API_KEY=your_grok_api_key
GROK_MODEL=grok-4.1-fast
```

`translate_i18n_todo.py` 会自动加载 `.env`，CLI 参数优先级更高。

可选：你也可以维护 `locale_map.json`（已内置 `zh-CN` 示例），
让 LLM 在接收 `--target-locale` 时拿到更清晰的人类可读语种提示。

`locale_map` 支持两种常用写法：

```json
{
  "ja-JP": "Japanese (Japan)",
  "jp": "ja-JP"
}
```

- `ja-JP: "Japanese (Japan)"`：给语种提供人类可读提示。
- `jp: "ja-JP"`：把简写别名映射到标准 locale（之后可直接用 `jp`）。

## 1) 扫描硬编码中文

```bash
python extract_i18n_todo.py --root <project_root>
```

扫描结果中的每一条 TODO 都会带稳定翻译键，例如：

```text
- [ ] `src/App.vue:11` [key:`hardcoded.src.app.line_11`] <v-btn ...>关闭</v-btn>
```

默认输出路径（自动推导）：

```text
<artifacts_dir>/<project_root_name>/i18n.todo.scan.<project_root_name>.md
```

默认 `artifacts_dir` 为当前目录下的 `output/`，可用 `--artifacts-dir` 覆盖。

可选：覆盖输出路径

```bash
python extract_i18n_todo.py --root <project_root> --output <todo_md>
```

可选：覆盖扫描扩展名

```bash
python extract_i18n_todo.py --root <project_root> --extensions ".py,.ts,.tsx,.vue,.svelte,.astro,.java,.kt"
```

可选：自定义 key 前缀

```bash
python extract_i18n_todo.py --root <project_root> --key-prefix hardcoded
```

可选：统一产物目录（推荐）

```bash
python extract_i18n_todo.py --root <project_root> --artifacts-dir output
```

默认支持示例：`py/pyi`、`js/ts/jsx/tsx/mjs/cjs`、`vue/svelte/astro/mdx`、`java/kt/go/rs/php/cs/c/cpp/swift/dart`、`rb/sh`、`yaml/toml/ini/conf/properties`、`html/xml/svg`。

## 2) 翻译 TODO（OpenAI 兼容接口，如 Grok）

```bash
python translate_i18n_todo.py --project-root <project_root> --target-locale en-US --concurrency 8 --continue-on-error
```

翻译产物会保留 `Key` 列；默认会把已处理条目在源 TODO 中自动打勾（`- [x]`）。  
如需关闭自动打勾可加 `--no-mark-checked`。
默认 `--key-mode llm`：会让 LLM 生成语义化 key 后缀（不再只用 `line_xxx`）。  
如需保持原 key（兼容旧流程）可使用 `--key-mode keep`。

例如翻译到简体中文：

```bash
python translate_i18n_todo.py --project-root <project_root> --target-locale zh-CN
```

如需指定自定义 locale map：

```bash
python translate_i18n_todo.py --project-root <project_root> --target-locale zh-CN --locale-map locale_map.json
```

多语种并行翻译（支持简写，如 `jp`）：

```bash
python translate_i18n_todo.py \
  --project-root <project_root> \
  --target-locales "en-US,jp,ko-KR" \
  --locale-map locale_map.json \
  --concurrency 8 \
  --locale-concurrency 2 \
  --continue-on-error
```

多语模式会在草稿里生成多列（例如 `translated_en-us`、`translated_jp`、`translated_ko-kr`），并写入 `Locale columns` 元数据，供同步脚本自动识别对应 locale。

默认输入路径（自动推导）：

```text
<artifacts_dir>/<project_root_name>/i18n.todo.scan.<project_root_name>.md
```

默认输出路径（自动推导）：

```text
<todo_stem>.translated<todo_suffix>
```

当不传 `--todo` 且使用默认路径时，翻译产物会落在同一 `output/<project_root_name>/` 目录。

可选：手动指定输入输出

```bash
python translate_i18n_todo.py --todo <todo_md> --output <translated_md> --env-file .env --target-locale en-US
```

可选：统一产物目录（推荐）

```bash
python translate_i18n_todo.py --project-root <project_root> --artifacts-dir output --target-locale en-US
```

## 3) 主工作流：同步到 locale 文件（JSON/FTL，手动映射，支持并发）

`sync_i18n_locale.py` 是通用同步器，**必须**通过 `--mapping` 指定“源码路径 -> locale 文件”的映射规则。

最小 mapping 示例：

```json
{
  "mappings": [
    { "source": "src/components/**", "file": "{locale}/features/common.json", "prefix": "components" },
    { "source": "src/views/**", "file": "{locale}/features/views.json" },
    { "source": "**", "file": "{locale}/core/common.json" }
  ]
}
```

支持占位符：`{locale}` / `{locale_lower}`（可由翻译草稿中的 `Target locale` 自动推断，也可用 `--locale` 覆盖）。

仓库内的通用示例映射（可选）：

- `examples/mappings/generic.mapping.json`

如果草稿是多语模式（包含多个 `translated_*` 列），可直接用 `--column all-translated` 一次性并行应用到对应 locale 目标文件：

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping <mapping_json> \
  --column all-translated \
  --concurrency 8 \
  --dry-run
```

执行同步（先 dry-run）：

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping <mapping_json> \
  --locale zh-CN \
  --concurrency 8 \
  --dry-run
```

确认无误后去掉 `--dry-run` 正式写入 locale JSON。  
同步成功后默认也会回写源 TODO 的勾选状态（`- [x]`），可用 `--no-mark-checked` 关闭。  
可选输出报告：

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping <mapping_json> \
  --report-json sync.report.json
```

## 4) 可选：按行回写源代码（不推荐作为 i18n 主流程）

默认是 `t('key')` 模板模式（优先用于 i18n 化）：

```bash
python apply_i18n_translation_draft.py --translated <translated_md> --project-root <project_root> --dry-run
```

确认无误后去掉 `--dry-run` 正式写回。

若你确实要“直接替换文本”而不是模板，可显式指定：

```bash
python apply_i18n_translation_draft.py --translated <translated_md> --project-root <project_root> --column translated
```

## 5) TODO

见 `TODO.md`（后续通用能力增强项）。

## 6) FTL 换行 token 基线回填（`{"\u000A"}`）

当历史提交里某些 FTL key 含有 `{"\u000A"}`，但后续被清洗脚本误去除时，可用以下脚本按基线自动回填：

```bash
python restore_ftl_newline_tokens.py --repo-root ../target-repo --base-commit 350a11be --dry-run
python restore_ftl_newline_tokens.py --repo-root ../target-repo --base-commit 350a11be --apply
```

默认会扫描：

```text
i18n/locales/**/*.ftl
```

可选参数：

- `--glob`：覆盖扫描范围（例如只处理 `builtin_stars.ftl`）
- `--token`：覆盖回填 token（默认 `{"\u000A"}`）

## 7) 可选：参数化导入检查/修复

通用 Python 导入修复器（不绑定任何项目）：

```bash
python ensure_t_imports.py \
  --root <project_root> \
  --target-module <canonical_module> \
  --symbol t \
  --legacy-module <legacy_module> \
  --auto-add-on-call \
  --dry-run
```

用于 CI 校验（发现可修复项时返回非 0）：

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

前端 `t(...)` 导入补全脚本：

```bash
python ensure_frontend_t_imports.py --frontend-root <frontend_root>
```
