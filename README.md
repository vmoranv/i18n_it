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
<project_root>/i18n.todo.scan.<project_root_name>.md
```

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

默认支持示例：`py/pyi`、`js/ts/jsx/tsx/mjs/cjs`、`vue/svelte/astro/mdx`、`java/kt/go/rs/php/cs/c/cpp/swift/dart`、`rb/sh`、`yaml/toml/ini/conf/properties`、`html/xml/svg`。

## 2) 翻译 TODO（OpenAI 兼容接口，如 Grok）

```bash
python translate_i18n_todo.py --project-root <project_root> --target-locale en-US --concurrency 8 --continue-on-error
```

翻译产物会保留 `Key` 列；默认会把已处理条目在源 TODO 中自动打勾（`- [x]`）。  
如需关闭自动打勾可加 `--no-mark-checked`。
默认 `--key-mode llm`：会让 LLM 生成语义化 key 后缀（不再只用 `line_xxx`）。  
如需保持原 key（兼容旧流程）可使用 `--key-mode keep`。

默认输入路径（自动推导）：

```text
<project_root>/i18n.todo.scan.<project_root_name>.md
```

默认输出路径（自动推导）：

```text
<todo_stem>.translated<todo_suffix>
```

可选：手动指定输入输出

```bash
python translate_i18n_todo.py --todo <todo_md> --output <translated_md> --env-file .env --target-locale en-US
```

## 3) 主工作流：同步到 i18n locale JSON（手动映射，支持并发）

先复制映射模板并按项目实际路径修改：

```bash
Copy-Item sync_i18n_locale.mapping.example.json sync_i18n_locale.mapping.json
```

执行同步（先 dry-run）：

```bash
python sync_i18n_locale.py \
  --translated <translated_md> \
  --i18n-root <locale_root_dir> \
  --mapping sync_i18n_locale.mapping.json \
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
  --mapping sync_i18n_locale.mapping.json \
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
