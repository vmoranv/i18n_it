# TODO

- [x] 给翻译流程增加 locale map（`translate_i18n_todo.locale_map.json`），支持 `zh-CN` 等标签的人类可读提示。
- [x] 给同步映射增加 locale 占位符（`{locale}` / `{locale_lower}`），避免把目标路径写死为 `en-us`。
- [x] 将项目示例映射移入 `examples/mappings/`，主流程文档改为通用说明。
- [ ] 给 `sync_i18n_locale.py` 增加最小示例 mapping 的 `--init-mapping` 生成功能。
- [ ] 为关键脚本补充最小化自测（解析 markdown / mapping 渲染 / locale 推断）。
