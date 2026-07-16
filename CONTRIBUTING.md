# 贡献指南

提交贡献即表示你有权提交相关代码，并同意该贡献按项目的 `AGPL-3.0-only` 许可证发布。

## 本地运行

Windows：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_services.ps1 -BuildLocal`

Linux / macOS：`sh scripts/start_services.sh --build-local`

## 验证

提交前运行 `python -m compileall -q src scripts tests` 和 `python -m pytest -q`。

不得提交 PDF、用户数据、模型文件、数据库、私有评估、`.env`、API 密钥或生成日志。
