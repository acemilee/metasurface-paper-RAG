# metasurface-paper-RAG

面向超表面研究论文的本地动态 RAG 知识库。系统将 PDF 入库、领域正向准入、BGE-M3 检索、证据约束问答、原文引用和公式可靠性控制整合在一个 GUI 中。

当前版本：`v0.1.0`

## 主要功能

- 支持批量上传 PDF，并在本地完成文本解析、切片、向量化和论文结构分析。
- 通过领域正向准入判断论文是否具有足够的超表面研究证据，避免污染。
- 按 PDF 原始文件名检索论文，支持单篇、多篇或全库问答范围。
- 使用 BGE-M3 和 Chroma 执行本地语义检索，使用 PostgreSQL 保存论文、任务和审计状态。
- 使用 DeepSeek 进行问题改写、证据约束回答和回答审计；关键实体无法链接或证据不足时拒绝补写。
- 展示引用原文与页码。
- 支持入库进度、问答阶段状态、论文删除、会话管理和服务就绪检查。

## 运行要求

普通用户只需准备：

- Windows 10/11、Linux 或 macOS；
- Docker Desktop，或 Docker Engine 与 Docker Compose v2；
- 可访问 Hugging Face 和 DeepSeek API 的网络；
- 用户自己的 DeepSeek API 密钥。

v0.1.0 预构建镜像只支持 `linux/amd64`。ARM64 设备暂不在本版本支持范围内。

## 快速开始

克隆仓库：

```bash
git clone https://github.com/acemilee/metasurface-paper-RAG.git
cd metasurface-paper-RAG
```

普通用户默认使用固定版本的预构建镜像：

```text
ghcr.io/acemilee/metasurface-paper-rag:0.1.0
```

### Windows

启动 Docker Desktop，等待 Docker Engine 就绪，然后双击：

```text
start.cmd
```

也可以在 PowerShell 中执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_services.ps1
```

不希望启动器自动打开浏览器时：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_services.ps1 -NoBrowser
```

### Linux / macOS

```sh
sh scripts/start_services.sh
```

启动器会拉取固定版本镜像、准备 BGE-M3 模型、执行数据库迁移，并等待所有服务通过就绪检查。成功后访问：

```text
http://127.0.0.1:8010
```

首次启动需要拉取应用镜像。若本地尚无 `models/BAAI-bge-m3`，还会下载约 2.3 GB 的 BGE-M3 模型；后续启动会复用已有镜像、模型和数据。

## 基本使用流程

1. 打开 `http://127.0.0.1:8010`，上传一篇或多篇 PDF。
2. 等待解析与领域正向准入完成。满足准入要求的论文进入 `accepted`；证据不足、相互冲突或依赖异常的论文进入 `review_required`，不会自动写入检索索引。
3. 对 `review_required` 论文检查判定证据。只有确认属于目标领域时才人工放行。
4. 在论文库中按 PDF 原始文件名检索，并选择单篇、多篇或全库范围。
5. 在 GUI 中连接 DeepSeek API 密钥，输入问题并核对答案后的引用、页码、公式状态和审计结果。

领域准入是降低知识库污染风险的安全门，不代表对论文质量、学术真实性或研究结论作出评价。人工放行会覆盖自动准入结论，应在核验论文内容后使用。

## 数据与安全

论文和索引默认保存在本机：

- `data/uploads`：上传的 PDF；
- `data/parsed`：解析页数据；
- `data/chroma`：本地向量索引；
- `models/BAAI-bge-m3`：Embedding 模型；
- `rag_paper_rag_postgres`：Docker PostgreSQL 数据卷。

DeepSeek API 密钥通过 GUI 提交，只保存在带有效期的服务端进程内存中，不写入 Compose、镜像、数据库、浏览器存储或 `.env.example`；服务重启后需要重新输入。

论文文件、解析数据和向量检索在本地处理。进行问答时，用户问题、必要的会话上下文和检索出的论文证据会发送给 DeepSeek API 以生成和审计回答。请根据论文的数据使用要求和所使用模型服务的隐私条款决定是否提交内容。

GUI 仅绑定到 `127.0.0.1:8010`，默认不向局域网或公网开放。本项目没有为公网部署提供身份认证，不应直接暴露端口。

## 日常管理

查看容器状态：

```powershell
docker compose ps
```

查看最近日志：

```powershell
docker compose logs --tail 200
```

Windows 停止服务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop_services.ps1
```

Linux / macOS 停止服务：

```sh
sh scripts/stop_services.sh
```

停止命令不会删除论文、解析结果、向量库、BGE-M3 模型或 PostgreSQL 数据卷。不要执行 `docker compose down -v`，除非你明确准备永久删除数据库卷。

## 故障排查

### Docker daemon 不可用

启动 Docker Desktop，等待 Docker Engine 就绪，然后检查：

```powershell
docker info
docker compose version
```

### 8010 端口冲突

Windows 可执行：

```powershell
Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue
```

停止占用端口的程序后重新运行启动器。

### `model-init` 失败

```powershell
docker compose logs --tail 200 model-init
```

确认能够访问 Hugging Face，并检查 `models` 所在磁盘是否有足够空间。失败下载使用临时 `.partial` 目录，不会把不完整模型报告为就绪。

### `migrate` 失败

```powershell
docker compose logs --tail 200 migrate postgres
```

迁移失败时 API 和 Worker 会被阻断，不会在旧数据库结构上显示服务就绪。

### 页面打不开或 `/ready` 失败

```powershell
docker compose ps
docker compose logs --tail 200 embedding worker api
Invoke-RestMethod http://127.0.0.1:8010/ready | ConvertTo-Json -Depth 5
```

容器处于运行状态不代表系统已经就绪。只有 PostgreSQL、Chroma、Worker 和 Embedding 均可用时，`/ready` 才会返回成功。

## 贡献者本地构建

普通用户无需在本机编译应用镜像。需要修改源码的贡献者可以显式构建本地镜像：

```powershell
docker build --tag paper-rag:local .
$env:PAPER_RAG_IMAGE='paper-rag:local'
docker compose up --detach --no-build --wait --wait-timeout 1800
```

安装开发依赖并运行测试：

```powershell
python -m pip install -e ".[dev]"
python -m compileall -q src scripts tests
python -m pytest -q
```

真实 Compose 生命周期测试必须显式启用，且不会删除数据卷：

```powershell
$env:PAPER_RAG_RUN_COMPOSE_TESTS='1'
python -m pytest tests/integration/test_compose_lifecycle.py -v -s
```

## v0.1.0 已知限制

- 预构建镜像仅提供 `linux/amd64`，尚未提供 ARM64 镜像。
- 首次运行需要下载应用镜像和约 2.3 GB 的 BGE-M3 模型。
- 问答依赖用户自行提供的 DeepSeek API 密钥和可用的 DeepSeek 网络服务，固定LLM为deepseek-v4-flash，支持1M上下文。
- 扫描质量较差、版式复杂或公式提取不可靠的 PDF 可能只能返回原文定位，系统不会猜测缺失内容。
- 当前没有论文重命名、引用点击跳转 PDF 对应页、远程多用户认证和托管服务。
- v0.1.0 是本地 MVP，不承诺生产级服务等级协议（SLA）。

## 超表面领域参考文献

以下文献可作为超表面相关研究的参考示例。

[1] HUANG C, SONG J, JI C, et al. Simultaneous control of absorbing frequency and amplitude using graphene capacitor and active frequency-selective surface[J]. IEEE Transactions on Antennas and Propagation, 2021, 69(3): 1793-1798. DOI: [10.1109/TAP.2020.3011115](https://doi.org/10.1109/TAP.2020.3011115).

[2] 周洪澄, 余潇然, 王豫, 等. 电控可重构极化调控超表面研究进展[J]. 雷达学报, 2024, 13(3): 696-713. DOI: [10.12000/JR23230](https://doi.org/10.12000/JR23230).

## 许可证

本项目中由项目作者拥有版权的源码与文档采用 [GNU Affero General Public License v3](LICENSE) 许可，SPDX 标识为 `AGPL-3.0-only`。第三方组件继续适用其各自许可证，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

通过网络向用户提供本项目修改版本时，必须按照 AGPL v3 向这些用户提供对应版本的完整源码。
