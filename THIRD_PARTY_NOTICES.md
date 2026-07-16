# 第三方组件声明

本项目自身源码与文档采用 `AGPL-3.0-only`。下列第三方组件继续适用其各自的上游许可证，本文件不改变或替代上游许可条款。

| 组件 | 用途 | 上游许可证 |
| --- | --- | --- |
| [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license) | PDF 解析、渲染与裁剪 | GNU AGPL v3 或 Artifex Commercial License；本项目按 GNU AGPL v3 路径使用；本项目未取得或主张 Artifex 商业许可证 |
| [pdfplumber](https://github.com/jsvine/pdfplumber) / [pdfminer.six](https://github.com/pdfminer/pdfminer.six) | 表格与文本解析 | MIT |
| [FastAPI](https://github.com/fastapi/fastapi) | HTTP API | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | ASGI 服务 | BSD-3-Clause |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | DeepSeek OpenAI 兼容接口客户端 | Apache-2.0 |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) / [Alembic](https://github.com/sqlalchemy/alembic) | 数据访问与迁移 | MIT |
| [psycopg](https://github.com/psycopg/psycopg) | PostgreSQL 客户端 | LGPL-3.0-only |
| [PostgreSQL](https://www.postgresql.org/about/licence/) | 数据库服务 | PostgreSQL License |
| [Chroma](https://github.com/chroma-core/chroma) | 向量存储 | Apache-2.0 |
| [Sentence Transformers](https://github.com/huggingface/sentence-transformers) / [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Embedding 运行与模型获取 | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | 张量与模型运行时 | BSD-3-Clause |
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) / [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | OCR | Apache-2.0 |
| [latex2mathml](https://github.com/roniemartinez/latex2mathml) | 公式渲染 | MIT |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Embedding 模型 | MIT |

完整依赖树及其许可证以构建时实际安装版本的上游元数据为准。DeepSeek 是用户自行连接的外部服务，不作为软件组件随本项目分发。
