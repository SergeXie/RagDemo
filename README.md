# 🏢 企业级智能客服咨询系统

基于 **LangChain + FastAPI + Redis + FAISS** 的企业级智能客服系统，实现多工具 Agent 协同、多轮对话记忆与摘要压缩、RAG 企业知识库检索、SSE 流式响应。

## 系统架构

```
┌────────────────────────────────────────────────────────────────┐
│  前端 (frontend/)  纯静态 HTML/JS/CSS，SSE 流式渲染             │
└────────────┬───────────────────────────────────────────────────┘
             │ fetch + ReadableStream (SSE)
┌────────────▼───────────────────────────────────────────────────┐
│  FastAPI 网关 (app/main.py)                                     │
│  ├── /api/v1/chat/*        SSE 流式对话 / 会话管理              │
│  ├── /api/v1/documents/*   文档上传 / 列表 / 删除               │
│  └── /api/v1/health        健康检查（含 Redis 探活）             │
├────────────────────────────────────────────────────────────────┤
│  Agent 编排层 (app/agent/service.py)  LangGraph ReAct Agent     │
│  ├── 🔍 document_search    FAISS 知识库检索工具                 │
│  ├── 🧮 calculator         AST 白名单安全计算器                 │
│  └── 🌐 web_search         OpenAI Responses API 联网搜索        │
├──────────────────┬─────────────────────────────────────────────┤
│  RAG 流水线       │  对话记忆 (app/memory/)                     │
│  (app/rag/)       │  ├── Redis 存储（多用户 key 隔离）          │
│  ├── 多格式解析   │  ├── LLM 摘要压缩（防上下文丢失）           │
│  ├── 递归分块     │  └── 自动降级：Redis 不可用 -> 内存          │
│  ├── Embedding    │                                             │
│  └── FAISS 持久化 │                                             │
└──────────────────┴─────────────────────────────────────────────┘
             │
   ┌─────────▼──────────┐     ┌──────────────────────┐
   │ Redis (多轮记忆)    │     │ LLM / Embedding API  │
   │ docker-compose 启动 │     │ OpenAI 兼容接口      │
   └────────────────────┘     └──────────────────────┘
```

## 核心特性

### 1️⃣ 多工具智能 Agent（LangGraph ReAct）

- **document_search**：企业制度文档检索，命中结果同时推送给前端展示引用来源
- **calculator**：AST 白名单解析的安全计算器，杜绝任意代码执行
- **web_search**：基于 OpenAI Responses API 的 `web_search` 工具，支持火山方舟等兼容网关
- 工具调用过程通过 SSE 实时呈现（`tool_start` / `tool_end` 事件）

### 2️⃣ Redis 多轮对话记忆 + 摘要压缩

- 消息按 `{prefix}:conv:{user_id}:{cid}:messages` 存入 Redis List
- **多用户数据物理隔离**：user_id 编入 key + owner 二次校验，杜绝越权访问
- **摘要算法**：消息数超过阈值（默认 20 条）时，自动将早期对话
  连同旧摘要交给 LLM 压缩为一段摘要，仅保留最近 6 条原文
- 会话级 asyncio.Lock 防止并发重复摘要；LLM 失败时降级为截断拼接
- Redis 不可达时自动降级为内存后端（`MEMORY_BACKEND=auto`）

### 3️⃣ RAG 知识库

- 支持 **PDF / DOCX / TXT / Markdown / CSV / Excel** 自动解析
- 递归语义分块（按段落 → 换行 → 中英文标点优先级），保留页码溯源
- FAISS 向量库本地持久化，支持整文档删除（筛选重建）
- 相似度过滤阈值，低质量命中不注入上下文

### 4️⃣ SSE 流式响应

统一事件协议：`data: {"type": "...", "data": {...}}\n\n`

| type | 含义 |
|------|------|
| `thinking` | 会话建立，返回 conversation_id |
| `tool_start` | Agent 开始调用工具 |
| `tool_end` | 工具执行完成 |
| `token` | LLM 生成的增量文本 |
| `sources` | 检索命中的知识片段（引用来源） |
| `done` | 回复完成 |
| `error` | 出错 |

## 快速开始

### 1. 启动 Redis

```bash
docker compose up -d redis
```

> 没有 Docker 也可以：系统会自动降级为内存记忆（仅开发可用）。

### 2. 配置环境变量

```bash
cd backend
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

编辑 `.env`，填入 API Key：

```ini
LLM_API_KEY=your-api-key          # 必填：对话模型 Key
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=doubao-seed-2-1-pro-260915
EMBEDDING_MODEL=doubao-embedding-text-240715
EMBEDDING_DIM=2560
```

> 使用 DeepSeek/通义等其他网关时，修改 `LLM_BASE_URL` / `LLM_MODEL` / `EMBEDDING_*` 即可。

### 3. 启动后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

打开 http://localhost:8000/docs 查看 API 文档。

### 4. 启动前端

前后端分离部署，任选静态服务器：

```bash
cd frontend
python -m http.server 5173
# 或使用 live-server / nginx
```

打开 http://localhost:5173 即可使用。前端默认连接 `http://localhost:8000`，
如后端地址不同，在 `index.html` 前注入：

```html
<script>window.__API_BASE__ = "http://your-host:8000/api/v1";</script>
```

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat/stream` | SSE 流式对话（body: query, user_id, conversation_id?） |
| GET | `/api/v1/chat/conversations?user_id=` | 会话列表 |
| POST | `/api/v1/chat/conversations?user_id=` | 创建空会话 |
| GET | `/api/v1/chat/conversations/{id}/history?user_id=` | 会话历史（含摘要） |
| DELETE | `/api/v1/chat/conversations/{id}?user_id=` | 删除会话 |
| POST | `/api/v1/documents/upload` | 上传文档（multipart） |
| GET | `/api/v1/documents` | 知识库文档列表 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |
| GET | `/api/v1/health` | 健康检查 |

## 运行测试

```bash
cd backend
pytest
```

覆盖：对话隔离、摘要压缩触发/降级/并发锁、计算器安全（恶意输入拒绝）、
分块器边界、文档解析（TXT/CSV/XLSX/DOCX/PDF）、FAISS 增删查与持久化、API 集成。

## 目录结构

```
RagDemo/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口 + lifespan + 全局异常
│   │   ├── core/                # 配置 / 日志 / 业务异常
│   │   ├── api/v1/              # 路由层（chat / documents / health）
│   │   ├── agent/               # Agent 编排 + 三大工具 + 提示词
│   │   ├── memory/              # Redis 记忆 + 摘要算法 + 内存降级
│   │   ├── rag/                 # 文档解析 + 分块 + Embedding + FAISS
│   │   └── schemas/             # Pydantic 请求/响应模型
│   ├── tests/                   # 单元 + 集成测试
│   ├── requirements.txt
│   └── .env.example
├── frontend/                    # 独立静态前端（无构建依赖）
├── docker-compose.yml           # Redis
└── README.md
```

## 生产部署建议

- **多 worker**：`uvicorn app.main:app --workers 4`；FAISS 为本地文件存储，
  多 worker 场景建议迁移到 Milvus/Qdrant 等独立向量服务
- **反向代理**：nginx 需对 SSE 路径关闭缓冲（代码已带 `X-Accel-Buffering: no`）
- **Redis**：开启 AOF 持久化（compose 已配置），生产使用密码 + TLS
- **密钥管理**：生产环境用环境变量注入或 KMS，不落 .env 文件
