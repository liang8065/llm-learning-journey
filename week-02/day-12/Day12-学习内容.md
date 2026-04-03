# 📚 Day 12 — API 设计与项目架构

> **周期**：Day 12 / Week 2
> **目标**：掌握 RESTful API 设计，FastAPI 项目架构
> **预计用时**：1.5 小时

---

## Part 1：RESTful API 设计原则（30分钟）

| 原则 | 说明 | 示例 |
|------|------|------|
| 资源导向 | URL 表示资源不是动作 | `/api/chat` 不是 `/api/sendMessage` |
| HTTP 方法 | GET/POST/PUT/DELETE | 对应 CRUD |
| 状态码 | 正确返回 | 200/400/404/500 |
| 统一响应格式 | JSON 结构一致 | `{"code":0,"data":{}}` |

### 完整可运行代码

```python
"""
Day12 Part 1: RESTful LLM API 设计
依赖安装:
    pip install fastapi uvicorn pydantic
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time, uuid

app = FastAPI(title="LLM API", version="1.0.0")

class ChatMessage(BaseModel):
    role: str = Field(..., description="system/user/assistant")
    content: str

class ChatRequest(BaseModel):
    model: str = Field(default="qwen-plus")
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2000)

class APIResponse(BaseModel):
    code: int = 0
    data: Optional[dict] = None
    msg: str = "ok"

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    """聊天接口（兼容 OpenAI 格式）"""
    r = {
        "id": f"chat-{uuid.uuid4().hex[:8]}",
        "model": req.model,
        "content": "这是 API 演示响应",
        "usage": {"total_tokens": 100},
    }
    return APIResponse(data=r)

@app.get("/v1/models")
async def list_models():
    return APIResponse(data={"models": [{"id": "qwen-plus"}]})

@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}

if __name__ == "__main__":
    print("启动: uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
```

---

## Part 2：项目架构设计（30分钟）

```python
"""
Day12 Part 2: LLM 项目模块化架构
"""

# 推荐架构
print("""
llm-app/
├── app/
│   ├── main.py           # FastAPI 入口
│   ├── config.py         # 配置管理
│   ├── models/           # 数据模型
│   ├── services/         # 业务逻辑
│   │   ├── llm_service.py
│   │   ├── rag_service.py
│   │   └── agent_service.py
│   └── routers/          # API 路由
├── tests/
├── prompts/              # Prompt 模板
├── .env
└── requirements.txt
""")

# 配置管理
from pydantic import BaseSettings

class Settings(BaseSettings):
    llm_model: str = "qwen-plus"
    dashscope_api_key: str = ""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"

print("配置: 使用 pydantic BaseSettings + .env")
```

---

## Part 3：实战 — 简易 LLM API（15分钟）

```python
"""
Day12 Part 3: 简易 LLM API 服务
"""

# 实际调用 LLM 的 API 示例
import os
from fastapi import FastAPI
from pydantic import BaseModel

app2 = FastAPI()

class Query(BaseModel):
    question: str

@app2.post("/ask")
async def ask(q: Query):
    from langchain_community.chat_models import ChatTongyi
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatTongyi(
        model="qwen-plus",
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
    )
    answer = llm.invoke(q.question)
    return {"answer": StrOutputParser().invoke(answer) if hasattr(answer, 'content') else str(answer)}

print("启动: uvicorn api:app2 --reload")
print("测试: curl -X POST http://localhost:8000/ask -d '{\"question\":\"你好\"}'')
```
