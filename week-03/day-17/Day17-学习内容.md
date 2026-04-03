# 📚 Day 17 — 流式输出与实时交互

> **周期**：Day 17 / Week 3
> **目标**：掌握流式响应原理，构建实时对话应用
> **预计用时**：1.5 小时

---

## Part 1：流式响应原理（30分钟）

### 为什么需要流式输出？

**非流式**：等待 10 秒 → 一次性收到全部
**流式**：1 秒看到第一个字 → 逐字接收

```python
"""
Day17 Part 1: 流式输出
依赖安装:
    pip install langchain langchain-community langchain-core dashscope
"""

import os, time
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

print("=== 方法1: LLM 流式 ===
")
for chunk in llm.stream("用 100 字介绍 Python"):
    print(chunk.content, end="", flush=True)
    time.sleep(0.03)
print("
")

# LCEL 流式
print("=== 方法2: LCEL 流式 ===
")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是诗歌助手。"),
    ("human", "{topic}")
])
chain = prompt | llm | StrOutputParser()
for chunk in chain.stream({"topic": "写一首关于春天的诗"}):
    print(chunk, end="", flush=True)
    time.sleep(0.03)
print("\n")
```

---

## Part 2：WebSocket/SSE 实时对话（30分钟）

```python
"""
Day17 Part 2: 实时对话服务
"""

print("""
# SSE 服务端 (server.py)
from fastapi.responses import StreamingResponse
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/chat/stream")
def stream_chat(question: str):
    def generate():
        for chunk in chain.stream({"question": question}):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

# 启动: uvicorn server:app --reload
""")
```

---

## Part 3：实战 — 完整流式对话流程（15分钟）

```python
"""
Day17 Part 3: 流式对话
环境变量:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
import time
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是实时助手。回答简洁。"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

def chat_stream(question, history=None):
    for chunk in chain.stream({"question": question, "history": history or []}):
        yield chunk

print("=== 流式对话测试 ===\n")
for chunk in chat_stream("用一句话解释什么是 RAG"):
    print(chunk, end="", flush=True)
    time.sleep(0.02)
```
