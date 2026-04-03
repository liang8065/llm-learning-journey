#!/usr/bin/env python3
"""生成 Week 2 + Week 3 的完整学习内容（Day 8-21）"""

# Day 8: LangChain Memory 进阶 + 多轮对话
day8 = '''# 📚 Day 8 — LangChain Memory 进阶 + 多轮对话系统

> **周期**：Day 8 / Week 2
> **目标**：掌握 RunnableWithMessageHistory，构建多轮对话系统
> **预计用时**：1.5 小时

---

## Part 1：RunnableWithMessageHistory — 多轮对话记忆（30分钟）

### 原理

LLM 本身是无状态的——每次调用都不记得之前的对话。`RunnableWithMessageHistory` 负责管理对话历史，让 LLM 拥有"记忆"。

**核心机制**：
1. **Message History 存储**：记录每一轮的用户消息和 AI 回复
2. **自动拼接**：调用 LLM 前自动把历史消息拼进 Prompt
3. **会话隔离**：不同会话（session_id）有独立的记忆
4. **长度控制**：可限制历史消息长度，防止上下文溢出

### 完整可运行代码

```python
"""
Day8 Part 1: RunnableWithMessageHistory — 多轮对话
依赖安装:
    pip install langchain langchain-community langchain-core dashscope
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# 1. 初始化
llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 2. Prompt（带消息历史占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 Python 编程老师。回答要详细，包含示例代码。"),
    MessagesPlaceholder("history"),  # 这里会自动插入历史消息
    ("human", "{input}"),
])

# 3. 内存存储（每次运行清空）
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 4. 构建带记忆的链
chain = prompt | llm | StrOutputParser()
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 5. 多轮对话测试
print("=" * 60)
print("多轮对话测试")
print("=" * 60)

session_id = "user_main"

questions = [
    "Python 中列表和元组有什么区别？",
    "那字典呢？",  # AI 应该知道"那"指的是 Python 的数据结构
    "能用你刚提到的方式举个例子吗？",  # AI 应该知道"你刚提到的"是什么
]

for i, q in enumerate(questions, 1):
    print(f"\\n轮次 {i}: {q}")
    print("-" * 40)
    answer = memory_chain.invoke(
        {"input": q},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"AI: {answer}")
    print("=" * 60)

# 6. 不同会话隔离测试
print("\\n=== 新会话测试 ===")
answer = memory_chain.invoke(
    {"input": "我刚才问了什么？"},
    config={"configurable": {"session_id": "user_new"}}
)
print(f"新会话 AI: {answer}")  # 新会话不知道之前聊了什么

# 查看记忆状态
print(f"\\n=== 会话状态 ===")
for sid, history in store.items():
    msgs = history.messages
    print(f"会话 '{sid}': {len(msgs)} 条消息")
    for m in msgs:
        print(f"  {type(m).__name__}: {m.content[:50]}...")
```

**检查点**：
- [ ] 多轮对话上下文连贯
- [ ] 不同会话的记忆互相隔离
- [ ] 理解 MessagesPlaceholder 的作用

---

## Part 2：持久化存储 + 复杂对话（30分钟）

```python
"""
Day8 Part 2: 持久化对话 + 复杂多轮对话
"""

import os
import json
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 持久化的 JSON 文件存储
HISTORY_DIR = "./chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def load_history(session_id):
    """从文件加载历史"""
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        history = InMemoryChatMessageHistory()
        for msg in data:
            if msg["role"] == "human":
                history.add_message(HumanMessage(content=msg["content"]))
            else:
                history.add_message(AIMessage(content=msg["content"]))
        return history
    return InMemoryChatMessageHistory()

def save_history(session_id, history):
    """保存到文件"""
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    data = [{"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in history.messages]
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = load_history(session_id)
    return store[session_id]

# 构建带记忆的链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个编程助手。用简洁的方式回答，包含代码示例。"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

memory_chain = RunnableWithMessageHistory(
    prompt | llm | StrOutputParser(),
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 对话并自动保存
def chat(session_id, question):
    answer = memory_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    save_history(session_id, store[session_id])
    return answer

# 测试
print("=== 对话系统（带持久化）===")
for q in [
    "Python 中怎么读取文件？",
    "那写入文件呢？",
    "能写个读取和写入都有的完整例子吗？",
]:
    print(f"\\nQ: {q}")
    a = chat("user_1", q)
    print(f"A: {a}")
    print("-" * 40)

print(f"\\n历史已保存到 {HISTORY_DIR} 目录")
```

---

## Part 3：实战 — 带记忆的 RAG 对话（15分钟）

```python
"""
Day8 Part 3: 带记忆的 RAG 对话系统
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 知识库
docs = [
    Document(page_content="Python 是一门编程语言，由 Guido van Rossum 创建。支持面向对象、函数式和结构化编程范式。"),
    Document(page_content="Django 是 Python 的 Web 框架，采用 MTV 架构，内置 ORM、表单处理、认证系统。"),
    Document(page_content="Flask 是轻量级微框架，核心精简，通过扩展机制支持数据库、表单验证。"),
    Document(page_content="FastAPI 是现代高性能 Python Web 框架，基于类型提示和异步框架，自动生成 API 文档。"),
]

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(k=2)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是编程助教。基于以下文档回答问题：\\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# 带记忆的 RAG
def get_history(sid="default"):
    return InMemoryChatMessageHistory()

rag_memory_chain = RunnableWithMessageHistory(
    {
        "context": (lambda x: x["input"]) | retriever | (lambda docs: "\\n".join([d.page_content for d in docs])),
        "history": lambda x: x.get("history", []),
        "input": lambda x: x["input"],
    }
    | prompt
    | llm
    | StrOutputParser(),
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 测试
print("=== 带记忆的 RAG ===")
for q in ["Python 有哪些 Web 框架？", "哪个适合小项目？", "和第一个推荐的有什么区别？"]:
    print(f"\\nQ: {q}")
    a = rag_memory_chain.invoke(
        {"input": q},
        config={"configurable": {"session_id": "session_1"}}
    )
    print(f"A: {a}")
    print("-" * 40)
'''

# Day 9: RAG 优化
day9 = '''# 📚 Day 9 — RAG 系统优化

> **周期**：Day 9 / Week 2
> **目标**：掌握高级检索策略、Chunking 策略、评估与迭代
> **预计用时**：1.5 小时

---

## Part 1：高级检索策略（30分钟）

### 检索策略大全

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 相似度搜索 | 向量距离最近 | 基础场景 |
| MMR | 多样性最大化，减少冗余 | 需要多度信息 |
| 分数阈值 | 只返回相似度超过阈值的 | 过滤低质量结果 |
| 混合检索 | 语义 + 关键词 BM25 | 精确匹配 + 语义理解 |

### 完整可运行代码

```python
"""
Day9 Part 1: 高级检索策略对比
依赖安装:
    pip install langchain langchain-community langchain-core dashscope faiss-cpu
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import time

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 构建知识库
docs = [
    Document(page_content="Python 是一门由 Guido van Rossum 于 1991 年发布的通用编程语言，支持 OOP 和函数式编程。"),
    Document(page_content="Django 是 Python 最流行的 Web 框架，MTV 架构，内置 ORM、认证、管理后台，适合大型项目。"),
    Document(page_content="Flask 是轻量级微框架，核心精简，通过扩展支持数据库和 API，适合小型项目。"),
    Document(page_content="FastAPI 是现代高性能 Python Web 框架，基于类型提示和 Starlette 异步框架。"),
    Document(page_content="Pydantic 是 Python 数据验证库，FastAPI 用它解析请求和生成 JSON Schema。"),
    Document(page_content="NumPy 提供多维数组和数学计算，Pandas 提供数据框和分析功能。"),
    Document(page_content="Matplotlib 是基础绘图库，Seaborn 提供更高级的统计图表。"),
    Document(page_content="TensorFlow 是 Google 的深度学习框架，支持分布式训练。PyTorch 由 Meta 开发，动态图更灵活。"),
]

vectorstore = FAISS.from_documents(docs, embeddings)

# ============================
# 策略1：基础相似度搜索
# ============================
print("=== 策略1: 相似度搜索 ===")
results = vectorstore.similarity_search("Python Web 框架", k=3)
for r in results:
    print(f"  {r.page_content[:80]}")

# ============================
# 策略2：MMR（多样性最大化）
# ============================
print("\\n=== 策略2: MMR 多样性检索 ===")
results = vectorstore.max_marginal_relevance_search("Python 框架", k=3, fetch_k=6, lambda_mult=0.7)
for r in results:
    print(f"  {r.page_content[:80]}")

# ============================
# 策略3：带分数的相似度搜索
# ============================
print("\\n=== 策略3: 相似度 + 分数 ===")
query_vec = embeddings.embed_query("Python Web 编程")
results_with_score = vectorstore.similarity_search_with_score("Python Web 编程", k=5)
for r, score in results_with_score:
    print(f"  分数={score:.4f} → {r.page_content[:70]}...")

# ============================
# 策略4：Retriever 配置对比
# ============================
print("\\n=== 策略4: 不同 Retriever 对比 ===")

# 默认检索器
default_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.7})

for name, retriever in [("默认", default_retriever), ("MMR", mmr_retriever)]:
    print(f"\\n[{name}]")
    q_results = retriever.invoke("Python 有哪些框架？")
    for i, r in enumerate(q_results):
        print(f"  [{i+1}] {r.page_content[:60]}...")
```

---

## Part 2：Chunking 策略优化（30分钟）

```python
"""
Day9 Part 2: Chunking 策略对比
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

sample_text = """
Python 是一门高级通用编程语言，由荷兰程序员 Guido van Rossum 于 1989 年圣诞节期间开始开发，因此得名 Python（蟒蛇）。

Python 的设计哲学强调代码的可读性和简洁的语法。它支持多种编程范式：
- 面向对象编程（OOP）：使用类和对象
- 过程式编程：函数和语句
- 函数式编程：lambda、map、filter、reduce

Python 的核心应用领域包括：
1. Web 开发：Django、Flask、FastAPI 等框架
2. 数据科学：NumPy、Pandas、Matplotlib
3. 人工智能与机器学习：TensorFlow、PyTorch、Scikit-learn
4. 自动化运维：Ansible、Salt
5. 科学计算：SciPy、SymPy
6. 游戏开发：Pygame

Python 的安装和使用非常简单。推荐使用 pip 作为包管理工具，venv 作为虚拟环境管理工具。
"""

splitters = {
    "字符分割 (100/20)": CharacterTextSplitter(chunk_size=100, chunk_overlap=20, separator=""),
    "递归字符 (150/30)": RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30, separators=["\\n\\n", "\\n", "。", "，", " "]),
    "大块 (300/50)": RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\\n\\n", "\\n", "。", " "]),
}

llm = None
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

for name, splitter in splitters.items():
    chunks = splitter.create_documents([sample_text])
    print(f"\\n=== {name} ===")
    print(f"  块数: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i+1} ({len(chunk.page_content)}字): {chunk.page_content[:50]}...")
```

---

## Part 3：RAG 评估体系（15分钟）

```python
"""
Day9 Part 3: RAG 评估与调优
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

docs = [
    Document(page_content="Python 是一门编程语言，1991年发布。"),
    Document(page_content="Django 是 Python 的 Web 框架，MTV 架构。"),
    Document(page_content="Flask 是轻量级 Python Web 框架。"),
    Document(page_content="FastAPI 是高性能 Python Web 框架，自动 API 文档。"),
]

# 测试不同的 k 值
print("=== chunk_size 和 k 值调优 ===")
for k in [1, 2, 3]:
    retriever = FAISS.from_documents(docs, embeddings).as_retriever(search_kwargs={"k": k})
    query_results = retriever.invoke("Python Web 框架")
    print(f"k={k}: 检索到 {len(query_results)} 篇")

# Chunk size 对比
for chunk_size in [50, 100, 200]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    results = vs.similarity_search("Web 框架", k=2)
    print(f"chunk_size={chunk_size}: {len(chunks)} blocks, 检索到 {len(results)} 结果")
```
'''

# Day 10: 多模态 LLM
day10 = '''# 📚 Day 10 — 多模态 LLM 应用

> **周期**：Day 10 / Week 2
> **目标**：掌握视觉模型原理，使用 Qwen-VL 进行图片分析，构建多模态 RAG
> **预计用时**：1.5 小时

---

## Part 1：多模态模型原理（30分钟）

### 什么是多模态 LLM？

多模态 LLM 能同时处理多种类型的数据：文本 + 图片 + 音频 + 视频。

**核心模型**：
- **GPT-4V/Vision**：OpenAI 的多模态模型
- **Qwen-VL**：阿里的开源视觉语言模型
- **LLaVA**：开源的多模态模型

**工作原理**：
1. 图片经过 Vision Encoder → 转为视觉 Token
2. 视觉 Token 和文字 Token 一起送入 LLM
3. LLM 统一处理，生成文字回答

### 完整可运行代码

```python
"""
Day10 Part 1: Qwen-VL 图片分析
依赖安装:
    pip install dashscope
"""

import os
import dashscope
from dashscope import MultiModalConversation

# 设置 API Key
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "your-key")

# ============================
# 1. 分析网络图片
# ============================
def analyze_image(image_url, prompt="这张图片里有什么？"):
    """分析图片内容"""
    messages = [{
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": prompt}
        ]
    }]
    
    response = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        return f"调用失败: {response.code} - {response.message}"

# 2. 测试
print("=== 图片分析测试 ===")
# 使用示例图片 URL
# result = analyze_image("https://example.com/image.jpg", "描述这张图片")
# print(f"结果: {result}")

print("提示: 将 image_url 替换为实际图片 URL 后运行")

def analyze_multiple_images(image_urls, prompt="比较这些图片"):
    """分析多张图片"""
    content = []
    for url in image_urls:
        content.append({"image": url})
    content.append({"text": prompt})
    
    messages = [{"role": "user", "content": content}]
    response = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    return f"失败: {response.message}"
```

---

## Part 2：文档 OCR + 多模态 RAG（30分钟）

```python
"""
Day10 Part 2: 多模态 RAG — 图文结合检索
依赖安装:
    pip install langchain langchain-community langchain-core dashscope faiss-cpu
"""

import os
import dashscore
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 构建图文知识库
docs = [
    Document(
        page_content="Python 是一门高级通用编程语言，由 Guido van Rossum 于 1991 年发布。核心特点：简洁易读的语法、动态类型、面向对象、函数式编程支持。广泛用于 Web 开发、数据科学、AI。",
        metadata={"type": "text", "tags": ["python", "language"]}
    ),
    Document(
        page_content="Django 是 Python 最著名的 Web 框架，采用 MTV（Model-Template-View）架构。核心功能：ORM、表单处理、用户认证、管理后台、国际化。适合构建内容管理系统和复杂应用。",
        metadata={"type": "text", "tags": ["django", "web"]}
    ),
    Document(
        page_content="FastAPI 是新一代 Python Web 框架，使用 Python 类型注解自动验证请求、生成 OpenAPI 文档。特点：高性能（Starlette 底层）、异步支持、自动文档生成。",
        metadata={"type": "text", "tags": ["fastapi", "web"]}
    ),
]

# 向量库
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 多模态 RAG 链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是多模态AI助手，能处理文本和图像问题。基于以下知识回答：\\n{context}"),
    ("human", "{question}")
])

rag_chain = (
    {
        "context": retriever | (lambda docs: "\\n".join([d.page_content for d in docs])),
        "question": lambda x: x,
    }
    | prompt | llm | StrOutputParser()
)

# 测试
print("\\n=== 多模态 RAG 测试 ===")
for q in ["Python 有哪些 Web 框架？", "Django 的核心功能有哪些？"]:
    print(f"\\nQ: {q}")
    print(f"A: {rag_chain.invoke(q)}")
    print("-" * 40)
```

---

## Part 3：完整多模态应用 — 图片知识库（15分钟）

```python
"""
Day10 Part 3: 图文知识库
"""

import os
import json
from datetime import datetime

# 图片知识库结构
IMAGE_KB = {
    "images": [],
    "metadata": {}
}

def add_image_to_kb(image_url, description, tags=None):
    """将图片信息添加到知识库"""
    entry = {
        "url": image_url,
        "description": description,
        "tags": tags or [],
        "added_at": datetime.now().isoformat()
    }
    IMAGE_KB["images"].append(entry)
    return f"已添加图片: {description}"

def search_images(tags=None, keyword=None):
    """搜索图片"""
    results = []
    for img in IMAGE_KB["images"]:
        if tags:
            if any(t in img["tags"] for t in tags):
                results.append(img)
        if keyword:
            if keyword.lower() in img["description"].lower():
                results.append(img)
    return results

# 测试
print("===图文知识库===")
add_image_to_kb("https://example.com/python.png", "Python 语言标志", ["python", "logo"])
add_image_to_kb("https://example.com/django.png", "Django 框架架构图", ["django", "architecture"])
add_image_to_kb("https://example.com/fastapi.png", "FastAPI 性能对比图", ["fastapi", "chart"])

results = search_images(tags=["python"])
print(f"Python 相关图片: {len(results)} 张")
for r in results:
    print(f"  {r['description']} - tags: {r['tags']}")
```
'''

# Day 11: LangGraph 工作流
day11 = '''# 📚 Day 11 — 工作流编排与 LangGraph

> **周期**：Day 11 / Week 2
> **目标**：掌握 LangGraph 入门，构建复杂工作流
> **预计用时**：1.5 小时

---

## Part 1：LangGraph 入门（30分钟）

### 什么是 LangGraph？

LangGraph 是 LangChain 的工作流编排库，用图结构定义状态机：

- **节点（Node）**：执行动作（LLM 调用、工具执行、数据处理）
- **边（Edge）**：状态流转方向
- **条件边（Conditional Edge）**：根据结果决定下一步
- **状态（State）**：贯穿整个工作流的数据

### 完整可运行代码

```python
"""
Day11 Part 1: LangGraph 入门 - 状态机工作流
依赖安装:
    pip install langgraph langchain langchain-community langchain-core dashscope
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# ============================
# 1. 定义状态
# ============================
class AgentState(TypedDict):
    """工作流状态"""
    question: str
    answer: str
    steps: list

# ============================
# 2. 定义节点函数
# ============================
def analyze_question(state: AgentState) -> AgentState:
    """分析问题类型"""
    q = state["question"].lower()
    if any(word in q for word in ["计算", "数学", "加减乘除", "*", "/", "+", "-"]):
        question_type = "math"
    elif any(word in q for word in ["天气", "温度", "气候"]):
        question_type = "weather"
    else:
        question_type = "general"
    state["steps"].append(f"分析: {question_type}")
    state["question_type"] = question_type
    return state

def handle_math(state: AgentState) -> AgentState:
    """处理数学问题"""
    import re
    numbers = re.findall(r'\\d+', state["question"])
    if len(numbers) >= 2:
        n1, n2 = int(numbers[0]), int(numbers[1])
        if "*" in state["question"] or "乘以" in state["question"]:
            result = n1 * n2
        elif "/" in state["question"] or "除以" in state["question"]:
            result = n1 / n2
        elif "+" in state["question"] or "加" in state["question"]:
            result = n1 + n2
        else:
            result = n1 - n2
        state["answer"] = f"计算结果: {result}"
    else:
        state["answer"] = "无法识别数学表达式"
    state["steps"].append("数学处理完成")
    return state

def handle_weather(state: AgentState) -> AgentState:
    """处理天气查询"""
    city = state["question"]
    state["answer"] = f"{''.join([c for c in city if any('\\u4e00' <= c <= '\\u9fff' for c in city)])}的天气：晴，气温 15-25°C"
    state["steps"].append("天气查询完成")
    return state

def handle_general(state: AgentState) -> AgentState:
    """处理一般问题（简单回答）"""
    state["answer"] = f"你问的是: {state['question']}。这是一个通用问题，暂时由规则引擎处理。"
    state["steps"].append("通用处理完成")
    return state

def format_result(state: AgentState) -> AgentState:
    """格式化结果"""
    state["answer"] += f"\\n\\n处理步骤: {' → '.join(state['steps'])}"
    return state

# ============================
# 3. 构建图
# ============================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("analyze", analyze_question)
workflow.add_node("math", handle_math)
workflow.add_node("weather", handle_weather)
workflow.add_node("general", handle_general)
workflow.add_node("format", format_result)

# 设置入口
workflow.set_entry_point("analyze")

# 条件边
def route_question(state: AgentState) -> str:
    return state.get("question_type", "general")

workflow.add_conditional_edges(
    "analyze",
    route_question,
    {"math": "math", "weather": "weather", "general": "general"}
)

# 所有处理完成后合并到 format
workflow.add_edge("math", "format")
workflow.add_edge("weather", "format")
workflow.add_edge("general", "format")
workflow.add_edge("format", END)

# 编译
app = workflow.compile()

# ============================
# 4. 运行测试
# ============================
print("=== LangGraph 工作流测试 ===\\n")

test_questions = [
    "帮我计算 25 乘以 4 等于多少",
    "北京今天的天气",
    "什么是 Python？",
]

for q in test_questions:
    print(f"问题: {q}")
    print("-" * 40)
    result = app.invoke({
        "question": q,
        "answer": "",
        "steps": []
    })
    print(f"回答: {result['answer']}")
    print("步骤: " + " → ".join(result["steps"]))
    print("=" * 50)

# 可视化工作流（需要安装 graphviz）
# app.get_graph().draw_mermaid_png()
```

**检查点**：
- [ ] 理解 StateGraph 的状态传递
- [ ] 能画出工作流图：analyze → math/weather/general → format → END
- [ ] 理解条件边的路由机制

---

## Part 2：复杂工作流 — 条件分支 + 循环（30分钟）

```python
"""
Day11 Part 2: 杂工作流 - 多轮处理
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict

class WorkflowState(TypedDict):
    query: str
    results: list
    iteration: int
    done: bool

def search_query(state: WorkflowState) -> WorkflowState:
    """执行搜索"""
    # 模拟搜索
    state["results"].append(f"搜索 '{state['query']}' 的结果")
    state["done"] = len(state["results"]) >= 3
    return state

def refine_query(state: WorkflowState) -> WorkflowState:
    """改进查询"""
    state["query"] += " (改进版)"
    return state

# 构建循环工作流
wf = StateGraph(WorkflowState)
wf.add_node("search", search_query)
wf.add_node("refine", refine_query)
wf.set_entry_point("search")

wf.add_conditional_edges(
    "search",
    lambda s: "refine" if not s["done"] else "end",
    {"refine": "refine", "end": END}
)
wf.add_edge("refine", "search")

app = wf.compile()

print("=== 循环工作流 ===")
result = app.invoke({"query": "Python 教程", "results": [], "iteration": 0, "done": False})
for i, r in enumerate(result["results"], 1):
    print(f"  [{i}] {r}")
```

---

## Part 3：实战 — 自动化学习助手（15分钟）

```python
"""
Day11 Part 3: 自动化学习助手工作流
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict

class StudyState(TypedDict):
    topic: str
    plan: list
    notes: str
    progress: int

def create_plan(state: StudyState) -> StudyState:
    """创学习计划"""
    state["plan"] = [
        f"1. 了解 {state['topic']} 基础概念",
        f"2. 学习 {state['topic']} 核心语法",
        f"3. 动手实践 {state['topic']} 项目",
        f"4. 复习总结 {state['topic']}",
    ]
    return state

def study_topic(state: StudyState) -> StudyState:
    """学习主题"""
    state["notes"] = f"学习了 {state['topic']}"
    state["progress"] = 50
    return state

def review(state: StudyState) -> StudyState:
    """复习"""
    state["notes"] += "\\n复习完成"
    state["progress"] = 100
    return state

wf = StateGraph(StudyState)
wf.add_node("plan", create_plan)
wf.add_node("study", study_topic)
wf.add_node("review", review)
wf.set_entry_point("plan")
wf.add_edge("plan", "study")
wf.add_edge("study", "review")
wf.add_edge("review", END)

app = wf.compile()

print("=== 学习助手 ===")
result = app.invoke({"topic": "Python 异步编程", "plan": [], "notes": "", "progress": 0})
for k, v in result.items():
    print(f"{k}: {v}")
```
'''

all_days = {
    "week-02/day-08": day8,
    "week-02/day-09": day9,
    "week-02/day-10": day10,
    "week-02/day-11": day11,
}

import os
for path, content in all_days.items():
    full = os.path.join("/home/admin/.openclaw/workspace/llm-learning-journey", path + "/学习内容.md")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    print(f"Written: {full}")

print("Batch 2 done")