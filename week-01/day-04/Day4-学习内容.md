# 📚 Day 4 — RAG 实战深化 + 向量数据库

> **周期**：Day 4 / Week 1
> **目标**：深入掌握向量数据库，多格式文档加载，构建知识库问答系统
> **预计用时**：1.5 小时

---

## Part 1：向量数据库深入对比（30分钟）

### 三大向量数据库对比

| 特性 | FAISS | ChromaDB | Milvus |
|------|-------|----------|--------|
| 厂商 | Meta | Chroma | Zilliz |
| 安装难度 | 简单 | 极简 | 中等 |
| 适用场景 | 中小规模、开发 | 开发/原型 | 企业级大规模 |
| 相似度搜索 | 支持 | 支持 | 支持 |
| 元数据过滤 | 不支持 | 支持 | 支持 |
| 持久化 | save_local / load_local | 自动 | 需配置 |

### 完整可运行代码

```python
"""
Day4 Part 1: 向量数据库深入（FAISS + 向量相似度）
依赖安装:
    pip install langchain langchain-community langchain-core dashscope faiss-cpu
"""

import os
import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

llm = None  # Part 1 不需要 LLM
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

texts = [
    "Python 是一门由 Guido van Rossum 于 1991 年发布的通用编程语言。",
    "Django 是 Python 最流行的 Web 框架，MTV 架构，内置 ORM、认证、管理后台。",
    "Flask 是轻量级微框架，核心精简，通过扩展支持数据库和 API。",
    "FastAPI 是现代高性能 Python Web 框架，基于类型提示和异步框架。",
    "Pydantic 是 Python 数据验证库，FastAPI 用它解析请求和生成 JSON Schema。",
    "Python GIL 限制多线程并行效率，可用多进程或异步编程绕过。",
    "pip 是包管理工具，venv 创建虚拟环境，poetry 是现代方案。",
]

# FAISS：创建、保存、加载
print("=== FAISS ===")
faiss_db = FAISS.from_texts(texts, embeddings)
faiss_db.save_local("faiss_index")
print("已保存")

loaded = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
docs = loaded.similarity_search("Python Web 框架", k=2)
for d in docs:
    print(f"  - {d.page_content}")

# 向量相似度
v1 = embeddings.embed_query("Python Web 框架")
v2 = embeddings.embed_query("FastAPI Django Flask")
v3 = embeddings.embed_query("今天天气不错")

def cos_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("\n=== 相似度 ===")
print(f"  Web框架 vs FastAPI/Django: {cos_sim(v1, v2):.4f}")
print(f"  Web框架 vs 天气: {cos_sim(v1, v3):.4f}")
```

---

## Part 2：多格式文档分割（30分钟）

```python
"""
Day4 Part 2: 文档分割策略
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text = """Python 是一门高级通用编程语言，由 Guido van Rossum 于 1991 年发布。

Python 的设计哲学强调代码可读性。支持编程范式：
- 面向对象（OOP）
- 过程式编程
- 函数式编程

Python 应用领域：
1. Web 开发：Django、Flask、FastAPI
2. 数据科学：NumPy、Pandas
3. AI/ML：TensorFlow、PyTorch
4. 自动化运维：Ansible
"""

# 方法1：按字符分割
char_split = CharacterTextSplitter(chunk_size=80, chunk_overlap=10, separator="\n")
c1 = char_split.create_documents([text])
print(f"字符分割: {len(c1)} 块")
for i, c in enumerate(c1):
    print(f"  {i+1}: {c.page_content[:60]}...")

# 方法2：递归分割（推荐）
rec_split = RecursiveCharacterTextSplitter(
    chunk_size=120, chunk_overlap=30,
    separators=["\n\n", "\n", "。", "，", " "]
)
c2 = rec_split.create_documents([text])
print(f"\n递归分割: {len(c2)} 块")
for i, c in enumerate(c2):
    print(f"  {i+1}: {c.page_content[:60]}...")

# 结构化文档
docs = [
    Document(page_content=text, metadata={"source": "docs", "type": "introduction"}),
]
print(f"\n结构化文档: {docs[0].metadata}")
```

---

## Part 3：知识库问答系统（30分钟）

```python
"""
Day4 Part 3: 知识库问答系统
环境变量:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

knowledge = """
LangChain 是构建 LLM 应用的主要框架。1.0+ 使用 LCEL 语法。

核心模块：
- Prompts: 提示词模板
- LLMs: 模型调用
- Chains: 链式组合
- Agents: 推理和工具调用
- Memory: 对话历史
- Retrieval: 检索增强（RAG）

RAG 流程：
1. 文档加载
2. 文本分割（RecursiveCharacterTextSplitter）
3. Embedding（DashScope）
4. 向量存储（FAISS/ChromaDB/Milvus）
5. 检索
6. Prompt 拼装 + LLM 生成

Agent = 推理引擎 + 工具 + Prompt
ReAct: Thought -> Action -> Observation
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
docs = splitter.create_documents([knowledge])
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(k=2)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 LangChain 专家。基于以下知识库回答问题：\n{context}"),
    ("human", "问题：{question}")
])

rag_chain = (
    {
        "context": retriever | RunnableLambda(lambda ds: "\n---\n".join([d.page_content for d in ds])),
        "question": lambda x: x,
    }
    | prompt | llm | StrOutputParser()
)

print("=== 知识库问答 ===")
for q in ["RAG 流程是什么？", "Agent 怎么工作？", "LangChain 核心模块有哪些？"]:
    print(f"\nQ: {q}")
    print(f"A: {rag_chain.invoke(q)}")
    print("-" * 40)
```