# 📚 Day 3 — LangChain 1.0+ LCEL 深度实践 + RAG 基础

> **周期**：Day 3 / Week 1 / 阶段一：基础奠基期
> **目标**：掌握 LCEL 表达式链，理解 RAG 原理，构建第一个 RAG 问答系统
> **预计用时**：1.5 小时
> **LangChain 版本**：1.0+

---

## Part 1：LangChain 1.0+ LCEL 深度实践（30分钟）

### 原理：什么是 LCEL？

LCEL（LangChain Expression Language）是 LangChain 1.0+ 的核心语法，用管道符 `|` 把组件串起来：

```
Prompt | LLM | OutputParser
```

相比旧版本的 `LLMChain`、`run()` 等方法，LCEL 的优势：
1. **声明式**：代码即流程，一眼看清数据流向
2. **可组合**：任意组件都能串联，像搭积木
3. **自动优化**：LCEL 自动处理异步、批处理、流式输出
4. **内置日志追踪**：自动记录每个步骤的输入输出

### LCEL 三大核心组件

| 组件 | 说明 | 示例 |
|------|------|------|
| `Runnable` | 所有组件的基类 | Prompt、LLM、OutputParser 都是 Runnable |
| `|` 操作符 | 链式组合 | `prompt | llm` |
| `invoke()` | 执行链 | `chain.invoke({"question": "..."})` |

### 完整可运行代码

```python
"""
Day3 Part 1: LCEL 深度实践
===============================================
依赖安装:
    pip install langchain langchain-community langchain-core dashscope

环境变量:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# ============================
# 1. 初始化 LLM
# ============================
llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# ============================
# 2. 基础 LCEL 链：Prompt -> LLM -> OutputParser
# ============================
# 创建 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，负责把中文翻译成英文。只输出翻译结果。"),
    ("human", "{text}")
])

# LCEL 链：使用 | 运算符串联
translate_chain = prompt | llm | StrOutputParser()

# 执行
result = translate_chain.invoke({"text": "今天天气很好"})
print(f"翻译结果: {result}")

# ============================
# 3. RunnableParallel：并行处理多个输入
# ============================
# 创建多个并行任务
parallel_chain = RunnableParallel({
    "english": (
        ChatPromptTemplate.from_template("Translate to English: {text}")
        | llm
        | StrOutputParser()
    ),
    "french": (
        ChatPromptTemplate.from_template("Translate to French: {text}")
        | llm
        | StrOutputParser()
    ),
    "japanese": (
        ChatPromptTemplate.from_template("Translate to Japanese: {text}")
        | llm
        | StrOutputParser()
    ),
})

print("\n=== 并行翻译 ===")
results = parallel_chain.invoke({"text": "你好，世界"})
for lang, translation in results.items():
    print(f"  {lang}: {translation.strip()}")

# ============================
# 4. RunnableLambda：自定义处理函数
# ============================
def word_count(text):
    """统计字数"""
    return len(text)

def to_upper(text):
    """转大写"""
    return text.strip().upper()

# 链中加入自定义函数
upper_chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(to_upper)
    | RunnableLambda(lambda s: f"{s} (字数: {word_count(s)})")
)

print("\n=== 自定义处理 ===")
result = upper_chain.invoke({"text": "学习 LangChain"})
print(f"结果: {result}")

# ============================
# 5. RunnableLambda 条件分支（RunnableBranch）
# ============================
from langchain_core.runnables import RunnableBranch

def is_chinese(text):
    """判断是否是中文"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

branch_chain = RunnableBranch(
    (lambda x: is_chinese(x.get("text", "")),  # 中文 -> 翻译
     ChatPromptTemplate.from_template("翻译成英文: {text}") | llm | StrOutputParser()),
    (lambda x: not is_chinese(x.get("text", "")),  # 英文 -> 翻译
     ChatPromptTemplate.from_template("翻译成中文: {text}") | llm | StrOutputParser()),
    # 默认
    llm,
)

print("\n=== 条件分支 ===")
r1 = branch_chain.invoke({"text": "你好"})
print(f"输入: 你好 -> 输出: {r1.strip() if hasattr(r1, 'strip') else r1}")

r2 = branch_chain.invoke({"text": "Hello"})
print(f"输入: Hello -> 输出: {r2.strip() if hasattr(r2, 'strip') else r2}")

# ============================
# 6. 链的链式组合（嵌套链）
# ============================
# 第一步：总结
summarize_prompt = ChatPromptTemplate.from_template("用一句话总结: {text}")
summarize_chain = summarize_prompt | llm | StrOutputParser()

# 第二步：翻译
translate_prompt = ChatPromptTemplate.from_template("翻译成英文: {summary}")
translate_chain2 = translate_prompt | llm | StrOutputParser()

# 组合：先总结再翻译
pipeline = {
    "summary": summarize_chain,
} | RunnableLambda(
    lambda x: translate_chain2.invoke({"summary": x["summary"]})
)

print("\n=== 链的组合 ===")
final = pipeline.invoke({"text": "Python 是一门编程语言，由 Guido van Rossum 创建，广泛应用于数据分析、AI、Web开发等领域。"})
print(f"总结后翻译: {final}")
```

**检查点**：
- [ ] 代码能直接运行
- [ ] 理解 `|` 操作符的组合方式
- [ ] 能解释 RunnableParallel、RunnableLambda、RunnableBranch 的区别

---

## Part 2：RAG 基础（30分钟）

### 原理：什么是 RAG？

RAG（Retrieval-Augmented Generation）= 检索增强生成。解决大模型"幻觉"和"知识过时"问题。

**核心流程**（7步）：
1. 文档加载（Loader）→ 2. 文本分割（Splitter）→ 3. 向量化（Embedding）→ 4. 存入向量库（VectorStore）→ 5. 检索（Retriever）→ 6. 拼装 Prompt → 7. LLM 生成回答

### 完整可运行代码

```python
"""
Day3 Part 2: RAG 基础原理与实现
===============================================
依赖安装:
    pip install langchain langchain-community langchain-core dashscope faiss-cpu

环境变量:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# ============================
# 1. 初始化
# ============================
llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# ============================
# 2. 第1步：文档加载（模拟，实际可从文件加载）
# ============================
raw_docs = [
    Document(page_content="Python 是一门由 Guido van Rossum 于1991年发布的通用编程语言，以简洁易读的语法著称。"),
    Document(page_content="Django 是 Python 最流行的 Web 框架，采用 MTV 架构，内置 ORM、表单处理、认证系统。"),
    Document(page_content="Flask 是轻量级微框架，核心精简，通过扩展支持数据库、表单验证等功能。"),
    Document(page_content="FastAPI 是现代高性能 Python Web 框架，支持类型提示和异步编程，自动生成 API 文档。"),
    Document(page_content="pip 是 Python 的包管理器，用于安装和管理第三方库。venv 用于创建虚拟环境。"),
]
print(f"加载了 {len(raw_docs)} 篇文档")

# ============================
# 3. 第2步：文本分割
# ============================
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separator=""
)
chunks = splitter.split_documents(raw_docs)
print(f"分割后得到 {len(chunks)} 个 chunk")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {chunk.page_content[:50]}...")

# ============================
# 4. 第3+4步：向量化 + 存入向量库
# ============================
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"\n向量库已创建，包含 {vectorstore.index.ntotal} 条向量")

# ============================
# 5. 第5步：检索
# ============================
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
query = "Python 有哪些 Web 框架？"
docs = retriever.invoke(query)
print(f"\n检索问题: {query}")
print(f"检索到 {len(docs)} 篇相关文档:")
for i, doc in enumerate(docs):
    print(f"  [{i+1}] {doc.page_content}")

# ============================
# 6. 第6+7步：拼装 Prompt + LLM 生成
# ============================
def format_context(docs):
    """格式化检索结果为上下文"""
    return "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是技术问答助手。基于以下文档回答问题：\n{context}"),
    ("human", "问题：{question}")
])

# 构建 RAG 链
rag_chain = (
    {
        "context": retriever | RunnableLambda(lambda docs: format_context(docs)),
        "question": lambda x: x,
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 测试
print("\n" + "=" * 50)
print("RAG 测试:")
answer = rag_chain.invoke("Python 有哪些 Web 框架？各自特点是什么？")
print(f"问题: Python 有哪些 Web 框架？")
print(f"回答: {answer}")
```

**检查点**：
- [ ] 能画出 RAG 的 7 步流程图
- [ ] 代码能运行并返回合理答案
- [ ] 理解为什么需要文本分割（chunking）

---

## Part 3：实战 RAG 问答系统（20分钟）

### 完整可运行代码

```python
"""
Day3 Part 3: 实战 RAG 问答系统
===============================================
环境变量:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 1. 初始化
llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 2. 构建知识库：LLM 学习路线图
docs_text = """
LLM 应用开发学习路线：

第一阶段（基础）：环境搭建、Python 基础、Prompt Engineering、API 调用。

第二阶段（框架）：LangChain 使用、RAG 原理与实践、向量数据库（FAISS/ChromaDB）。

第三阶段（Agent）：Agent 概念、ReAct 模式、工具开发、工作流编排。

第四阶段（部署）：微调基础（LoRA）、部署方案（vLLM/Ollama）、性能优化。

核心技术栈：Python、LangChain、Qwen/GPT API、FAISS、FastAPI、Docker。

推荐资源：LangChain 官方文档、arXiv 论文、HuggingFace 课程、GitHub 开源项目。
"""

# 按段落分割
docs = [Document(page_content=d.strip()) for d in docs_text.strip().split("\n\n") if d.strip()]
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(k=2)

# 3. RAG 链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 LLM 学习顾问。基于以下信息回答问题：\n{context}"),
    ("human", "问题：{question}")
])

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])),
        "question": lambda x: x,
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 对话界面
print("=" * 50)
print("LLM 学习顾问 RAG 系统")
print("输入 'quit' 退出")
print("=" * 50)

# 测试问题
test_questions = [
    "第三阶段学什么？",
    "推荐哪些学习资源？",
    "核心技术栈有哪些？",
]

for q in test_questions:
    print(f"\nQ: {q}")
    print(f"A: {rag_chain.invoke(q)}")
    print("-" * 50)
```

**检查点**：
- [ ] 能成功对话
- [ ] 答案是基于知识库内容，不是瞎编的
- [ ] 理解 retriever 在 RAG 中的作用

---

## 🔗 参考资料

- LCEL 官方文档: https://python.langchain.com/docs/expression_language/
- RAG 教程: https://python.langchain.com/docs/tutorials/rag/
- FAISS 文档: https://github.com/facebookresearch/faiss
