# Day 3（2026-03-14）学习内容详细拆解

> **学习者**: 亮亮
> **日期**: 2026-03-14
> **主题**: LangChain 1.0+ 深入 + RAG 基础
> **总时长**: ~1.5小时（90分钟）
> **拆解方式**: 按任务拆解为具体步骤，每个步骤有明确的输入、操作、输出

---

## 📋 今日任务总览

| 序号 | 任务 | 时长 | 状态 |
|------|------|------|------|
| 1 | LangChain 1.0+ LCEL 深度实践 | 30分钟 | ⏳ 待完成 |
| 2 | RAG（检索增强生成）基础 | 30分钟 | ⏳ 待完成 |
| 3 | 实战：构建简单 RAG 问答系统 | 30分钟 | ⏳ 待完成 |
| 4 | GitHub 更新 & 学习笔记 | 15分钟 | ⏳ 待完成 |

---

## 🎯 Part 1: LangChain 1.0+ LCEL 深度实践（30分钟）

### 目标
掌握 LCEL（LangChain Expression Language）的核心组件：RunnableParallel、RunnableLambda、RunnableBranch

### 拆解步骤

#### Step 1.1: 理解 LCEL 基础（10分钟）
**输入**: LangChain 1.0+ 官方文档
**操作**:
- 阅读 LCEL 核心概念文档
- 理解 `|` 操作符的含义
- 理解 Runnable 协议
**输出**: 笔记要点（3-5条）
**检查点**: 能解释 LCEL 是什么，为什么使用它

#### Step 1.2: RunnableParallel 实践（10分钟）
**输入**: 示例代码模板
**操作**:
```python
from langchain_core.runnables import RunnableParallel
from langchain_community.chat_models import ChatTongyi

# 创建并行处理链
llm = ChatTongyi(model="qwen-plus", dashscope_api_key="your-key")

# 并行处理：同时生成多个结果
parallel_chain = RunnableParallel(
    summary=lambda x: f"总结：{x[:50]}...",
    analysis=lambda x: f"分析：{len(x)}字",
    sentiment=lambda x: "正面" if "好" in x else "负面"
)

result = parallel_chain.invoke("今天天气很好，心情愉快")
print(result)
# 输出: {'summary': '总结：今天天气很好...', 'analysis': '分析：8字', 'sentiment': '正面'}
```
**输出**: 能运行并理解并行处理的效果
**检查点**: 理解 RunnableParallel 如何同时处理多个任务

#### Step 1.3: RunnableLambda 实践（10分钟）
**输入**: 自定义函数
**操作**:
```python
from langchain_core.runnables import RunnableLambda

# 使用 Lambda 包装自定义函数
def preprocess(text: str) -> str:
    """预处理：去除空白，转换小写"""
    return text.strip().lower()

def postprocess(text: str) -> str:
    """后处理：添加格式"""
    return f"【结果】{text}"

# 创建 Lambda 链
pre_chain = RunnableLambda(preprocess)
post_chain = RunnableLambda(postprocess)

# 组合链
full_chain = pre_chain | llm | post_chain
result = full_chain.invoke("  Hello World  ")
print(result)
```
**输出**: 理解 RunnableLambda 的作用
**检查点**: 能将任意函数包装为 Runnable

### Part 1 完成检查
- [ ] 理解 LCEL 核心概念
- [ ] 能运行 RunnableParallel 示例
- [ ] 能运行 RunnableLambda 示例
- [ ] 能组合多个 Runnable

---

## 🎯 Part 2: RAG（检索增强生成）基础（30分钟）

### 目标
理解 RAG 原理、文档加载器、文本分割器、向量存储基础

### 拆解步骤

#### Step 2.1: RAG 原理学习（10分钟）
**输入**: RAG 概念文档/教程
**操作**:
- 理解 RAG 的定义和工作原理
- 理解检索（Retrieval）+ 生成（Generation）的两阶段流程
- 理解向量数据库的作用
**输出**: 笔记要点（RAG 流程图 + 关键概念）
**检查点**: 能解释 RAG 为什么需要向量数据库

#### Step 2.2: 文档加载器实践（10分钟）
**输入**: 示例文档（txt/pdf/markdown）
**操作**:
```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# 加载文本文档
loader = TextLoader("path/to/document.txt")
documents = loader.load()

# 加载 PDF
pdf_loader = PyPDFLoader("path/to/document.pdf")
pdf_docs = pdf_loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"第一个文档内容：{documents[0].page_content[:200]}...")
```
**输出**: 成功加载文档并查看内容
**检查点**: 能加载至少一种格式的文档

#### Step 2.3: 文本分割器实践（10分钟）
**输入**: 加载的文档
**操作**:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建文本分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 每个块的最大字符数
    chunk_overlap=20,    # 块之间的重叠字符数
    separators=["\n\n", "\n", " ", ""]  # 分割符优先级
)

# 分割文档
chunks = splitter.split_documents(documents)
print(f"分割成 {len(chunks)} 个块")
print(f"第一个块：{chunks[0].page_content[:100]}...")
```
**输出**: 文档被分割成多个 chunk
**检查点**: 理解 chunk_size 和 chunk_overlap 的作用

### Part 2 完成检查
- [ ] 理解 RAG 原理和流程
- [ ] 能加载文档（至少一种格式）
- [ ] 能分割文本为 chunk
- [ ] 理解 chunk_size 和 chunk_overlap

---

## 🎯 Part 3: 实战：构建简单 RAG 问答系统（30分钟）

### 目标
使用 LangChain 1.0+ 构建一个基于文档的问答系统

### 拆解步骤

#### Step 3.1: 准备文档和向量存储（10分钟）
**输入**: 示例文档
**操作**:
```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = TextLoader("学习笔记.txt")
docs = loader.load()

# 2. 分割文档
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# 3. 创建向量存储
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever()
```
**输出**: 向量数据库和检索器就绪
**检查点**: 能创建向量存储并检索

#### Step 3.2: 构建 RAG 链（10分钟）
**输入**: 向量存储 + LLM
**操作**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnableParallel

# 1. 初始化 LLM
llm = ChatTongyi(model="qwen-plus", dashscope_api_key="your-key")

# 2. 创建 Prompt 模板
prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：

上下文：
{context}

问题：{question}

回答：
""")

# 3. 构建 RAG 链（1.0+ LCEL 方式）
rag_chain = (
    RunnableParallel(
        {"context": retriever, "question": lambda x: x}
    )
    | prompt
    | llm
)
```
**输出**: 可运行的 RAG 链
**检查点**: 理解 RAG 链的结构

#### Step 3.3: 测试 RAG 系统（10分钟）
**输入**: 用户问题
**操作**:
```python
# 测试问答
question = "什么是 RAG？"
response = rag_chain.invoke(question)
print(f"问题: {question}")
print(f"回答: {response.content}")

# 查看检索到的上下文
docs = retriever.get_relevant_documents(question)
print(f"\n检索到 {len(docs)} 个相关文档")
for i, doc in enumerate(docs):
    print(f"文档 {i+1}: {doc.page_content[:100]}...")
```
**输出**: 问答结果 + 检索到的上下文
**检查点**: 能正确回答问题，并显示检索来源

### Part 3 完成检查
- [ ] 能创建向量存储
- [ ] 能构建 RAG 链
- [ ] 能测试问答并获得正确回答
- [ ] 能查看检索到的上下文

---

## 🎯 Part 4: GitHub 更新 & 学习笔记（15分钟）

### 目标
更新学习日志，推送细化学习计划到 GitHub

### 拆解步骤

#### Step 4.1: 整理学习笔记（5分钟）
**操作**:
- 记录今天学到的关键点
- 记录遇到的问题和解决方案
- 记录代码示例

#### Step 4.2: 更新学习日志（5分钟）
**操作**:
- 更新 `learning-log.md` 的 Day 3 部分
- 标记完成的任务
- 添加收获与反思

#### Step 4.3: 推送 GitHub（5分钟）
**操作**:
- 推送更新到 `liang8065/llm-learning-journey`
- 包括：REFINED-LEARNING-PLAN.md、learning-log.md、README.md

---

## 📊 今日产出清单

| 产出 | 说明 | 文件 |
|------|------|------|
| 1 | LCEL 实践笔记 | `week-01/day-03/LCEL-实践笔记.md` |
| 2 | RAG 基础笔记 | `week-01/day-03/RAG-基础笔记.md` |
| 3 | RAG 问答系统代码 | `week-01/day-03/rag-qa-system.py` |
| 4 | 学习日志更新 | `learning-log.md` (Day 3 部分) |
| 5 | GitHub 推送 | `REFINED-LEARNING-PLAN.md` 等 |

---

## 🔄 拆解任务本身说明

**拆解目的**: 将每个学习任务拆解为具体、可执行的步骤，明确输入、操作、输出和检查点。

**拆解原则**:
1. **具体性**: 每个步骤有明确的操作和输出
2. **可衡量**: 有检查点可以验证完成
3. **时间可控**: 每个步骤有预估时长
4. **代码导向**: 包含可运行的代码示例

**后续约定**:
- 每次推送学习计划时，同步完成该计划的详细拆解
- 拆解结果保存在 `week-NN/day-DD/DayDD-学习内容-详细拆解.md`
- 拆解任务本身也作为每日任务的一部分

---

*创建时间: 2026-03-14 10:43*
*拆解版本: v1.0*
