# 📚 Day 22 — 高级 RAG 技术（2026-04-03）

> **周期**：Day 22 / Week 4 / 阶段二：技能提升期
> **目标**：掌握 HyDE、Multi-query 检索、Self-RAG 等高级 RAG 技术
> **预计用时**：1.5 - 2 小时
> **LangChain 版本**：1.0+（全部使用 LCEL 表达式链）

---

## Part 1：HyDE 假设性文档嵌入 🎯

### 原理

HyDE（Hypothetical Document Embeddings）的核心思想：

1. **问题**：用户的自然语言问题和知识库里的专业文档，语义表达方式差异很大，直接向量检索可能匹配不上
2. **解法**：让 LLM 先假设一个"理想答案文档"，用这个假设文档去检索相似度，而不是用原始问题
3. **效果**：假设文档的语义分布更接近知识库内容，检索准确度大幅提升

**流程**：用户问题 → LLM 生成假设答案 → 假设答案向量化 → 检索相似真实文档 → 用检索到的文档生成最终回答

### 完整可运行代码

```python
"""
Day22 Part 1: HyDE 假设性文档嵌入 — 完整可运行示例
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
from langchain_core.runnables import RunnableLambda

# ============================
# 1. 初始化组件
# ============================
llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 2. 模拟知识库文档（实际使用可从 PDF/网页/数据库加载）
knowledge_docs = [
    "Python 是一门由 Guido van Rossum 于 1991 年创建的通用编程语言，支持面向对象、函数式和结构化编程范式。Python 以其简洁清晰的语法著称，被广泛应用于 Web 开发、数据科学、人工智能、自动化脚本等领域。",
    "Django 是 Python 最流行的 Web 框架之一，遵循 MTV（Model-Template-View）架构模式。它提供了 ORM、表单处理、用户认证、管理后台等开箱即用的功能，适合快速开发复杂的数据库驱动网站。Django 采用'约定优于配置'的设计理念。",
    "Flask 是 Python 的轻量级微框架，核心功能精简，通过扩展机制支持数据库集成、表单验证、OAuth 认证等功能。Flask 适合小型项目和 API 服务，开发者可以根据需求选择所需组件，灵活度高。",
    "FastAPI 是现代高性能 Python Web 框架，基于 Python 3.7+ 的类型提示（Type Hints）和 Starlette 异步框架。它自动生成 OpenAPI/Swagger 文档，支持异步请求处理，性能接近 NodeJS 和 Go 框架。",
    "Pydantic 是 Python 的数据验证和设置管理库，基于类型注解进行运行时数据校验。FastAPI 使用 Pydantic 来解析请求数据、生成 JSON Schema、自动生成 API 文档。",
    "Python 的 GIL（全局解释器锁）限制了 CPython 在多线程场景下的 CPU 并行效率。要绕过 GIL，可以使用多进程（multiprocessing）、异步编程（asyncio）、或者换用 Jython/PyPy 等替代解释器。",
    "pip 是 Python 的包管理工具，用于安装和管理第三方库。virtualenv 和 venv 用于创建虚拟环境。pipenv 和 poetry 是更现代的依赖管理工具。",
]

# ============================
# 3. 构建向量库
# ============================
vectorstore = FAISS.from_texts(knowledge_docs, embeddings)

# ============================
# 4. HyDE 核心：生成假设答案
# ============================
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的专家。请基于你的知识，对用户的问题写出一段简短的答案草稿（2-3句话）。这是为了帮助检索，不是最终答案。直接写出你认为正确的内容。"),
    ("human", "问题：{question}")
])

generate_hypothetical = hyde_prompt | llm | StrOutputParser()

# ============================
# 5. 用假设答案去检索
# ============================
def retrieve_with_hyde(hypothetical_answer):
    """用 LLM 生成的假设答案去检索向量库"""
    docs = vectorstore.similarity_search(hypothetical_answer, k=2)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context

# ============================
# 6. 最终回答生成
# ============================
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术问答助手。请基于以下文档内容回答问题。\n\n要求：\n- 回答必须基于所给文档\n- 信息不足时请明确说明\n- 回答要详细、有条理"),
    ("human", "问题：{question}\n\n相关文档：\n{context}\n\n请回答问题。")
])

# ============================
# 7. LCEL 完整 HyDE RAG 链
# ============================
hyde_rag_chain = (
    {
        "question": lambda x: x,
        "hypothetical": generate_hypothetical,
    }
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "hypothetical": x["hypothetical"],
        "context": retrieve_with_hyde(x["hypothetical"]),
    })
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ============================
# 8. 运行测试
# ============================
if __name__ == "__main__":
    print("=" * 60)
    print("HyDE 假设性文档嵌入 — 测试")
    print("=" * 60)

    for question in [
        "Python Web 开发用什么框架比较好？",
        "如何绕过 Python 的 GIL 限制？",
        "Python 的依赖管理有哪些工具？",
    ]:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        answer = hyde_rag_chain.invoke(question)
        print(f"回答: {answer}")
        print("=" * 60)
```

**检查点**：
- [ ] 安装完依赖后直接 `python part1_hyde_rag.py` 能跑通
- [ ] 能理解 HyDE 和传统 RAG 的区别
- [ ] 能用自己的话解释"为什么要生成假设答案"

---

## Part 2：Multi-query 多路检索 🔀

### 原理

1. **问题**：用户问题的表达方式可能不够全面，单路查询可能遗漏相关内容
2. **解法**：让 LLM 生成 3-5 个不同角度的等价查询，每路都去检索，最后合并去重
3. **效果**：检索覆盖率大幅提升

**流程**：用户问题 → LLM 生成多种查询 → 多路并行检索 → 合并去重 → 生成回答

### 完整可运行代码

```python
"""
Day22 Part 2: Multi-query 多路检索 — 完整可运行示例
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

# 2. 知识库文档
knowledge_docs = [
    "Python 是一门由 Guido van Rossum 于 1991 年创建的通用编程语言，支持面向对象、函数式和结构化编程范式。广泛应用于 Web 开发、数据科学、AI 等领域。",
    "Django 是 Python 最流行的 Web 框架之一，遵循 MTV 架构，提供 ORM、表单处理、用户认证、管理后台等功能，适合大型数据库驱动网站。",
    "Flask 是 Python 的轻量级微框架，核心精简，通过扩展机制支持数据库集成、表单验证、OAuth 认证。适合小型项目和 API 服务。",
    "FastAPI 是现代高性能 Python Web 框架，基于类型提示和 Starlette 异步框架，自动生成 API 文档。",
    "Pydantic 是 Python 的数据验证库，FastAPI 用它来解析请求和生成 JSON Schema。",
    "Python 的 GIL 限制了多线程 CPU 并行，可用多进程、异步编程绕过。",
    "pip 是包管理工具，venv 创建虚拟环境，poetry 是现代依赖管理工具。",
    "NumPy 提供多维数组和数学计算，Pandas 基于 NumPy 提供数据框。",
    "Matplotlib 是基础绘图库，Seaborn 基于它提供更高级的统计图表。",
    "TensorFlow (Google) 和 PyTorch (Meta) 是主流深度学习框架。",
]

vectorstore = FAISS.from_texts(knowledge_docs, embeddings)

# ============================
# 3. Multi-query Prompt
# ============================
mq_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息检索专家。根据用户问题生成 3 条不同角度但语义等价的查询，每条一行。只输出查询语句。"),
    ("human", "原始问题：{question}")
])

# ============================
# 4. 多路检索 + 合并去重
# ============================
def multi_retrieve(question):
    queries_text = (mq_prompt | llm | StrOutputParser()).invoke({"question": question})
    queries = [line.strip() for line in queries_text.strip().split("\n") if line.strip()][:3]
    
    all_docs = []
    seen = set()
    for q in queries:
        print(f"  查询: {q}")
        for doc in vectorstore.similarity_search(q, k=2):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    
    print(f"  共检索到 {len(all_docs)} 篇不重复文档")
    context = "\n\n---\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(all_docs)])
    return context, len(all_docs)

# ============================
# 5. 回答生成
# ============================
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下文档详细回答问题：\n{context}"),
    ("human", "问题：{question}")
])

# ============================
# 6. Multi-query RAG 完整流程
# ============================
def multi_query_rag(question):
    print(f"\n问题: {question}")
    context, doc_count = multi_retrieve(question)
    answer = (qa_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    return answer, doc_count

# ============================
# 7. 测试
# ============================
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-query 多路检索 — 测试")
    print("=" * 60)
    
    for q in [
        "Python 适合做什么？",
        "Python 有哪些好的 Web 框架？",
        "Python 如何高效处理数据？",
    ]:
        answer, count = multi_query_rag(q)
        print(f"回答 (共 {count} 篇文档): {answer}")
        print("=" * 60)
```

**检查点**：
- [ ] 直接 `python part2_multi_query_rag.py` 能跑通
- [ ] 能看到 LLM 生成了 3 条不同的查询
- [ ] 多路检索的文档数比单路多

---

## Part 3：Self-RAG 自我反思 🤔

### 原理

1. **问题**：传统 RAG 检索到的文档可能不相关或质量不佳，系统不知道
2. **解法**：让 LLM 对检索结果进行自我评估（相关性、完整性），如果不满意就重新检索
3. **关键**：引入"反思"步骤，检索后判断是否足够，不足则改进查询再次检索

**流程**：检索 → 自我评估 → 不足则改进查询 → 再检索 → 直到足够则生成回答

### 完整可运行代码

```python
"""
Day22 Part 3: Self-RAG 自我反思 — 完整可运行示例
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

# 2. 知识库
knowledge_docs = [
    "Python 是一门由 Guido van Rossum 于 1991 年创建的编程语言，支持面向对象、函数式编程。",
    "Django 是 Python 的 Web 框架，MTV 架构，内置 ORM、认证、管理后台，适合大型项目。",
    "Flask 是轻量级微框架，核心精简，扩展丰富，适合小型项目、微服务和 API。",
    "FastAPI 是现代高性能 Python Web 框架，自动 API 文档，异步支持。",
    "Pydantic 是数据验证库，FastAPI 深度集成。",
    "Python GIL 限制多线程并行，可用多进程、异步解决。",
    "pip + venv 是标准依赖管理，poetry 是现代方案。",
    "NumPy + Pandas 是数据科学基础，Scikit-learn 是机器学习库。",
    "TensorFlow (Google) 和 PyTorch (Meta) 是主流深度学习框架。",
    "Matplotlib + Seaborn 用于 Python 数据可视化。",
    "Scikit-learn 提供分类、回归、聚类等传统机器学习算法。",
]

vectorstore = FAISS.from_texts(knowledge_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ============================
# 3. 自我评估 Prompt
# ============================
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "评估检索文档是否足够回答问题。\n\n格式：\n- 相关性: [充分/不充分]\n- 完整性: [充分/不充分]\n- 建议: [如何改进]\n\n最后一行单独写 ENOUGH 或 NEED_MORE"),
    ("human", "问题：{question}\n\n检索文档：\n{context}")
])

# 4. 改进查询 Prompt
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", "重新设计一个检索查询以更好地找到答案。只输出一句话。"),
    ("human", "问题：{question}\n当前文档：{context}")
])

# 5. 回答 Prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下文档详细回答问题。信息不足请说明。\n\n文档：\n{context}"),
    ("human", "问题：{question}")
])

# ============================
# 6. Self-RAG 核心流程
# ============================
def self_rag(question, max_iterations=2):
    """Self-RAG：带自我评估的多轮检索"""
    all_docs = []
    context = ""
    
    print("开始 Self-RAG 流程...")
    
    for i in range(max_iterations):
        # 检索（第一轮直接检索，后续用改进查询）
        if i == 0:
            docs = retriever.invoke(question)
        else:
            refined_query = (refine_prompt | llm | StrOutputParser()).invoke({
                "question": question,
                "context": context
            })
            print(f"  改进查询: {refined_query}")
            docs = vectorstore.similarity_search(refined_query, k=2)
        
        # 合并去重
        new_docs = [d for d in docs if d.page_content not in {old.page_content for old in all_docs}]
        all_docs.extend(new_docs)
        context = "\n\n---\n\n".join([d.page_content for d in all_docs])
        print(f"  第 {i+1} 轮检索到 {len(new_docs)} 篇新文档，共 {len(all_docs)} 篇")
        
        # 自我评估
        evaluation = (eval_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "context": context
        })
        print(f"  评估: {evaluation}")
        
        if "ENOUGH" in evaluation:
            print("  文档足够，生成答案")
            break
        elif not new_docs:
            print("  没有新内容，停止迭代")
            break
    
    # 生成最终答案
    answer = (qa_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    return answer, len(all_docs)

# ============================
# 7. 测试
# ============================
if __name__ == "__main__":
    print("=" * 60)
    print("Self-RAG 自我反思 — 测试")
    print("=" * 60)
    
    for q in [
        "Python 在机器学习领域有哪些主流框架？各自特点？",
        "Python 的 Web 开发框架怎么选？",
        "Python 数据可视化有哪些方案？",
    ]:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        answer, count = self_rag(q)
        print(f"\n回答 (参考 {count} 篇文档): {answer}")
        print("=" * 60)
```

**检查点**：
- [ ] 直接 `python part3_self_rag.py` 能跑通
- [ ] 能看到自我评估的过程
- [ ] 能看到改进查询和补充检索

---

## Part 4：三种策略对比总结 ⚔️

| 策略 | 原理 | 适用场景 | 核心代码行数 |
|------|------|----------|-------------|
| 标准 RAG | 问题直接向量化检索 | 问题表达清晰，和知识库语义一致 | 最少 |
| HyDE | 生成假设答案后再检索 | 问题表达模糊，和文档语义差异大 | 少 |
| Multi-query | 多路查询合并去重 | 需要全面覆盖，避免遗漏 | 中等 |
| Self-RAG | 自我评估 + 迭代检索 | 对回答质量要求高，复杂问题 | 最多 |

**经验法则**：
- 日常项目优先用标准 RAG
- 知识库内容专业性强 → 加 HyDE
- 问题类型多样 → 加 Multi-query
- 对质量要求极高 → 用 Self-RAG

---

## 🔗 参考资料

- HyDE 论文: https://arxiv.org/abs/2212.10496
- Self-RAG 论文: https://arxiv.org/abs/2310.11511
- LangChain RAG: https://python.langchain.com/docs/tutorials/rag/
