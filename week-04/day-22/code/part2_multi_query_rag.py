"""
Day22 Part 2: Multi-query 多路检索 — 完整可运行示例
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

# 1. 初始化
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
    "Python 是一门由 Guido van Rossum 于 1991 年创建的通用编程语言，广泛应用于 Web 开发、数据科学、AI等领域。",
    "Django 是 Python 最受欢迎的 Web 框架，MTV 架构，内置 ORM、表单处理、用户认证、管理后台，适合大型项目。",
    "Flask 是轻量级微框架，核心精简，扩展丰富，适合小型项目、微服务和 API 服务。",
    "FastAPI 是现代高性能 Python Web 框架，自动 API 文档，异步支持，性能接近 NodeJS 和 Go。",
    "Pydantic 是 Python 数据验证库，FastAPI 深度集成，用于请求解析和 JSON Schema 生成。",
    "Python GIL 限制多线程 CPU 并行，可用多进程、asyncio 或替代解释器绕过。",
    "pip + venv 是标准依赖管理，poetry 是现代方案，支持依赖锁定。",
    "NumPy + Pandas 是数据科学基础，Scikit-learn 是机器学习库。",
    "TensorFlow (Google) 和 PyTorch (Meta) 是主流深度学习框架。",
]

vectorstore = FAISS.from_texts(knowledge_docs, embeddings)

# 3. Multi-query Prompt
mq_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息检索专家。根据用户问题生成3条不同角度但语义等价的查询，每行一条。只输出查询，不要其他内容。"),
    ("human", "原始问题：{question}")
])

# 4. 检索
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
    context = "\n\n---\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(all_docs)])
    return context, len(all_docs)

# 5. 最终回答
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下文档详细回答问题：\n{context}"),
    ("human", "问题：{question}")
])

# 6. 完整流程
def multi_query_rag(question):
    print("生成的查询：")
    context, doc_count = multi_retrieve(question)
    answer = (qa_prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, doc_count

# 7. 测试
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-query 多路检索 - 测试")
    print("=" * 60)
    for q in [
        "Python 适合做什么？",
        "Python 有哪些好的 Web 框架？",
        "Python 如何高效处理数据？",
    ]:
        print(f"\n问题: {q}")
        answer, count = multi_query_rag(q)
        print(f"回答: {answer}")
        print("=" * 60)
