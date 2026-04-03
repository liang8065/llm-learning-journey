"""
Day22 Part 1: HyDE 假设性文档嵌入 — 完整可运行示例
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
    "Python 是一门由 Guido van Rossum 于 1991 年创建的通用编程语言，支持面向对象、函数式和结构化编程范式。广泛应用于 Web 开发、数据科学、人工智能、自动化脚本。",
    "Django 是 Python 最流行的 Web 框架之一，遵循 MTV 架构模式。提供 ORM、表单处理、用户认证、管理后台，适合快速开发数据库驱动网站。",
    "Flask 是 Python 的轻量级微框架，核心精简，通过扩展机制支持数据库集成、表单验证、OAuth 认证等。适合小型项目和 API 服务。",
    "FastAPI 是现代高性能 Python Web 框架，基于类型提示和 Starlette 异步框架。自动生成 OpenAPI 文档，支持异步请求。",
    "Pydantic 是 Python 的数据验证库，FastAPI 使用它来解析请求和生成 JSON Schema。",
    "Python 的 GIL 限制了多线程的 CPU 并行效率，可用多进程、异步编程绕过。",
    "pip 是 Python 的包管理工具，venv 用于虚拟环境，poetry 是更现代的依赖管理工具。",
]

# 3. 向量库
vectorstore = FAISS.from_texts(knowledge_docs, embeddings)

# 4. HyDE: 生成假设答案
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是知识专家。请对用户问题写出2-3句话的答案草稿，用于检索相关内容。直接写，不要加任何解释。"),
    ("human", "问题：{question}")
])
generate_hypothetical = hyde_prompt | llm | StrOutputParser()

# 5. 用假设答案检索
def retrieve_with_hyde(hypothetical):
    docs = vectorstore.similarity_search(hypothetical, k=2)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# 6. 最终回答
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下文档详细回答问题：\n{context}"),
    ("human", "问题：{question}")
])

# 7. LCEL 完整 HyDE RAG 链
hyde_chain = (
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

# 8. 测试
if __name__ == "__main__":
    print("=" * 60)
    print("HyDE 假设性文档嵌入 - 测试")
    print("=" * 60)
    for q in [
        "Python Web 开发用什么框架比较好？",
        "如何绕过 Python 的 GIL 限制？",
        "Python 的依赖管理有哪些工具？",
    ]:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        answer = hyde_chain.invoke(q)
        print(f"回答: {answer}")
        print("=" * 60)
