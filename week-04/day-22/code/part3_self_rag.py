"""
Day22 Part 3: Self-RAG 自我反思 — 完整可运行示例
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
    "Python 是一门由 Guido van Rossum 于 1991 年创建的通用编程语言，支持面向对象、函数式和结构化编程范式。",
    "Django 是 Python 最流行的 Web 框架之一，MTV 架构，内置 ORM、表单处理、用户认证、管理后台。",
    "Flask 是 Python 的轻量级微框架，核心精简，通过扩展机制支持各种功能，适合小型项目和 API 服务。",
    "FastAPI 是现代高性能 Python Web 框架，基于类型提示和 Starlette 异步框架，自动生成 API 文档。",
    "Pydantic 是 Python 的数据验证库，FastAPI 用它来解析请求和生成 JSON Schema。",
    "Python 的 GIL 限制了多线程 CPU 并行，可用多进程、异步编程绕过。",
    "pip 是包管理工具，venv 创建虚拟环境，poetry 是更现代的依赖管理工具。",
    "NumPy 提供多维数组和数学计算，Pandas 基于 NumPy 提供数据框和数据分析。",
    "Matplotlib 是基础绘图库，Seaborn 基于它提供更高级的统计图表。",
    "Scikit-learn 是 Python 的机器学习库。深度学习通常使用 TensorFlow 或 PyTorch。",
    "TensorFlow 是 Google 开发的深度学习框架，PyTorch 由 Meta 开发，以动态图和易用性著称。",
]

vectorstore = FAISS.from_texts(knowledge_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. 自我评估 prompt
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "评估检索文档是否足够回答问题。格式：\n- 相关性: [充分/不充分]\n- 完整性: [充分/不充分]\n- 建议: [如何改进]\n\n最后一行只写 ENOUGH 或 NEED_MORE"),
    ("human", "问题：{question}\n\n检索文档：\n{context}")
])

# 4. 改进查询 prompt
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", "重新设计一个检索查询，以更好地找到答案。只输出一句话。"),
    ("human", "问题：{question}\n\n当前文档：{context}")
])

# 5. 回答 prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下文档详细回答问题。如果信息不足请说明。\n\n文档：\n{context}"),
    ("human", "问题：{question}")
])

# 6. Self-RAG 核心流程
def self_rag(question, max_iterations=2):
    all_docs = []
    context = ""
    
    print("开始 Self-RAG 流程...")
    
    for i in range(max_iterations):
        # 检索
        docs = retriever.invoke(question) if i == 0 else vectorstore.similarity_search(
            (refine_prompt | llm | StrOutputParser()).invoke({"question": question, "context": context}),
            k=2
        )
        
        new_docs = [d for d in docs if d.page_content not in {old.page_content for old in all_docs}]
        all_docs.extend(new_docs)
        context = "\n\n---\n\n".join([d.page_content for d in all_docs])
        print(f"第 {i+1} 轮检索到 {len(new_docs)} 篇新文档，共 {len(all_docs)} 篇")
        
        # 评估
        evaluation = (eval_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "context": context
        })
        print(f"评估: {evaluation}")
        
        if "ENOUGH" in evaluation:
            print("文档足够，生成答案")
            break
        elif not new_docs:
            print("没有新的相关内容，停止迭代")
            break
    
    # 生成答案
    answer = (qa_prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, len(all_docs)

# 7. 测试
if __name__ == "__main__":
    print("=" * 60)
    print("Self-RAG 自我反思 - 测试")
    print("=" * 60)
    for q in [
        "Python 在机器学习领域有哪些主流框架？各自特点是什么？",
        "Python 的 Web 开发框架该怎么选？",
        "Python 数据可视化有哪些方案？",
    ]:
        print(f"\n问题: {q}")
        answer, count = self_rag(q)
        print(f"\n回答 (参考 {count} 篇文档): {answer}")
        print("=" * 60)
