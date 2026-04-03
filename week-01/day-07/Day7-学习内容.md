# 📚 Day 7 — Week 1 总结 + 项目实战

> **周期**：Day 7 / Week 1
> **目标**：回顾总结，名著改写器优化，Week 2 计划
> **预计用时**：1.5 小时

---

## Part 1：Week 1 知识回顾（30分钟）

```python
"""
Day7 Part 1: Week 1 知识回顾
"""

summary = {
    "Day 1": ("环境搭建 + 核心概念", ["Python环境", "LLM", "Token", "Prompt", "Embedding", "RAG"]),
    "Day 2": ("Prompt + LangChain 入门", ["Few-shot", "CoT", "LCEL", "LangChain 1.0+"]),
    "Day 3": ("LCEL + RAG 基础", ["RunnableParallel", "RAG 7步流程", "向量存储"]),
    "Day 4": ("RAG 实战 + 向量数据库", ["FAISS", "ChromaDB", "文档分割", "知识库问答"]),
    "Day 5": ("Agent 开发基础", ["ReAct", "工具开发", "@tool", "推理循环"]),
    "Day 6": ("微调 + 部署", ["LoRA", "QLoRA", "Ollama", "vLLM"]),
}

print("=== Week 1 回顾 ===\n")
for day, (topic, keywords) in summary.items():
    print(f"{day}: {topic}")
    print(f"  → {', '.join(keywords)}\n")

# 自测
quiz = [
    ("RAG 的 7 步流程？", "加载→分割→向量化→存储→检索→Prompt组装→LLM生成"),
    ("LCEL 怎么组合？", "用 | 运算符: prompt | llm | output_parser"),
    ("Agent 核心？", "ReAct: Thought → Action → Observation → 循环"),
]

print("\n=== 自测 ===")
for i, (q, a) in enumerate(quiz, 1):
    print(f"Q{i}: {q}")
    print(f"A: {a}\n")
```

---

## Part 2：名著改写器优化（30分钟）

```python
"""
Day7 Part 2: Prompt 优化实战
环境变量:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 优化后 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是文学风格改写专家。将文本改写成指定风格，要求：
1. 保持原文核心内容和段落结构
2. 完全模仿目标作者的遣词造句
3. 使用目标作者常用的修辞手法
4. 语言流畅自然
5. 只输出改写结果"""),
    ("human", "风格：{style}\n\n原文：{text}")
])

rewriter = prompt | llm | StrOutputParser()

print("=== 名著改写器优化测试 ===\n")

text = "今天天气很好，公园里很多人在散步。孩子们在草地上奔跑，老人在树下聊天。"

for style in ["鲁迅", "张爱玲", "老舍"]:
    print(f"--- {style} ---")
    result = rewriter.invoke({"style": style, "text": text})
    print(result)
    print()
```

---

## Part 3：Week 2 计划（15分钟）

```python
"""
Day7 Part 3: Week 2 计划
"""

week2 = {
    "Day 8": "多轮对话系统（RunnableWithMessageHistory）",
    "Day 9": "RAG 系统优化（检索策略、Chunking、评估）",
    "Day 10": "多模态 LLM（Qwen-VL 图片分析）",
    "Day 11": "LangGraph 工作流编排",
    "Day 12": "API 设计与项目架构（FastAPI）",
    "Day 13": "测试与质量保障",
    "Day 14": "Week 2 总结 + 阶段回顾",
}

print("=== Week 2 计划 ===\n")
for day, topic in week2.items():
    print(f"  {day}: {topic}")
```