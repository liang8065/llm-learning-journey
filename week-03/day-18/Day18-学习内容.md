# 📚 Day 18 — 高级 Prompt 技术

> **周期**：Day 18 / Week 3
> **目标**：Prompt 模板化、版本管理、动态生成
> **预计用时**：1.5 小时

---

## Part 1：Prompt 模板与版本管理（30分钟）

```python
"""
Day18 Part 1: Prompt 模板系统
依赖安装:
    pip install langchain langchain-community langchain-core dashscope
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# Prompt 模板定义
PROMPTS = {
    "translator": {
        "system": "你是翻译专家。将{source_lang}翻译成{target_lang}。",
        "human": "原文：{text}",
    },
    "translator_v2": {
        "system": "你是资深翻译。翻译{source_lang}到{target_lang}，考虑上下文。",
        "human": "上下文：{context}\n原文：{text}",
    },
    "summarizer": {
        "system": "你是摘要专家。用{words}字总结以下内容。",
        "human": "内容：{text}",
    },
}

# Prompt 工厂
class PromptFactory:
    def get(self, name, version=None):
        key = f"{name}_v{version}" if version else name
        if not version:
            variants = [k for k in PROMPTS if k.startswith(f"{name}_")]
            if variants:
                key = sorted(variants)[-1]
        t = PROMPTS.get(key)
        if not t:
            raise ValueError(f"Prompt '{name}' not found")
        return ChatPromptTemplate.from_messages([("system", t["system"]), ("human", t["human"])])

factory = PromptFactory()
prompt = factory.get("translator", "v2")
chain = prompt | llm | StrOutputParser()

print("=== Prompt 模板测试 ===")
result = chain.invoke({
    "source_lang": "中文",
    "target_lang": "英文",
    "context": "科技",
    "text": "人工智能正在改变世界"
})
print(f"翻译: {result}")

# 版本对比
print("\n=== 版本对比 ===")
v1 = factory.get("translator")
v2 = factory.get("translator", "v2")
print("v1 系统: 简单翻译")
print("v2 系统: 考虑上下文的资深翻译")
```

---

## Part 2：动态 Prompt 生成（30分钟）

```python
"""
Day18 Part 2: 动态 Prompt
"""

def build_prompt(task, context="", examples=[], output_format="text"):
    """动态构建 Prompt"""
    parts = []

    # 系统角色
    parts.append(("system", f"你是{task}专家。"))

    # Few-shot examples
    if examples:
        for ex in examples:
            parts.append(("human", ex["input"]))
            parts.append(("ai", ex["output"]))

    # 输出格式
    if output_format == "json":
        parts.append(("system", "请以 JSON 格式输出。"))

    return ChatPromptTemplate.from_messages(parts)

# 测试
p1 = build_prompt("翻译", examples=[
    {"input": "你好", "output": "Hello"},
])
p2 = build_prompt("翻译", examples=[
    {"input": "你好", "output": "Hello"},
    {"input": "世界", "output": "World"},
], output_format="json")

print("=== 动态 Prompt ===")
print(f"P1 (简单): {len(p1.messages)} 条消息")
print(f"P2 (复杂): {len(p2.messages)} 条消息")
```

---

## Part 3：Prompt 评估与优化（15分钟）

```python
"""
Day18 Part 3: Prompt 评估
"""

def evaluate_prompt(prompt_template, test_cases, llm):
    """评估 Prompt 质量"""
    results = []
    for tc in test_cases:
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke(tc["input"])
        score = 1 if tc["expected_keyword"] in answer else 0
        results.append({"input": tc["input"][:50], "score": score, "answer": answer[:100]})
    return results

print("=== Prompt 评估 ===")
print("评估维度: 答案包含预期关键词 = 1 分")
print("持续优化 Prompt 直到测试全通过")
```
