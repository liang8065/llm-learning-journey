# 📚 Day 15 — 复杂 Agent 系统设计

> **周期**：Day 15 / Week 3
> **目标**：掌握多 Agent 协作，角色分配与任务分解
> **预计用时**：1.5 小时

---

## Part 1：多 Agent 协作架构（30分钟）

### 为什么需要多 Agent？

单个 Agent 能力有限。多个专业 Agent 各司其职：研究员→分析师→作家→审核

```python
"""
Day15 Part 1: 多 Agent 协作
依赖安装:
    pip install langgraph langchain langchain-community langchain-core dashscope
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ResearchState(TypedDict):
    topic: str
    research: str
    analysis: str
    article: str
    review: str
    approved: bool

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

def researcher(state):
    p = ChatPromptTemplate.from_messages([
        ("system", "你是研究员。搜集关于主题的 3-5 个关键信息点。"),
        ("human", "主题: {topic}")
    ])
    state["research"] = (p | llm | StrOutputParser()).invoke({"topic": state["topic"]})
    return state

def analyst(state):
    p = ChatPromptTemplate.from_messages([
        ("system", "你是分析师。提取核心观点并深入分析。"),
        ("human", "研究: {research}")
    ])
    state["analysis"] = (p | llm | StrOutputParser()).invoke({"research": state["research"]})
    return state

def writer(state):
    p = ChatPromptTemplate.from_messages([
        ("system", "你是科技作家。写一篇 300 字科普文章。"),
        ("human", "分析: {analysis}")
    ])
    state["article"] = (p | llm | StrOutputParser()).invoke({"analysis": state["analysis"]})
    return state

def reviewer(state):
    p = ChatPromptTemplate.from_messages([
        ("system", "你是编辑。审核质量，输出 APPROVE 或 REJECT。"),
        ("human", "文章: {article}")
    ])
    review = (p | llm | StrOutputParser()).invoke({"article": state["article"]})
    state["review"] = review
    state["approved"] = "APPROVE" in review.upper()
    return state

# 构建工作流
wf = StateGraph(ResearchState)
for name, fn in [("research", researcher), ("analyze", analyst), ("write", writer), ("review", reviewer)]:
    wf.add_node(name, fn)
wf.set_entry_point("research")
wf.add_edge("research", "analyze")
wf.add_edge("analyze", "write")
wf.add_edge("write", "review")
wf.add_conditional_edges("review", lambda s: "research" if not s["approved"] else "end",
    {"research": "research", "end": END})

app = wf.compile()

print("=== 多 Agent 协作 ===")
r = app.invoke({"topic": "大语言模型在教育中的应用", "research": "", "analysis": "", "article": "", "review": "", "approved": False})
print(f"研究: {r['research'][:150]}...")
print(f"文章: {r['article'][:200]}...")
print(f"审核: {r['review'][:80]}...")
```

---

## Part 2：角色分配与任务分解（30分钟）

```python
"""
Day15 Part 2: Agent 角色系统
"""

roles = {
    "researcher": {"描述": "资料搜集", "技能": ["搜索", "摘要", "分类"], "工具": ["搜索引擎"]},
    "coder": {"描述": "代码编写", "技能": ["编程", "调试"], "工具": ["Python 解释器"]},
    "reviewer": {"描述": "质量审核", "技能": ["审查", "评估"], "工具": ["检查清单"]},
    "writer": {"描述": "内容创作", "技能": ["写作", "编辑"], "工具": ["模板引擎"]},
}

print("=== Agent 角色 ===")
for n, info in roles.items():
    print(f"  【{n}】{info['描述']} → 技能: {', '.join(info['技能'])}")

# 任务分解引擎
def decompose(task):
    subtasks = []
    if "调研" in task or "研究" in task:
        subtasks.append({"role": "researcher", "action": "搜集资料"})
    if "写" in task or "文章" in task:
        subtasks.append({"role": "writer", "action": "撰写"})
    if "代码" in task or "开发" in task:
        subtasks.append({"role": "coder", "action": "编写代码"})
    subtasks.append({"role": "reviewer", "action": "审核质量"})
    return subtasks

print("
=== 任务分解 ===")
for i, st in enumerate(decompose("调研并写一份 Python Web 框架报告"), 1):
    print(f"  {i}. [{st['role']}] {st['action']}")
```

---

## Part 3：实战 — 学习小组 Agent（15分钟）

```python
"""
Day15 Part 3: 学习小组 Agent
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict

class StudyState(TypedDict):
    question: str
    teacher: str
    tutor: str
    peer: str

wf = StateGraph(StudyState)
wf.add_node("teacher", lambda s: {**s, "teacher": f"👩‍🏫 老师: [给出答案]"})
wf.add_node("tutor", lambda s: {**s, "tutor": f"🧑‍🏫 导师: [详细解释]"})
wf.add_node("peer", lambda s: {**s, "peer": f"👨‍🎓 同学: [补充理解]"})
wf.set_entry_point("teacher")
wf.add_edge("teacher", "tutor")
wf.add_edge("tutor", "peer")
wf.add_edge("peer", END)

app = wf.compile()
r = app.invoke({"question": "什么是 RAG？", "teacher": "", "tutor": "", "peer": ""})
for k, v in r.items():
    print(f"{k}: {v}")
```
