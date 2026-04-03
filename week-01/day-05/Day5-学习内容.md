# 📚 Day 5 — Agent 开发基础

> **周期**：Day 5 / Week 1
> **目标**：理解 Agent 概念，掌握 ReAct 模式，构建第一个 Agent
> **预计用时**：1.5 小时

---

## Part 1：Agent 概念与 ReAct 模式（30分钟）

### 什么是 Agent？

Agent = **LLM + 工具 + 推理循环**。传统 LLM 只会"说"，有了 Agent 能"做"事情。

**ReAct 模式**：
1. **Thought（思考）**：分析意图，决定步骤
2. **Action（行动）**：调用工具
3. **Observation（观察）**：获取工具结果
4. 循环直到得到答案

### 完整可运行代码

```python
"""
Day5 Part 1: Agent 概念 + 第一个 Agent
依赖安装:
    pip install langchain langchain-community langchain-core dashscope langgraph
"""

import os
import ast
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# ============================
# 1. 定义工具
# ============================
@tool
def calculate(expression: str) -> str:
    """执行数学计算。输入格式如 '123 * 456' 或 '100+200*3'"""
    try:
        result = ast.literal_eval(expression)
        return f"计算结果: {result}"
    except Exception:
        return f"计算错误: '{expression}'"

@tool
def weather(city: str) -> str:
    """查询城市天气。输入城市名"""
    db = {"北京": "晴，5~15°C", "上海": "多云，8~12°C", "广州": "阴，18~25°C"}
    return db.get(city, f"未找到 {city} 的天气")

@tool
def word_count(text: str) -> str:
    """统计文字数。输入任意文本"""
    cn = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    return f"中文 {cn} 字，总 {len(text)} 字符"

tools = [calculate, weather, word_count]

# ============================
# 2. 创建 Agent
# ============================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，可以使用计算器、天气查询和字数统计工具。直接回答用户问题。"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ============================
# 3. 测试
# ============================
print("=== Agent 测试 ===")
for q in [
    "今天北京天气怎么样？",
    "123 乘以 456 等于几？",
    "统计一下 'LangChain 是大语言模型框架' 的字数",
]:
    print(f"\nQ: {q}")
    r = agent_executor.invoke({"input": q})
    print(f"A: {r['output']}")
    print("=" * 50)
```

---

## Part 2：自定义工具开发深入（30分钟）

```python
"""
Day5 Part 2: 自定义工具开发
环境变量:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
import json
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 工具 1: JSON 格式化
@tool
def format_json(raw: str) -> str:
    """将文本格式化为 JSON。输入 JSON 字符串"""
    try:
        obj = json.loads(raw)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"JSON 解析失败: {e}"

# 工具 2: 字符串处理
@tool
def text_tool(text: str, op: str = "length") -> str:
    """文本处理。op: length(长度), upper(大写), lower(小写), reverse(反转)"""
    ops = {
        "length": f"长度: {len(text)}",
        "upper": text.upper(),
        "lower": text.lower(),
        "reverse": text[::-1],
    }
    return ops.get(op, f"未知操作: {op}")

# 工具 3: 时间
@tool
def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [format_json, text_tool, get_time]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你有 JSON 格式化、文本处理、时间查询能力。"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("=== 自定义工具 Agent ===")
for q in [
    "格式化: {'name': 'Python', 'year': 1991}",
    "把 hello world 变成大写",
    "现在几点了？",
]:
    print(f"\nQ: {q}")
    r = executor.invoke({"input": q})
    print(f"A: {r['output']}")
    print("=" * 50)
```

---

## Part 3：Agent 调试技巧（15分钟）

```python
"""
Day5 Part 3: Agent 调试
"""

# Agent 调试技巧：verbose=True 显示推理过程
# 观察 Thought -> Action -> Observation 循环
# 确认 LLM 选择了正确的工具
# 确认工具输入输出格式正确

checklist = [
    "确认工具描述清晰准确",
    "确认工具输入输出类型正确",
    "确认 Prompt 中说明了可用工具",
    "设置 verbose=True 观察推理过程",
    "设置 max_iterations 防止无限循环",
    "每个工具添加异常处理",
]

print("=== Agent 调试清单 ===")
for i, item in enumerate(checklist, 1):
    print(f"  {i}. {item}")
```