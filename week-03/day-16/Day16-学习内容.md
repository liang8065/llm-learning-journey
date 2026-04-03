# 📚 Day 16 — 工具生态扩展

> **周期**：Day 16 / Week 3
> **目标**：自定义工具深入、外部 API 集成、工具安全
> **预计用时**：1.5 小时

---

## Part 1：高级自定义工具开发（30分钟）

### 工具设计原则
1. **描述准确** — Agent 靠描述选工具
2. **类型安全** — 用 Pydantic 定义输入
3. **错误处理** — 不能崩溃，返回友好信息

```python
"""
Day16 Part 1: 高级自定义工具
依赖安装:
    pip install langchain langchain-community langchain-core dashscope requests
"""

import os
import json, requests
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

@tool
def json_tool(raw: str, op: str = "format") -> str:
    """处理 JSON。op: format(格式化), parse(解析), keys(提取键名)"""
    try:
        obj = json.loads(raw)
        if op == "format":
            return json.dumps(obj, indent=2, ensure_ascii=False)
        elif op == "keys":
            return f"键名: {', '.join(obj.keys())}"
        return str(obj)
    except Exception as e:
        return f"JSON 错误: {e}"

@tool
def system_info() -> str:
    """查看系统信息"""
    import platform
    return json.dumps({
        "Python": platform.python_version(),
        "系统": f"{platform.system()} {platform.release()}",
    }, ensure_ascii=False)

tools = [json_tool, system_info]
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是工具助手，处理 JSON、查看系统信息。"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("=== 工具测试 ===")
for q in [
    "格式化: {'name': 'Python', 'version': 3.10}",
    "查系统信息",
]:
    print(f"
Q: {q}")
    r = agent_executor.invoke({"input": q})
    print(f"A: {r['output']}")
    print("=" * 50)
```

---

## Part 2：外部 API 集成（30分钟）

```python
"""
Day16 Part 2: 外部 API 集成
"""

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3&lang=zh", timeout=5)
        return resp.text
    except Exception as e:
        return f"查询失败: {e}"

@tool
def wiki_search(query: str) -> str:
    """维基百科搜索"""
    try:
        url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{query}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            d = resp.json()
            return f"标题: {d.get('title')}\n摘要: {d.get('extract')}"
        return f"HTTP {resp.status_code}"
    except Exception as e:
        return f"搜索失败: {e}"

print("=== 外部 API ===")
print("1. 天气:", get_weather.invoke("Beijing"))
print("2. 维基:", wiki_search.invoke("Python编程语言"))
```

---

## Part 3：工具安全清单（15分钟）

```python
"""
Day16 Part 3: 工具安全
"""

checklist = [
    "✅ 验证输入：Pydantic 严格定义输入",
    "✅ 最小权限：工具只开放必要权限",
    "✅ 超时控制：防止无限阻塞",
    "✅ 错误处理：捕获异常，返回友好信息",
    "✅ 日志记录：记录每次调用",
    "✅ 频率限制：防止滥用",
    "❌ 禁止执行未信任代码",
]

print("=== 工具安全清单 ===")
for c in checklist:
    print(f"  {c}")
```
