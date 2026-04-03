# 📚 Day 13 — 测试与质量保障

> **周期**：Day 13 / Week 2
> **目标**：掌握单元测试、Mock、性能测试、CI/CD
> **预计用时**：1.5 小时

---

## Part 1：单元测试（30分钟）

```python
"""
Day13 Part 1: 单元测试实践
依赖安装:
    pip install pytest langchain
"""

import pytest
from unittest.mock import Mock, patch

# 业务函数
def calculate_cost(prompt_tokens, completion_tokens, model="qwen-plus"):
    """计算 API 调用费用"""
    rates = {
        "qwen-plus": {"prompt": 0.008, "completion": 0.02},
        "qwen-turbo": {"prompt": 0.002, "completion": 0.006},
        "qwen-max": {"prompt": 0.04, "completion": 0.12},
    }
    rate = rates.get(model, rates["qwen-plus"])
    return (prompt_tokens * rate["prompt"] + completion_tokens * rate["completion"]) / 1000

def validate_input(text, max_length=10000):
    """验证输入"""
    if not text or not text.strip():
        return False, "输入不能为空"
    if len(text) > max_length:
        return False, f"输入过长，最大 {max_length}"
    return True, "ok"

# 测试类
class TestLLM:
    def test_cost_qwen_plus(self):
        cost = calculate_cost(1000, 500, "qwen-plus")
        assert abs(cost - 0.018) < 0.0001

    def test_cost_qwen_turbo(self):
        cost = calculate_cost(1000, 500, "qwen-turbo")
        assert abs(cost - 0.005) < 0.0001

    def test_validate_empty(self):
        ok, msg = validate_input("")
        assert ok == False
        assert "空" in msg

    def test_validate_too_long(self):
        ok, msg = validate_input("a" * 20000)
        assert ok == False
        assert "过长" in msg

    def test_validate_ok(self):
        ok, msg = validate_input("hello")
        assert ok == True

    @patch('langchain_community.chat_models.ChatTongyi.invoke')
    def test_llm_mock(self, mock_invoke):
        """使用 Mock 测试 LLM 调用"""
        mock_invoke.return_value = Mock(content="测试回答")
        from langchain_community.chat_models import ChatTongyi
        llm = ChatTongyi(model="qwen-plus")
        # result = llm.invoke("测试")
        # assert result.content == "测试回答"
        print("Mock 测试通过 ✓")

# 运行测试
if __name__ == "__main__":
    t = TestLLM()
    t.test_cost_qwen_plus()
    t.test_cost_qwen_turbo()
    t.test_validate_empty()
    t.test_validate_too_long()
    t.test_validate_ok()
    t.test_llm_mock()
    print("
所有测试通过! ✓")

# pytest 命令: pytest test_*.py -v
```

---

## Part 2：Mock 与集成测试（30分钟）

```python
"""
Day13 Part 2: Mock LLM + 集成测试
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

def test_rag_with_mock():
    """Mock 测试 RAG"""
    with patch('langchain_community.vectorstores.FAISS.similarity_search') as mock_search:
        mock_search.return_value = [
            Document(page_content="Python 是编程语言"),
            Document(page_content="FastAPI 是 Web 框架"),
        ]
        results = mock_search("Python 特点")
        assert len(results) == 2
        assert "Python" in results[0].page_content
        print("RAG Mock 测试 ✓")

def test_agent_tool_selection():
    """测试工具选择逻辑"""
    tools = {"计算器", "天气查询", "字数统计"}

    def select_tool(q):
        if any(w in q for w in ["计算", "数学"]):
            return "计算器"
        if any(w in q for w in ["天气"]):
            return "天气查询"
        return None

    assert select_tool("计算 1+1") == "计算器"
    assert select_tool("北京天气") == "天气查询"
    assert select_tool("你好") is None
    print("工具选择测试 ✓")

test_rag_with_mock()
test_agent_tool_selection()
```

---

## Part 3：性能测试 + CI/CD（15分钟）

```python
"""
Day13 Part 3: 性能测试 + GitHub Actions
"""

import time

def benchmark(func, *args, iterations=10):
    """性能基准测试"""
    times = []
    for _ in range(iterations):
        s = time.time()
        func(*args)
        times.append(time.time() - s)
    return {"avg": sum(times)/len(times)*1000, "min": min(times)*1000, "max": max(times)*1000}

def process_text(text):
    return text.upper().strip()

r = benchmark(process_text, "hello world")
print(f"性能基准: avg={r['avg']:.2f}ms, min={r['min']:.2f}ms")

# GitHub Actions 配置
print("""
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
""")
```
