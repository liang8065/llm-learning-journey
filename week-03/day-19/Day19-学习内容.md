# 📚 Day 19 — LangChain 高级组件

> **周期**：Day 19 / Week 3
> **目标**：自定义 Runnable、错误处理、缓存策略
> **预计用时**：1.5 小时

---

## Part 1：自定义 Runnable 组件（30分钟）

```python
"""
Day19 Part 1: 自定义 Runnable
依赖安装:
    pip install langchain langchain-community langchain-core dashscope
"""

import os
import time
from cache import CacheManager
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "your-key")
)

# 自定义 Runnable: 字数统计
word_counter = RunnableLambda(lambda x: {"input": x, "word_count": len(x)})

# 自定义 Runnable: 情感判断（简化版）
sentiment = RunnableLambda(lambda x: {
    **x,
    "sentiment": "积极" if any(w in x["input"] for w in ["好", "棒", "开心", "快乐"])
                 else "消极" if any(w in x["input"] for w in ["坏", "差", "难过", "生气"])
                 else "中性"
})

# 链式组合
analysis_chain = word_counter | sentiment | RunnableLambda(
    lambda r: f"输入: {r['input']} ({r['word_count']} 字, 情感: {r['sentiment']})"
)

print("=== 自定义 Runnable ===")
print(analysis_chain.invoke("今天天气真好，心情很愉快"))
print(analysis_chain.invoke("这个产品质量太差了，很失望"))

# 缓存策略
class SimpleCache:
    """简单 LLM 结果缓存"""
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

cache = SimpleCache()

def cached_llm(prompt, messages):
    key = str(messages)
    result = cache.get(key)
    if result:
        print("  [缓存命中]")
        return result
    result = llm.invoke(messages)
    cache.set(key, result)
    return result
```

---

## Part 2：错误处理与重试（30分钟）

```python
"""
Day19 Part 2: 错误处理 + 重试
"""

import time
import functools

def retry(max_retries=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"  重试 {i+1}/{max_retries}: {e}")
                    time.sleep(delay * (i + 1))
            raise Exception(f"失败 {max_retries} 次")
        return wrapper
    return decorator

@retry(max_retries=3, delay=0.5)
def call_llm_with_retry(prompt):
    """带重试的 LLM 调用"""
    return llm.invoke(prompt)

# 超时处理
import signal
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("LLM 调用超时")

def call_with_timeout(func, args, timeout=30):
    """超时控制"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args)
    finally:
        signal.alarm(0)
    return result

print("=== 错误处理 ===")
print("• retry 装饰器: 自动重试失败操作")
print("• 超时控制: 防止 LLM 调用卡死")
print("• 异常捕获: 返回友好错误信息")
```

---

## Part 3：实战 — 高可用 LLM 服务（15分钟）

```python
"""
Day19 Part 3: 高可用 LLM 服务
"""

# 多模型降级策略
class HighAvailabilityLLM:
    """高可用 LLM 客户端"""

    def __init__(self, primary, fallbacks=None):
        self.primary = primary
        self.fallbacks = fallbacks or []

    def invoke(self, messages):
        """自动降级调用"""
        models = [self.primary] + self.fallbacks
        for model in models:
            try:
                result = model.invoke(messages)
                print(f"  ✓ 使用 {type(model).__name__} 成功")
                return result
            except Exception as e:
                print(f"  ✗ {type(model).__name__} 失败: {e}")
        raise Exception("所有模型都失败了")

print("=== 高可用 LLM ===")
print("• 多模型降级: primary -> fallback1 -> fallback2")
print("• 自动重试: 网络错误时重试")
print("• 结果缓存: 相同输入直接返回缓存")
print("• 超时控制: 防止单个请求卡死")
```
