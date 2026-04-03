# 📚 Day 6 — 微调与部署基础

> **周期**：Day 6 / Week 1
> **目标**：理解微调原理和部署方案
> **预计用时**：1.5 小时

---

## Part 1：微调基础概念（30分钟）

### 微调方法对比

| 方法 | 参数量 | 显存 | 效果 | 适用场景 |
|------|--------|------|------|----------|
| 全量微调 | 100% | 80GB+ | 最高 | 资源充足 |
| LoRA | ~1% | 8-16GB | 接近全量 | 大多数场景 |
| QLoRA | ~0.5% | 4-8GB | 稍低于 LoRA | 单卡用户 |

**LoRA 原理**：不改原始权重 W，添加低秩矩阵 A×B 作为增量更新。参数大幅减少但效果接近全量微调。

```python
"""
Day6 Part 1: LoRA 原理 + PEFT 配置
依赖安装:
    pip install peft transformers torch numpy
"""

import numpy as np

# LoRA 原理演示
original_W = np.random.randn(768, 768)
r = 8
A = np.random.randn(r, 768) * 0.01
B = np.random.randn(768, r) * 0.01

delta_W = B @ A
W_new = original_W + delta_W

print(f"原始权重参数: {original_W.size:,}")
print(f"LoRA 新增参数: {A.size + B.size:,}")
print(f"参数节省: {(1 - (A.size + B.size) / original_W.size) * 100:.2f}%")

# PEFT 配置
try:
    from peft import LoraConfig, TaskType
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    print(f"\nLoRA 配置:")
    print(f"  r = {lora_config.r}")
    print(f"  alpha = {lora_config.lora_alpha}")
    print(f"  target_modules = {lora_config.target_modules}")
except ImportError:
    print("\n安装: pip install peft transformers torch")
```

---

## Part 2：部署方案对比（30分钟）

```python
"""
Day6 Part 2: 部署方案详解
"""

solutions = {
    "Ollama": {"特点": "一键启动，支持多模型", "安装": "curl -fsSL url | sh", "使用": "ollama run qwen2.5", "场景": "个人开发"},
    "vLLM": {"特点": "高性能推理，PagedAttention", "安装": "pip install vllm", "场景": "生产 API 服务"},
    "HuggingFace TGI": {"特点": "生产级推理，容器化", "安装": "Docker 运行", "场景": "企业级部署"},
    "云服务": {"特点": "按需付费，免运维", "场景": "不想运维的公司"},
}

print("=== 部署方案对比 ===\n")
for name, info in solutions.items():
    print(f"【{name}】")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

# OpenAI 兼容 API 调用
print("=== 本地模型 API 调用 ===")
print("""
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "你好"}]
)
""")
```

---

## Part 3：项目优化计划（15分钟）

```python
"""
Day6 Part 3: 名著改写器优化计划
"""

plan = {
    "Prompt 优化": "基于 Day 2 技巧优化风格 Prompt",
    "流式输出": "减少首 Token 等待时间",
    "记忆功能": "支持多轮对话优化",
    "Docker 部署": "一键部署",
}

print("=== 优化计划 ===")
for k, v in plan.items():
    print(f"  • {k}: {v}")
```