# 📚 Day 21 — Week 3 总结 + 项目中期检查

> **周期**：Day 21 / Week 3
> **目标**：总结 Week 3，评估项目进度
> **预计用时**：1.5 小时

---

## Part 1：Week 3 知识回顾（30分钟）

```python
"""
Day21 Part 1: Week 3 回顾
"""

week3 = {
    "Day 15": ("复杂 Agent 系统", ["多 Agent 协作", "角色分配", "任务分解"]),
    "Day 16": ("工具生态扩展", ["高级工具开发", "外部 API", "安全"]),
    "Day 17": ("流式输出", ["流式响应", "WebSocket/SSE", "实时对话"]),
    "Day 18": ("高级 Prompt", ["模板化", "版本管理", "动态生成", "评估"]),
    "Day 19": ("高级组件", ["自定义 Runnable", "错误处理", "缓存"]),
    "Day 20": ("Agent 安全", ["输入验证", "权限控制", "限流", "输出过滤"]),
}

print("=== Week 3 回顾 ===
")
for day, (t, k) in week3.items():
    print(f"{day}: {t}")
    print(f"  → {', '.join(k)}
")

quiz = [
    ("多 Agent 怎么协作?", "每个 Agent 负责不同角色，通过 StateGraph 传递数据"),
    ("流式输出优势?", "降低首字等待时间，提升用户体验"),
    ("Agent 安全风险?", "Prompt 注入、工具滥用、信息泄漏、无限循环"),
    ("如何防止 Prompt 注入?", "输入验证+内容过滤+输出脱敏"),
]

print("
=== 自检 ===")
for i, (q, a) in enumerate(quiz, 1):
    print(f"Q{i}: {q}")
    print(f"A: {a}
")
```

---

## Part 2：项目进度评估（30分钟）

```python
"""
Day21 Part 2: 项目进度
"""

projects = {
    "名著改写器": {
        "状态": "运行中",
        "完成": ["环境搭建", "API 调用", "8种风格", "Prompt 优化"],
        "待做": ["流式输出", "Docker 部署", "测试覆盖"],
    },
    "学习知识库": {
        "状态": "已构建",
        "完成": ["Day3-4 RAG 系统", "知识库问答"],
        "待做": ["多格式文档加载", "检索优化", "评估体系"],
    },
}

print("=== 项目进度 ===
")
for name, info in projects.items():
    print(f"【{name}】{info['状态']}")
    print(f"  ✓ {', '.join(info['完成'])}")
    print(f"  → {', '.join(info['待做'])}
")

# 能力自评
skills = {
    "Prompt Engineering": 80,
    "LangChain 1.0+": 75,
    "RAG 实战": 70,
    "Agent 开发": 65,
    "测试": 60,
    "部署": 50,
    "微调": 30,
}

print("=== 能力雷达 ===")
for skill, lvl in skills.items():
    bar = "█" * (lvl // 5) + "░" * (20 - lvl // 5)
    print(f"  {skill:>18} [{bar}] {lvl}%")
```

---

## Part 3：阶段二总结与下一阶段（15分钟）

```python
"""
Day21 Part 3: 阶段二总结
"""

phase2_summary = "阶段二（Week 3-4）涵盖了 Agent 进阶、工作流编排、流式输出、高级 Prompt、安全加固、高级 RAG 技术和部署实践。至此，基础知识和技能全面提升。"
print(phase2_summary)

phase3 = {
    "Week 5-6": "独立项目一: 智能学习助手系统 (RAG + Agent)",
    "Week 7-8": "独立项目二: AI 内容创作平台",
    "Week 9-10": "行业前沿: 多 Agent、推理优化、量化",
    "Week 11-12": "专家项目: 企业级 LLM 应用平台",
}

print("
=== 阶段三/四预览 ===")
for wk, desc in phase3.items():
    print(f"  {wk}: {desc}")
```
