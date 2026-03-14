# 📖 学习日志 - LLM 应用开发之旅

> 记录每一天的学习进展、收获和反思
> **重要约定**: 所有学习计划推送前必须细化，LangChain 使用 1.0+ 版本。详见 [REFINED-LEARNING-PLAN.md](REFINED-LEARNING-PLAN.md)

---

## Day 1 (2026-03-12) 🎯

### 📋 今日任务
1. **环境搭建** (30分钟) — 安装 Python 3.10+、VS Code、配置 API Key
2. **核心概念学习** (30分钟) — 了解 LLM、Token、Prompt、Embedding、RAG 基本概念
3. **第一个API调用** (30分钟) — 用 Python 跑通 OpenAI API / 通义千问 API
4. **阅读 & 记录** (15分钟) — 看一份入门文档，整理笔记

### ✅ 完成情况
- [x] 环境搭建 — ✅ 已完成（Python 3.10+, VS Code, API Key）
- [x] 核心概念学习 — ✅ 已完成（LLM、Token、Prompt、Embedding、RAG）
- [x] 第一个API调用 — ✅ 已完成（通义千问）
- [x] 阅读与记录 — ✅ 已完成

### 📝 今日笔记
- 完整学习内容已整理为独立文件: [week-01/day-01/Day1-学习内容.md](week-01/day-01/Day1-学习内容.md)
- 学习内容包含：环境搭建详解、核心概念完整解释（LLM/Token/Prompt/Embedding/RAG）、OpenAI和通义千问API完整代码示例
- 所有内容自包含，无需额外参考其他材料

### 💡 收获与反思
- Day 1 是入门第一天，重点在于打好基础
- 已完成GitHub项目搭建，学习内容已上传
- **亮亮于 19:27 完成Day 1全部任务（4/4）**
- API调用使用通义千问，国产模型完全够用
- 明天（Day 2）将根据进展调整学习计划

### ⏰ 用时统计
- 计划用时：~1.5小时
- 实际用时：（待记录）

### 🔗 相关文件
- 完整学习内容: [week-01/day-01/Day1-学习内容.md](week-01/day-01/Day1-学习内容.md)
- LLM日报 (2026-03-12): [reports/2026-03-12-llm-daily-report.md](reports/2026-03-12-llm-daily-report.md)
- LLM学习资源大全: [resources/llm-learning-resources.md](resources/llm-learning-resources.md)
- GitHub仓库: https://github.com/liang8065/llm-learning-journey

---

## Day 2 (2026-03-13) 🎯

### 📋 今日任务
1. **Prompt Engineering 深入学习** (30分钟) — Few-shot、Chain-of-Thought、System Prompt等高级技巧
2. **LangChain 1.0+ 入门** (30分钟) — Chain、Memory、OutputParser核心概念（使用1.0+ API）
3. **名著改写器测试与优化** (30分钟) — 测试已部署服务，记录改进点
4. **学习笔记 & GitHub 更新** (15分钟) — 整理笔记，更新仓库

### ✅ 完成情况
- [x] Prompt Engineering深入学习 — ✅ 已完成（Few-shot、CoT、System Prompt）
- [x] LangChain 1.0+ 入门 — ✅ 已完成（Chain、Memory、OutputParser，使用LCEL）
- [x] 名著改写器测试 — ⏳ 待完成（用户将进行实际测试）
- [x] 学习笔记 & GitHub更新 — ✅ 已完成（已上传Day 2内容）

### 📝 今日笔记
- 完整学习内容已整理为独立文件: [week-01/day-02/Day2-学习内容.md](week-01/day-02/Day2-学习内容.md)
- 学习内容包含：Prompt Engineering详解（Few-shot/CoT/System Prompt）、LangChain 1.0+核心概念（LCEL/RunnableWithMessageHistory/JsonOutputParser）、名著改写器测试方案
- Day 2重点：深化Prompt技能 → 掌握LangChain 1.0+ → 实战验证项目
- **LangChain 1.0+ 迁移完成**: LLMChain → LCEL, ConversationChain → RunnableWithMessageHistory, StructuredOutputParser → JsonOutputParser + Pydantic

### 💡 收获与反思
- Day 2 在Day 1基础上深化Prompt工程和框架理解
- LangChain 1.0+ 是LLM应用开发的核心框架，需要持续实践
- 名著改写器项目是一个很好的实战案例，边学边用效果更好
- **重要约定**: 以后所有学习计划推送前必须细化，LangChain 使用 1.0+ 版本
- 明天（Day 3）将继续LangChain 1.0+实践，探索RAG基础

### ⏰ 用时统计
- 计划用时：~1.75小时
- 实际用时：（待记录）

### 🔗 相关文件
- 完整学习内容: [week-01/day-02/Day2-学习内容.md](week-01/day-02/Day2-学习内容.md)
- **细化学习计划**: [REFINED-LEARNING-PLAN.md](REFINED-LEARNING-PLAN.md)
- GitHub仓库: https://github.com/liang8065/llm-learning-journey
- 名著改写器服务: http://47.102.222.123:5000

---

## Day 3 (2026-03-14) 🎯

### 📋 今日任务
0. **拆解任务** (15分钟) — 将今日学习内容拆解为具体步骤，明确输入、操作、输出、检查点 ✅
1. **LangChain 1.0+ LCEL 深度实践** (30分钟) — RunnableParallel、RunnableLambda、RunnableBranch
2. **RAG（检索增强生成）基础** (30分钟) — 理解RAG原理、文档加载器、文本分割器、向量存储基础
3. **实战：构建简单RAG问答系统** (30分钟) — 使用LangChain 1.0+构建基于文档的问答系统
4. **GitHub 更新 & 学习笔记** (15分钟) — 更新learning-log.md，推送细化学习计划，记录拆解结果

### ✅ 完成情况
- [x] 拆解任务 — ✅ 已完成（详细拆解见 [week-01/day-03/Day3-学习内容-详细拆解.md](week-01/day-03/Day3-学习内容-详细拆解.md)）
- [ ] LangChain 1.0+ LCEL 深度实践
- [ ] RAG 基础学习
- [ ] 实战：RAG问答系统
- [ ] GitHub 更新 & 学习笔记

### 📝 今日笔记
- **拆解任务完成**：已将 Day 3 学习内容拆解为详细步骤，包括：
  - Part 1: LCEL 深度实践（3步：理解基础 → RunnableParallel → RunnableLambda）
  - Part 2: RAG 基础（3步：原理学习 → 文档加载器 → 文本分割器）
  - Part 3: RAG 问答系统实战（3步：准备向量存储 → 构建RAG链 → 测试系统）
  - Part 4: GitHub 更新（3步：整理笔记 → 更新日志 → 推送）
- 每个拆解步骤都有：输入、操作（含代码）、输出、检查点
- **拆解约定**：以后每次推送学习计划时，同步完成拆解，拆解作为 Part 0（第一项任务）

### 💡 收获与反思
- 拆解任务让学习计划更清晰、可执行
- 用户要求：拆解动作加入任务计划，推送计划时同步完成拆解
- 这是长期约定，需要严格执行

### ⏰ 用时统计
- 计划用时：~1.5小时（含拆解15分钟）
- 实际用时：（待记录）

### 🔗 相关文件
- **详细拆解**: [week-01/day-03/Day3-学习内容-详细拆解.md](week-01/day-03/Day3-学习内容-详细拆解.md)
- **细化学习计划**: [REFINED-LEARNING-PLAN.md](REFINED-LEARNING-PLAN.md)
- GitHub仓库: https://github.com/liang8065/llm-learning-journey

---

## 模板：后续日志格式

### Day N (YYYY-MM-DD) 🎯

**今日任务**
1. 任务1
2. 任务2
3. 任务3

**完成情况**
- [ ] 任务1
- [ ] 任务2
- [ ] 任务3

**今日笔记**
- 要点1
- 要点2

**收获与反思**
- 学到了什么
- 遇到了什么问题
- 如何改进

**用时统计**
- 实际用时：X小时

**重要约定**
- 所有学习计划推送前必须细化
- LangChain 使用 1.0+ 版本
- 推送到 GitHub 仓库: liang8065/llm-learning-journey
