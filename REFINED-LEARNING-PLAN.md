# 🚀 LLM 应用开发学习计划（细化版）v2.0

> **创建日期**: 2026-03-14
> **更新说明**: 本计划为细化版本，所有后续推送的学习计划均以本版本为基础进行细化。
> **版本约定**: LangChain 使用 **1.0+ 版本**，所有代码示例均使用 1.0+ API（LCEL、RunnableWithMessageHistory、JsonOutputParser 等）。
> **GitHub 仓库**: https://github.com/liang8065/llm-learning-journey

---

## 📋 计划总览

- **学习周期**: 12周（84天），2026-03-12 ~ 2026-06-03
- **每日投入**: 1-2小时
- **目标**: 3个月内成为 LLM 应用开发专家
- **学习者**: 亮亮
- **当前进度**: Day 2 已完成，Day 3 起使用本细化计划

---

## 🎯 学习阶段划分

### 阶段一：基础奠基期（第1-2周，Day 1-14）
**目标**: 掌握核心概念、环境搭建、Prompt 工程、LangChain 基础

### 阶段二：技能提升期（第3-4周，Day 15-28）
**目标**: LangChain 深度实践、RAG 入门、Agent 开发基础

### 阶段三：项目实战期（第5-8周，Day 29-56）
**目标**: 独立项目开发、微调与部署、性能优化

### 阶段四：专家进阶期（第9-12周，Day 57-84）
**目标**: 行业前沿探索、完整项目实战、专家级应用开发

---

## 📅 详细学习计划

---

## 阶段一：基础奠基期（Day 1-14）

### Week 1（Day 1-7）：核心概念 + 环境搭建

#### ✅ Day 1（2026-03-12）— 已完成
- [x] 环境搭建（Python 3.10+、VS Code、API Key）
- [x] 核心概念学习（LLM、Token、Prompt、Embedding、RAG）
- [x] 第一个API调用（通义千问）
- [x] 入门文档阅读与笔记整理

#### ✅ Day 2（2026-03-13）— 已完成
- [x] Prompt Engineering 深入学习（Few-shot、CoT、System Prompt）
- [x] LangChain 1.0+ 入门（Chain、Memory、OutputParser）
- [x] 名著改写器测试（已部署服务测试）
- [x] 学习笔记 & GitHub 更新

#### Day 3（2026-03-14）— LangChain 1.0+ 深入 + RAG 基础
**时间**: ~1.5小时（90分钟）

**拆解任务**：✅ 已完成（见 [week-01/day-03/Day3-学习内容-详细拆解.md](week-01/day-03/Day3-学习内容-详细拆解.md)）

| 时间段 | 任务 | 详细内容 | 时长 | 拆解步骤 |
|--------|------|----------|------|----------|
| Part 0 | **拆解任务** | 将今日学习内容拆解为具体步骤，明确输入、操作、输出、检查点 | 15分钟 | 见下方拆解说明 |
| Part 1 | LangChain 1.0+ LCEL 深度实践 | 学习 LCEL 表达式链的高级用法：RunnableParallel、RunnableLambda、RunnableBranch | 30分钟 | Step 1.1 → 1.2 → 1.3 |
| Part 2 | RAG（检索增强生成）基础 | 理解 RAG 原理、文档加载器、文本分割器、向量存储基础 | 30分钟 | Step 2.1 → 2.2 → 2.3 |
| Part 3 | 实战：构建简单 RAG 问答系统 | 使用 LangChain 1.0+ 构建一个基于文档的问答系统 | 30分钟 | Step 3.1 → 3.2 → 3.3 |
| Part 4 | GitHub 更新 & 学习笔记 | 更新 learning-log.md，推送细化学习计划，记录拆解结果 | 15分钟 | Step 4.1 → 4.2 → 4.3 |

**拆解说明**：
- **拆解目的**：将每个学习任务拆解为具体、可执行的步骤，明确输入、操作、输出和检查点
- **拆解原则**：具体性（有明确操作和输出）、可衡量（有检查点）、时间可控（有预估时长）、代码导向（包含可运行代码）
- **拆解输出**：保存到 `week-NN/day-DD/DayDD-学习内容-详细拆解.md`
- **拆解时机**：推送当天学习计划时同步完成拆解

**代码重点**：
```python
# LangChain 1.0+ LCEL 高级示例
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 1.0+ 方式：使用 RunnableParallel 并行处理
llm = ChatTongyi(model="qwen-plus", dashscope_api_key="your-key")
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# 创建向量存储（1.0+ 方式）
vectorstore = FAISS.from_texts(["文档内容..."], embeddings)
retriever = vectorstore.as_retriever()

# LCEL 链式组合
chain = RunnableParallel(
    {"context": retriever, "question": lambda x: x}
) | prompt | llm
```

#### Day 4（2026-03-15）— RAG 实战深化 + 向量数据库
**时间**: ~1.5小时（90分钟）

**拆解任务**：✅ 按约定推送计划时同步完成拆解（拆解文档：`week-01/day-04/Day4-学习内容-详细拆解.md`）

| 时间段 | 任务 | 详细内容 | 时长 | 拆解步骤 |
|--------|------|----------|------|----------|
| Part 0 | **拆解任务** | 将今日学习内容拆解为具体步骤，明确输入、操作、输出、检查点 | 15分钟 | 见拆解文档 |
| Part 1 | 向量数据库深入 | FAISS、ChromaDB、Milvus 对比，选择合适方案 | 30分钟 | Step 1.1 → 1.2 → 1.3 |
| Part 2 | 文档加载与处理 | PDF、Markdown、网页等多格式文档加载与分割 | 30分钟 | Step 2.1 → 2.2 → 2.3 |
| Part 3 | 实战：构建知识库问答系统 | 完整 RAG 系统：文档入库 → 检索 → 生成 | 30分钟 | Step 3.1 → 3.2 → 3.3 |
| Part 4 | 性能优化 | 检索质量评估、chunk size 优化、embedding 选择 | 15分钟 | Step 4.1 → 4.2 → 4.3 |

#### Day 5（2026-03-16）— Agent 开发基础
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | Agent 概念理解 | 什么是 Agent、ReAct 模式、工具调用原理 | 30分钟 |
| Part 2 | LangChain 1.0+ Agent 实践 | 使用 create_react_agent 或 create_openai_functions_agent | 30分钟 |
| Part 3 | 工具开发实战 | 自定义 Tool：计算器、搜索工具、API 调用 | 30分钟 |
| Part 4 | Agent 测试与调试 | 调试 Agent 行为，理解推理链 | 15分钟 |

**代码重点**：
```python
# LangChain 1.0+ Agent 示例
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

llm = ChatTongyi(model="qwen-plus", dashscope_api_key="your-key")
tools = [calculate]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "计算 123 * 456"})
```

#### Day 6（2026-03-17）— 微调与部署基础
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | 微调基础概念 | LoRA、QLoRA、SFT 原理 | 30分钟 |
| Part 2 | 微调实战准备 | 数据集准备、评估方法、训练配置 | 30分钟 |
| Part 3 | 部署方案了解 | vLLM、Ollama、云服务部署对比 | 30分钟 |
| Part 4 | 项目优化方案制定 | 针对名著改写器的优化方案 | 15分钟 |

#### Day 7（2026-03-18）— Week 1 总结 + 项目实战
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | Week 1 总结回顾 | 整理本周学习笔记，查漏补缺 | 30分钟 |
| Part 2 | 名著改写器优化实施 | 根据 Day 6 制定的方案进行优化 | 30分钟 |
| Part 3 | 实战测试 | 全面测试优化后的名著改写器 | 30分钟 |
| Part 4 | Week 2 计划调整 | 根据 Week 1 进展调整 Week 2 计划 | 15分钟 |

---

### Week 2（Day 8-14）：LangChain 深度 + RAG 进阶

#### Day 8（2026-03-19）— LangChain 1.0+ 高级特性
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | RunnableWithMessageHistory 高级用法 | 多会话管理、持久化存储 | 30分钟 |
| Part 2 | OutputParser 高级实践 | 结构化输出、Pydantic 模型验证 | 30分钟 |
| Part 3 | 链的组合与嵌套 | 复杂工作流设计 | 30分钟 |
| Part 4 | 实战练习 | 构建多轮对话系统 | 15分钟 |

#### Day 9（2026-03-20）— RAG 系统优化
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | 检索策略优化 | 相似度搜索、混合检索、重排序 | 30分钟 |
| Part 2 | Chunking 策略 | 递归分割、语义分割、固定大小分割 | 30分钟 |
| Part 3 | 实战：优化知识库问答 | 应用优化策略提升回答质量 | 30分钟 |
| Part 4 | 评估与迭代 | 建立评估指标，持续改进 | 15分钟 |

#### Day 10（2026-03-21）— 多模态 LLM 应用
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | 多模态模型原理 | GPT-4V、Qwen-VL 等视觉模型介绍 | 30分钟 |
| Part 2 | 图像理解实战 | 使用 Qwen-VL 进行图片分析 | 30分钟 |
| Part 3 | 多模态 RAG | 结合图像和文本的检索增强生成 | 30分钟 |
| Part 4 | 应用场景探索 | 图文问答、图像描述生成等 | 15分钟 |

#### Day 11（2026-03-22）— 工作流编排与自动化
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | LangGraph 入门 | 图结构工作流、状态管理 | 30分钟 |
| Part 2 | 复杂工作流设计 | 条件分支、循环、并行处理 | 30分钟 |
| Part 3 | 实战：自动化学习助手 | 构建一个自动整理学习资料的 Agent | 30分钟 |
| Part 4 | 部署与集成 | 将工作流集成到实际应用 | 15分钟 |

#### Day 12（2026-03-23）— API 设计与项目架构
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | API 设计原则 | RESTful API 设计、错误处理、认证 | 30分钟 |
| Part 2 | 项目架构设计 | 模块化设计、依赖注入、配置管理 | 30分钟 |
| Part 3 | 实战：重构名著改写器 | 应用新架构重新设计项目 | 30分钟 |
| Part 4 | 文档编写 | API 文档、使用说明、开发指南 | 15分钟 |

#### Day 13（2026-03-24）— 测试与质量保障
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | 单元测试 | 针对核心功能编写测试用例 | 30分钟 |
| Part 2 | 集成测试 | 端到端测试流程 | 30分钟 |
| Part 3 | 性能测试 | 响应时间、并发处理能力 | 30分钟 |
| Part 4 | 持续集成 | GitHub Actions 自动化测试 | 15分钟 |

#### Day 14（2026-03-25）— Week 2 总结 + 阶段回顾
**时间**: ~1.5小时

| 时间段 | 任务 | 详细内容 | 时长 |
|--------|------|----------|------|
| Part 1 | Week 2 总结 | 整理本周学习笔记 | 30分钟 |
| Part 2 | 阶段一回顾 | 检验基础奠基期目标完成情况 | 30分钟 |
| Part 3 | 阶段二计划 | 制定技能提升期详细计划 | 30分钟 |
| Part 4 | GitHub 完整推送 | 推送所有更新到 GitHub | 15分钟 |

---

## 阶段二：技能提升期（Day 15-28）

### Week 3（Day 15-21）：LangChain 深度实践 + Agent 进阶

#### Day 15（2026-03-26）— 复杂 Agent 系统设计
**时间**: ~1.5小时
- 多 Agent 协作系统
- 角色分配与任务分解
- Agent 间通信机制
- 实战：构建学习小组 Agent 系统

#### Day 16（2026-03-27）— 工具生态扩展
**时间**: ~1.5小时
- 自定义工具开发深入
- 外部 API 集成（搜索引擎、数据库、文件系统）
- 工具安全与权限管理
- 实战：构建一个全能工具 Agent

#### Day 17（2026-03-28）— 流式输出与实时交互
**时间**: ~1.5小时
- 流式响应原理
- LangChain 1.0+ 流式处理
- WebSocket 实时通信
- 实战：实时对话应用

#### Day 18（2026-03-29）— 高级 Prompt 技术
**时间**: ~1.5小时
- Prompt 模板化与版本管理
- 动态 Prompt 生成
- Prompt 评估与优化
- 实战：构建 Prompt 工程平台

#### Day 19（2026-03-30）— LangChain 1.0+ 高级组件
**时间**: ~1.5小时
- 自定义 Runnable 组件
- 错误处理与重试机制
- 缓存策略
- 实战：构建高可用应用

#### Day 20（2026-03-31）— Agent 安全与可靠性
**时间**: ~1.5小时
- Agent 安全风险分析
- 输入验证与输出过滤
- 沙箱环境与权限控制
- 实战：加固 Agent 系统

#### Day 21（2026-04-01）— Week 3 总结 + 项目中期检查
**时间**: ~1.5小时
- Week 3 总结回顾
- 项目进度评估
- 下一阶段计划调整
- GitHub 推送

---

### Week 4（Day 22-28）：RAG 进阶 + 部署实践

#### Day 22（2026-04-02）— 高级 RAG 技术
**时间**: ~1.5小时
- HyDE（假设性文档嵌入）
- Multi-query 检索
- Self-RAG 与反思机制
- 实战：构建企业级 RAG 系统

#### Day 23（2026-04-03）— 向量数据库深度优化
**时间**: ~1.5小时
- 大规模向量索引优化
- 混合检索策略
- 实时更新与增量索引
- 实战：百万级文档检索优化

#### Day 24（2026-04-04）— 部署方案实践
**时间**: ~1.5小时
- Docker 容器化部署
- 云服务部署（阿里云/腾讯云）
- CI/CD 流水线搭建
- 实战：将 RAG 系统部署上线

#### Day 25（2026-04-05）— 性能监控与日志
**时间**: ~1.5小时
- 应用性能监控（APM）
- 日志收集与分析
- 告警与通知
- 实战：构建监控体系

#### Day 26（2026-04-06）— 微调实践准备
**时间**: ~1.5小时
- 数据集构建与清洗
- 微调框架对比（Axolotl、LLaMA-Factory）
- 评估指标设计
- 实战：准备一个定制化微调项目

#### Day 27（2026-04-07）— 微调实施
**时间**: ~1.5小时
- LoRA 微调实战
- 训练监控与调整
- 模型评估
- 实战：完成一个微调任务

#### Day 28（2026-04-08）— Week 4 总结 + 阶段二回顾
**时间**: ~1.5小时
- Week 4 总结
- 阶段二全面回顾
- 阶段三计划制定
- GitHub 推送

---

## 阶段三：项目实战期（Day 29-56）

### Week 5-6（Day 29-42）：独立项目一

**项目**: 智能学习助手系统
- 基于 RAG 的学习资料问答
- 个性化学习计划生成
- 学习进度追踪与反馈
- 多模态学习支持（图文理解）

**技术栈**:
- LangChain 1.0+（核心框架）
- Qwen-VL（多模态）
- FAISS/ChromaDB（向量存储）
- Flask/FastAPI（API 层）
- Docker（部署）

**每日安排**（Day 29-42）:
- Day 29-31: 需求分析、系统设计、技术选型
- Day 32-34: 核心功能开发（RAG 问答、计划生成）
- Day 35-37: 多模态集成、UI 开发
- Day 38-40: 测试、优化、部署
- Day 41-42: 文档、总结、复盘

### Week 7-8（Day 43-56）：独立项目二

**项目**: AI 内容创作平台
- 多风格文本生成
- 图文混排内容创作
- 版本管理与协作
- API 开放平台

**技术栈**:
- LangChain 1.0+（核心框架）
- Agent（内容生成 Agent）
- RAG（知识检索）
- 向量数据库（内容索引）
- Next.js / Vue（前端）

**每日安排**（Day 43-56）:
- Day 43-46: 需求分析、系统设计、技术选型
- Day 47-50: 核心功能开发（生成引擎、Agent 编排）
- Day 51-53: 前端开发、API 设计
- Day 54-56: 测试、优化、部署、文档

---

## 阶段四：专家进阶期（Day 57-84）

### Week 9-10（Day 57-70）：行业前沿探索

**学习主题**:
- 多 Agent 协作框架（CrewAI、AutoGen）
- LLM 推理优化（vLLM、SGLang）
- 模型量化与压缩（GGUF、AWQ）
- 边缘部署与移动端优化
- 大规模 RAG 系统架构

**每日安排**:
- 每日学习一个前沿技术
- 阅读论文/文档
- 动手实验
- 记录学习笔记

### Week 11-12（Day 71-84）：专家级项目实战

**项目**: 企业级 LLM 应用平台
- 完整的多 Agent 协作系统
- 企业级 RAG 知识库
- 微调模型集成
- 生产级部署方案
- 监控、日志、告警体系

**最终目标**:
- 完成一个生产级 LLM 应用
- 掌握从设计到部署的完整流程
- 达到专家级应用开发水平

---

## 🔧 技术规范

### LangChain 版本规范
- **必须使用 LangChain 1.0+ 版本**
- 所有代码示例使用 LCEL（LangChain Expression Language）
- 模型调用使用 `invoke()` 而非 `run()`
- Memory 使用 `RunnableWithMessageHistory` 而非 `ConversationChain`
- OutputParser 使用 `JsonOutputParser` 或 `PydanticOutputParser`

### 代码规范
- Python 3.10+
- 类型注解
- 完整的错误处理
- 清晰的代码注释

### 文档规范
- 所有学习笔记包含：日期、目标、完成情况、收获与反思
- 代码示例必须可运行
- 包含信息来源和参考链接

---

## 📊 学习进度追踪

### 追踪指标
- 每日任务完成率
- 累计学习时长
- 项目完成数量
- GitHub 提交次数

### 进度文件
- `learning-log.md` — 每日学习日志
- `memory/learning-progress.json` — 进度数据
- `REFINED-LEARNING-PLAN.md` — 本文件（细化学习计划）

---

## 📝 约定与承诺

### 1. LangChain 版本约定
- 所有 LangChain 相关内容使用 **1.0+ 版本**
- 代码示例、教程、文档均遵循此规范
- 如遇版本问题，优先查阅官方 1.0+ 文档

### 2. 学习计划推送约定
- 所有学习计划推送前必须**细化**（本文件为基础）
- 推送内容必须包含详细的时间安排、任务分解、代码示例
- 推送到 GitHub 仓库：`liang8065/llm-learning-journey`
- **拆解任务**：推送当天学习计划时，同步完成该计划的详细拆解（见下方拆解规则）

### 3. 拆解任务约定（新增）
- **拆解时机**：推送当天学习计划时，同步完成拆解
- **拆解内容**：将每个学习任务拆解为具体步骤，明确：
  - **输入**：需要准备的材料/知识
  - **操作**：具体执行的动作（含代码示例）
  - **输出**：预期的结果/产物
  - **检查点**：验证完成的标准
- **拆解输出**：保存到 `week-NN/day-DD/DayDD-学习内容-详细拆解.md`
- **拆解任务本身**：作为每日任务的 Part 0（第一项），计入完成率
- **后续约定**：以后每次推送学习计划，都必须同步完成拆解

### 4. GitHub 管理约定
- 每日学习结束后更新 GitHub
- 保持仓库结构清晰，按周/日组织
- 重要更新附带 commit message 说明

### 5. 学习节奏约定
- 每日投入 1-2 小时
- 早上 9 点推送当日学习计划（含拆解）
- 每天结束汇报进展
- 根据进展调整次日计划

---

## 🔗 核心资源

### 官方文档
- [LangChain 1.0+ 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [LangChain 1.0+ API 参考](https://api.python.langchain.com/en/latest/)
- [通义千问 API](https://help.aliyun.com/zh/dashscope/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/zh)

### 学习资源
- [LangChain 中文教程](https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide)
- [Hugging Face 课程](https://huggingface.co/learn)
- [OpenAI Cookbook](https://cookbook.openai.com/)

### 项目参考
- [经典改写器项目](classic-rewriter/)
- [GitHub 仓库](https://github.com/liang8065/llm-learning-journey)

---

*文档版本: v2.1 (细化版 + 拆解任务约定)*
*创建时间: 2026-03-14*
*更新时间: 2026-03-14 10:43*
*创建者: OpenClaw AI 助手*
*下次更新: 根据学习进展动态调整*

---

## 📋 每日计划模板（含拆解任务）

> **重要**：所有每日计划必须包含 Part 0（拆解任务）作为第一项。

### 每日计划结构

| 时间段 | 任务 | 详细内容 | 时长 | 拆解步骤 |
|--------|------|----------|------|----------|
| **Part 0** | **拆解任务** | 将今日学习内容拆解为具体步骤，明确输入、操作、输出、检查点 | 15分钟 | 见拆解文档 |
| Part 1 | [任务1] | [详细内容] | [时长] | Step 1.1 → 1.2 → 1.3 |
| Part 2 | [任务2] | [详细内容] | [时长] | Step 2.1 → 2.2 → 2.3 |
| Part 3 | [任务3] | [详细内容] | [时长] | Step 3.1 → 3.2 → 3.3 |
| Part 4 | GitHub 更新 & 学习笔记 | 更新日志，推送计划，记录拆解结果 | 15分钟 | Step 4.1 → 4.2 → 4.3 |

### 拆解任务输出格式

拆解文档保存位置：`week-NN/day-DD/DayDD-学习内容-详细拆解.md`

拆解文档应包含：
1. **今日任务总览**（表格）
2. **每个 Part 的详细拆解**（步骤、输入、操作、输出、检查点）
3. **代码示例**（可运行）
4. **完成检查清单**
5. **今日产出清单**

### 拆解任务完成标准

- [ ] 每个学习任务都有明确的步骤
- [ ] 每个步骤都有输入、操作、输出、检查点
- [ ] 包含可运行的代码示例
- [ ] 有完成检查清单
- [ ] 文档已保存到正确位置
- [ ] 已更新学习日志
