# Day 2: Prompt Engineering 深入 + LangChain 入门 + 实战测试

> **学习者**: 亮亮
> **目标**: 成为LLM应用开发专家（3个月计划）
> **当前阶段**: 入门阶段（Day 2）
> **今日投入**: ~1.75小时
> **前置完成**: Day 1 全部完成（环境搭建、核心概念、API调用）

---

## 今日概览

Day 2 继续强化基础，围绕三个核心方向：
1. **Prompt Engineering 深入** — 掌握Few-shot、Chain-of-Thought等高级技巧
2. **LangChain 入门** — 了解Chain、Memory、OutputParser核心概念
3. **名著改写器测试** — 实战测试已部署的项目，发现改进点

---

## Part 1: Prompt Engineering 深入学习（30分钟）

### 1.1 回顾：什么是Prompt？

Prompt（提示词）是你给LLM的指令，决定了模型的输出质量和方向。昨天你已经了解了基本概念，今天深入几种高级策略。

### 1.2 Few-shot Prompting（少样本提示）

**原理**: 在Prompt中提供几个示例，让模型"学会"你的期望格式和风格。

**为什么有效**: LLM有极强的模式识别能力，几个示例就能让它理解你想要什么。

**示例代码**:

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

# Few-shot示例：让模型学习情感分析
few_shot_prompt = """
你是一个情感分析助手。请分析以下文本的情感倾向。

示例1:
文本: "今天天气真好，心情特别愉快！"
分析: 正面情感，表达快乐和满足。

示例2:
文本: "等了两个小时，外卖还是没送到，气死我了。"
分析: 负面情感，表达愤怒和失望。

示例3:
文本: "今天的工作完成了，但没什么特别的感觉。"
分析: 中性情感，平淡陈述。

现在请分析：
文本: "这部电影的结局让我又哭又笑，太感人了。"
分析:
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": few_shot_prompt}],
    max_tokens=200
)

print(response.choices[0].message.content)
```

**通义千问版本**:

```python
import dashscope

dashscope.api_key = "your-dashscope-key"

response = dashscope.Generation.call(
    model="qwen-plus",
    messages=[
        {"role": "user", "content": few_shot_prompt}
    ],
    max_tokens=200
)

print(response.output.text)
```

**实践练习**: 写3个few-shot示例，让模型帮你：
1. 把口语化文字改写成正式邮件
2. 将长段落总结成3句话
3. 识别文本中的关键实体（人名、地点、时间）

### 1.3 Chain-of-Thought（CoT）思维链

**原理**: 要求模型"一步一步思考"，展示推理过程，而不是直接给出答案。

**为什么有效**: 复杂问题直接回答容易出错，分步推理能显著提升准确性。

**基本用法**:

```python
# 直接提问（不推荐）
prompt_direct = "如果一个班级有30人，男生占60%，女生占多少人？"

# Chain-of-Thought（推荐）
prompt_cot = """
如果一个班级有30人，男生占60%，女生占多少人？

请一步一步推理：
1. 首先，计算男生人数...
2. 然后，计算女生人数...
3. 最后，得出答案。
"""

response = dashscope.Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": prompt_cot}],
    max_tokens=300
)
```

**Zero-shot CoT（无示例思维链）**:

最简单的CoT技巧，只需加上一句魔法咒语：

```python
prompt = """
解决以下问题：{你的问题}

让我们一步一步思考。
"""
```

**进阶：Self-Consistency（自洽性）**:

对于复杂问题，让模型生成多个推理路径，选择最常见的答案：

```python
# 生成3个不同的推理
for i in range(3):
    response = dashscope.Generation.call(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt_cot}],
        temperature=0.7  # 增加多样性
    )
    print(f"推理 {i+1}: {response.output.text}")
```

### 1.4 System Prompt（系统提示词）

**原理**: 为模型设定角色、风格和行为规则，影响整个对话的一致性。

**示例**:

```python
response = dashscope.Generation.call(
    model="qwen-plus",
    messages=[
        {
            "role": "system",
            "content": """你是一位专业的Python代码审查员。
            你的职责是：
            1. 检查代码的正确性和效率
            2. 指出潜在的bug
            3. 提供改进建议
            4. 使用简洁、专业的语言
            
            回答格式：
            - 正确性评分：X/10
            - 发现的问题：[列表]
            - 改进建议：[列表]
            """
        },
        {
            "role": "user",
            "content": "请审查这段代码：def add(a, b): return a + b"
        }
    ]
)
```

**实践练习**: 为你自己设计3个不同的System Prompt：
1. 一个"技术文档撰写助手"
2. 一个"创意写作伙伴"
3. 一个"学习辅导老师"

### 1.5 Prompt Engineering 最佳实践总结

| 技巧 | 适用场景 | 效果 |
|------|----------|------|
| Few-shot | 需要模型模仿特定格式或风格 | 高 |
| Chain-of-Thought | 复杂推理、数学计算 | 很高 |
| System Prompt | 长对话、角色扮演 | 高 |
| 明确指令 | 简单任务 | 中 |
| 输出格式约束 | 需要结构化输出 | 高 |

**关键原则**:
1. **具体 > 模糊** — 明确说明你想要什么
2. **示例 > 描述** — 一个示例胜过千言万语
3. **分步 > 一步到位** — 复杂任务拆解
4. **约束 > 开放** — 限定输出格式减少混乱

### 1.6 资源链接

- Prompt Engineering Guide（中文版）: https://www.promptingguide.ai/zh
- OpenAI Prompt 最佳实践: https://platform.openai.com/docs/guides/prompt-engineering
- 通义千问Prompt技巧: https://help.aliyun.com/zh/dashscope/

---

## Part 2: LangChain 入门（30分钟）

### 2.1 什么是LangChain？

LangChain是一个开源框架，专门用于构建基于LLM的应用。它提供了：
- **Chains（链）** — 将多个步骤串联起来
- **Memory（记忆）** — 让LLM记住上下文
- **OutputParser（输出解析器）** — 规范化模型输出
- **Tools（工具）** — 让LLM调用外部功能
- **Agents（智能体）** — 自主决策的AI系统

### 2.2 安装LangChain

```bash
# 基础安装
pip install langchain

# 完整安装（推荐）
pip install langchain langchain-openai langchain-community

# 如果用通义千问
pip install dashscope
```

### 2.3 第一个LangChain程序：LLMChain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 通义千问

# 初始化LLM
llm = Tongyi(
    model_name="qwen-plus",
    dashscope_api_key="your-key"
)

# 创建Prompt模板
template = """你是一位{role}。请用{style}的风格回答以下问题：

问题：{question}

回答："""

prompt = PromptTemplate(
    input_variables=["role", "style", "question"],
    template=template
)

# 创建Chain
chain = LLMChain(llm=llm, prompt=prompt)

# 运行
result = chain.run(
    role="资深Python工程师",
    style="简洁专业",
    question="如何优化一个Python列表的查找性能？"
)

print(result)
```

### 2.4 Memory（记忆）让对话有上下文

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi

llm = Tongyi(model_name="qwen-plus", dashscope_api_key="your-key")

# 创建带记忆的对话链
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 打印中间过程
)

# 第一轮
response1 = conversation.predict(input="我叫亮亮，我喜欢LLM开发。")
print(response1)

# 第二轮 - 模型会记住上一轮的内容
response2 = conversation.predict(input="你还记得我的名字和兴趣吗？")
print(response2)  # 应该提到"亮亮"和"LLM开发"
```

**Memory类型对比**:

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| ConversationBufferMemory | 保存完整对话历史 | 短对话 |
| ConversationSummaryMemory | 自动总结历史 | 长对话 |
| ConversationBufferWindowMemory | 只保留最近N轮 | 控制token消耗 |
| ConversationSummaryBufferMemory | 混合策略 | 最优平衡 |

### 2.5 OutputParser（输出解析器）

让模型输出结构化的数据（如JSON）：

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi

# 定义输出格式
response_schemas = [
    ResponseSchema(name="sentiment", description="情感分类：正面/负面/中性"),
    ResponseSchema(name="score", description="情感得分：0-10"),
    ResponseSchema(name="summary", description="一句话总结")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式指令
format_instructions = parser.get_format_instructions()

template = """
分析以下文本的情感：

文本：{text}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)

llm = Tongyi(model_name="qwen-plus", dashscope_api_key="your-key")
chain = prompt | llm | parser

result = chain.invoke({"text": "今天终于完成了项目，虽然很累但很满足！"})
print(result)
# 输出：{'sentiment': '正面', 'score': '8', 'summary': '完成项目带来的满足感'}
```

### 2.6 综合示例：简单问答系统

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi

llm = Tongyi(model_name="qwen-plus", dashscope_api_key="your-key")
memory = ConversationBufferMemory(memory_key="chat_history")

template = """
你是一个专业的技术问答助手。

对话历史：
{chat_history}

用户问题：{question}

请用简洁、准确的方式回答。
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=template
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 多轮对话
print(chain.run(question="什么是RAG？"))
print(chain.run(question="它和微调有什么区别？"))  # 会记住"它"指的是RAG
```

### 2.7 资源链接

- LangChain官方文档（中文）: https://python.langchain.com/docs/get_started/introduction
- LangChain GitHub: https://github.com/langchain-ai/langchain
- LangChain中文教程: https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide

---

## Part 3: 名著改写器测试与优化（30分钟）

### 3.1 服务回顾

昨天我们部署了"名著改写器"项目，服务地址：http://47.102.222.123:5000

**当前状态**（从learning-progress.json）:
- ✅ 服务运行中（systemd管理，PID 48982）
- ✅ API Key已配置（通义千问）
- ✅ 前端已优化（v2版本）
- ✅ 支持8种改写风格

### 3.2 测试计划

今天需要进行系统性测试，验证不同风格的改写效果：

**测试用例设计**:

| 测试编号 | 输入文本 | 预期风格 | 验证重点 |
|----------|----------|----------|----------|
| TC-01 | 《红楼梦》第一段 | 诙谐幽默 | 是否保留原意+增加趣味 |
| TC-02 | 《西游记》开篇 | 现代口语 | 是否通俗易懂 |
| TC-03 | 《水浒传》人物描写 | 简洁风格 | 是否精简但不丢失信息 |
| TC-04 | 《三国演义》对话 | 戏剧化 | 是否有戏剧张力 |
| TC-05 | 《论语》选段 | 儿童友好 | 是否适合孩子理解 |

### 3.3 测试步骤

**Step 1: 访问前端界面**
- 打开浏览器访问 http://47.102.222.123:5000
- 检查页面加载速度
- 验证导航栏、输入框、风格选择等UI元素

**Step 2: 测试每种风格**
1. 选择一种改写风格（如"诙谐幽默"）
2. 粘贴一段经典文本
3. 点击"开始改写"按钮
4. 记录：
   - 响应时间
   - 输出质量
   - 是否保留原意
   - 风格是否符合预期

**Step 3: 检查API响应**
```bash
# 直接测试API
curl -X POST http://47.102.222.123:5000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{
    "text": "宝玉看罢，因笑道：这个妹妹我曾见过的。",
    "style": "诙谐幽默"
  }'
```

**Step 4: 记录发现的问题**

| 问题类型 | 描述 | 严重程度 |
|----------|------|----------|
| 功能问题 | [描述] | 高/中/低 |
| 性能问题 | [描述] | 高/中/低 |
| UI问题 | [描述] | 高/中/低 |
| 输出质量问题 | [描述] | 高/中/低 |

### 3.4 优化方向思考

基于测试结果，考虑以下优化点：
1. **Prompt质量** — 当前的改写Prompt是否足够好？能否更精确？
2. **响应速度** — 是否可以加缓存？是否需要优化模型调用？
3. **前端体验** — 输入框是否够用？风格选择是否清晰？
4. **错误处理** — API失败时是否有友好提示？
5. **输出格式** — 是否需要支持导出为文件（PDF、Markdown）？

### 3.5 实战记录模板

```markdown
## 测试记录

### 测试环境
- 时间：2026-03-13
- 服务地址：http://47.102.222.123:5000
- 浏览器：[填写]
- 网络环境：[填写]

### 测试结果

#### TC-01: 诙谐幽默风格
- 输入：[填写]
- 输出：[填写]
- 响应时间：[填写]秒
- 质量评分：[1-10]
- 问题：[如有]

#### TC-02: 现代口语风格
...

### 发现的问题
1. [问题描述]
2. [问题描述]

### 优化建议
1. [建议描述]
2. [建议描述]
```

---

## Part 4: 学习笔记 & GitHub 更新（15分钟）

### 4.1 整理今日笔记

将今天学到的内容整理成结构化的笔记：

**要点回顾**:
1. Prompt Engineering高级技巧：Few-shot、CoT、System Prompt
2. LangChain核心概念：Chain、Memory、OutputParser
3. 名著改写器测试结果

### 4.2 更新GitHub仓库

**步骤**:
1. 创建 `week-01/day-02/` 目录
2. 上传今天的笔记文件 `Day2-学习内容.md`
3. 更新 `learning-log.md` 添加Day 2记录

**GitHub API推送示例**:

```bash
# 创建文件（通过API）
curl -X PUT \
  -H "Authorization: token ghp_..." \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/liang8065/llm-learning-journey/contents/week-01/day-02/Day2-学习内容.md \
  -d '{
    "message": "Add Day 2 learning content",
    "content": "'$(base64 -w0 Day2-学习内容.md)'"
  }'
```

### 4.3 更新学习进度

更新 `memory/learning-progress.json`:
- currentDay: 2 → 3
- completedTasks: 添加今天完成的任务
- 添加Day 2学习笔记

---

## 今日时间安排

| 时间段 | 任务 | 时长 |
|--------|------|------|
| Part 1 | Prompt Engineering深入学习 | 30分钟 |
| Part 2 | LangChain入门 | 30分钟 |
| Part 3 | 名著改写器测试 | 30分钟 |
| Part 4 | 学习笔记&GitHub更新 | 15分钟 |
| **总计** | | **~105分钟** |

---

## 总结

Day 2的学习重点是：
1. **深化Prompt技能** — 从基础到高级，掌握Few-shot、CoT等策略
2. **理解LangChain** — 掌握Chain、Memory、OutputParser三大核心
3. **实战验证** — 测试名著改写器，发现问题，规划优化

**明天预告（Day 3）**:
- 继续LangChain实践，尝试Agent开发
- 针对名著改写器的优化方案实施
- 开始探索RAG（检索增强生成）基础概念

---

*文档创建时间：2026-03-13*
*创建者：OpenClaw AI助手*
*GitHub仓库：https://github.com/liang8065/llm-learning-journey*
