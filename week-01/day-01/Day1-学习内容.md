# Day 1: LLM应用开发入门 - 完整学习内容 📚

> **学习目标**: 完成环境搭建，理解LLM核心概念，跑通第一个API调用
> **预计用时**: 1.5小时
> **适用对象**: 完全零基础的初学者

---

## 前言

欢迎开始你的LLM应用开发之旅！今天是第一天，我们将从最基础的环境搭建开始，然后快速了解LLM的核心概念，最后亲手跑通第一个API调用。

**今日学习路线图**:
```
环境搭建 (30分钟) → 核心概念 (30分钟) → API调用 (30分钟) → 总结记录 (15分钟)
```

---

# Part 1: 环境搭建 (30分钟)

## 1.1 Python 3.10+ 安装

### 为什么需要Python 3.10+？
Python是目前最主流的AI/ML开发语言，LLM应用开发几乎都基于Python。3.10+版本有更好的类型提示和性能优化。

### 检查当前Python版本
打开终端（Windows: CMD/PowerShell，Mac/Linux: Terminal），输入：
```bash
python --version
```
或
```bash
python3 --version
```

如果显示的是Python 3.10或更高版本，恭喜你，可以直接跳到下一步！

### 安装步骤

**Windows用户**:
1. 访问 https://www.python.org/downloads/
2. 点击 "Download Python 3.12.x"（最新稳定版）
3. 运行安装程序
4. ⚠️ **重要**: 勾选 "Add Python to PATH" 选项
5. 点击 "Install Now"
6. 安装完成后，重新打开CMD验证：`python --version`

**Mac用户**:
```bash
# 使用Homebrew安装（推荐）
brew install python@3.11

# 或者直接从官网下载安装包
```

**Linux用户** (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### 验证安装
```bash
python --version
# 应该显示 Python 3.10.x 或更高

pip --version
# 应该显示 pip 的版本
```

---

## 1.2 VS Code 安装与配置

### 为什么选择VS Code？
VS Code是目前最流行的代码编辑器，轻量、免费、插件丰富，完美支持Python和AI开发。

### 安装VS Code
1. 访问 https://code.visualstudio.com/
2. 下载对应系统的版本
3. 运行安装程序，一路"下一步"即可

### 安装Python扩展
1. 打开VS Code
2. 点击左侧扩展图标（或按 `Ctrl+Shift+X`）
3. 搜索 "Python"
4. 安装 Microsoft 官方的 Python 扩展
5. 同时建议安装 "Python Debugger" 扩展

### 配置Python解释器
1. 在VS Code中按 `Ctrl+Shift+P`（Mac: `Cmd+Shift+P`）
2. 输入 "Python: Select Interpreter"
3. 选择你刚刚安装的Python 3.10+版本
4. 现在VS Code就知道用哪个Python了

### 创建你的第一个Python文件
1. 新建文件夹 `llm-learning`
2. 在VS Code中打开这个文件夹
3. 新建文件 `hello.py`
4. 输入以下代码：
```python
print("Hello, LLM World!")
```
5. 在终端中运行：`python hello.py`
6. 看到输出 "Hello, LLM World!" 就说明环境没问题了！

---

## 1.3 API Key 配置

### 什么是API Key？
API Key是你访问LLM服务（如OpenAI、通义千问）的"通行证"。没有它，你就无法调用这些服务。

### 获取OpenAI API Key
1. 访问 https://platform.openai.com
2. 注册/登录账号
3. 点击右上角头像 → "View API keys"
4. 点击 "Create new secret key"
5. 复制生成的Key（格式：`sk-xxxxxxxxxxxxxxxxxxxx`）
6. ⚠️ **重要**: 这个Key只会显示一次，请妥善保存！

### 获取通义千问 API Key（国产替代）
1. 访问 https://dashscope.aliyun.com/
2. 注册/登录阿里云账号
3. 进入控制台 → "API-KEY管理"
4. 创建新的API Key
5. 复制并保存

### 配置环境变量
为了安全，不要把API Key直接写在代码里，而是配置为环境变量。

**Windows**:
```bash
# CMD
setx OPENAI_API_KEY "sk-你的key"
setx DASHSCOPE_API_KEY "你的通义千问key"

# PowerShell
$env:OPENAI_API_KEY="sk-你的key"
$env:DASHSCOPE_API_KEY="你的通义千问key"
```

**Mac/Linux**:
```bash
# 编辑 ~/.bashrc 或 ~/.zshrc
export OPENAI_API_KEY="sk-你的key"
export DASHSCOPE_API_KEY="你的通义千问key"

# 然后执行
source ~/.bashrc  # 或 source ~/.zshrc
```

### 验证配置
```bash
# Windows CMD
echo %OPENAI_API_KEY%

# Mac/Linux
echo $OPENAI_API_KEY
```

---

# Part 2: 核心概念学习 (30分钟)

## 2.1 LLM (Large Language Model) - 大语言模型

### 什么是LLM？
LLM（Large Language Model，大语言模型）是一种基于深度学习的人工智能模型，它通过学习海量文本数据，学会了理解和生成人类语言。

**简单比喻**: 把LLM想象成一个读过全世界所有书籍、文章、网页的"超级学生"。它能回答问题、写文章、翻译、写代码...几乎任何与语言相关的任务。

### LLM是怎么工作的？
LLM的核心是"预测下一个词"。当你给它一个输入（Prompt），它会根据已有的文本，一个词一个词地预测接下来应该说什么。

```
输入: "中国的首都是"
LLM预测: "北京" (概率最高)
```

这个过程称为"自回归生成"（Autoregressive Generation）。

### 常见的LLM模型
| 模型名称 | 开发公司 | 特点 |
|---------|---------|------|
| GPT-4 | OpenAI | 最强大的通用模型之一 |
| Claude 3 | Anthropic | 安全性高，适合企业应用 |
| 通义千问 | 阿里云 | 国产，中文理解强 |
| 文心一言 | 百度 | 国产，百度生态集成好 |
| Llama 3 | Meta | 开源，可本地部署 |

### LLM能做什么？
- ✅ 文本生成（写作、改写、总结）
- ✅ 问答对话
- ✅ 代码编写和调试
- ✅ 翻译
- ✅ 情感分析
- ✅ 数据提取
- ... 几乎任何语言任务

---

## 2.2 Token - 模型处理的基本单位

### 什么是Token？
Token是LLM处理文本的基本单位。一个Token可以是一个单词、一个字、或一个标点符号。

**中文Token示例**:
```
"我爱中国" → 可能被分成: ["我", "爱", "中国"] (3个token)
"Hello World!" → 可能被分成: ["Hello", " World", "!"] (3个token)
```

### 为什么要关心Token？
1. **计费**: LLM API通常按Token数量收费
   - OpenAI GPT-4: ~$0.03/1K输入token, ~$0.06/1K输出token
   - 通义千问: 按Token计费，价格更便宜

2. **限制**: 每个模型有最大Token限制（上下文窗口）
   - GPT-4: 8K-128K token
   - 超过限制会被截断

3. **性能**: Token越多，处理越慢，成本越高

### Token计数示例
```python
import tiktoken

# 计算文本的token数量
encoding = tiktoken.encoding_for_model("gpt-4")
text = "我今天学习了LLM应用开发"
tokens = encoding.encode(text)
print(f"Token数量: {len(tokens)}")
print(f"Token列表: {tokens}")
```

**经验法则**:
- 1个中文字符 ≈ 1-2个token
- 1个英文单词 ≈ 1-1.5个token
- 1000个token ≈ 750个英文单词 ≈ 500个中文字

---

## 2.3 Prompt - 提示词工程

### 什么是Prompt？
Prompt（提示词）是你给LLM的"指令"或"问题"。Prompt的质量直接决定了LLM输出的质量。

**简单比喻**: Prompt就像是给厨师的点菜说明。你说得越清楚，厨师做得越好。

### Prompt的组成部分
一个完整的Prompt通常包含：

```
1. 角色设定 (System): "你是一个专业的编程助手"
2. 任务描述 (User): "帮我写一个Python函数，计算斐波那契数列"
3. 输出格式: "请用markdown格式输出，包含代码和解释"
4. 示例 (Few-shot): "例如: 输入5，输出[0,1,1,2,3]"
```

### Prompt设计原则

#### 原则1: 明确具体
```
❌ 差: "写点代码"
✅ 好: "写一个Python函数，接收一个整数n，返回斐波那契数列的前n项"
```

#### 原则2: 提供上下文
```
❌ 差: "翻译这句话"
✅ 好: "将以下中文句子翻译成英文，保持口语化风格：今天天气真好"
```

#### 原则3: 指定输出格式
```
❌ 差: "解释一下机器学习"
✅ 好: "用3个要点解释机器学习，每个要点不超过20字，用markdown列表格式"
```

#### 原则4: 使用分步思考
```
✅ 好: "请按以下步骤思考：
1. 首先分析问题的关键点
2. 然后给出解决方案
3. 最后总结要点"
```

### 实战练习
尝试写一个Prompt，让LLM帮你：
"用Python写一个函数，判断一个字符串是否是回文（正读反读都一样）"

你的Prompt可能是这样的：
```
你是一个Python编程专家。请写一个Python函数：
- 函数名: is_palindrome
- 输入: 一个字符串
- 输出: 布尔值（True如果是回文，False如果不是）
- 要求: 忽略大小写和空格
- 请提供完整的代码和使用示例
```

---

## 2.4 Embedding - 向量嵌入

### 什么是Embedding？
Embedding（嵌入）是将文本转换为数字向量的技术。这些向量能够捕捉文本的"语义"，即文本的含义。

**简单比喻**: 想象把每个词、每句话都变成地图上的一个点。意思相近的文本，它们在地图上的位置也相近。

### Embedding能做什么？
- **语义搜索**: 找到意思相近的文本，而不是关键词匹配
- **文本分类**: 根据语义相似度分类
- **推荐系统**: 找到相似的内容
- **聚类分析**: 将相似的文本分组

### Embedding示例
```
"猫" → [0.2, 0.8, 0.1, 0.5, ...] (768维向量)
"狗" → [0.3, 0.7, 0.2, 0.6, ...] (768维向量)
"汽车" → [0.9, 0.1, 0.8, 0.2, ...] (768维向量)
```

"猫"和"狗"的向量会比较接近（都是动物），而"汽车"的向量会比较远。

### 用代码获取Embedding
```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="我今天学习了LLM应用开发"
)

embedding = response.data[0].embedding
print(f"Embedding维度: {len(embedding)}")
print(f"前10个值: {embedding[:10]}")
```

### 相似度计算
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 比较两个文本的相似度
text1_embedding = [...]  # "猫"的向量
text2_embedding = [...]  # "狗"的向量
similarity = cosine_similarity(text1_embedding, text2_embedding)
print(f"相似度: {similarity}")  # 接近1表示很相似
```

---

## 2.5 RAG (Retrieval-Augmented Generation) - 检索增强生成

### 什么是RAG？
RAG是一种技术，它让LLM在回答问题时，先从外部知识库检索相关信息，然后再基于这些信息生成回答。

**简单比喻**: 就像考试时允许带"小抄"。LLM本身的知识有截止日期，但通过RAG，它可以查阅最新的、私有的文档来回答问题。

### 为什么需要RAG？
1. **知识截止问题**: LLM的知识有训练截止日期，不知道最新信息
2. **私有数据**: LLM不知道你公司的内部文档
3. **减少幻觉**: 有参考依据，减少胡说八道

### RAG的工作流程
```
用户问题 → 检索相关文档 → 把文档和问题一起给LLM → LLM基于文档生成回答
```

### RAG实战示例
假设你有一个公司内部文档库：

1. **索引阶段** (离线):
   ```
   文档: "公司2024年员工手册.pdf"
   → 分割成小块 (chunking)
   → 转换成Embedding向量
   → 存储到向量数据库
   ```

2. **查询阶段** (在线):
   ```
   用户问: "公司年假有几天？"
   → 把问题转成Embedding
   → 在向量数据库中搜索相似文档
   → 找到: "员工手册第5章: 年假制度规定工作满1年享有5天年假"
   → 把这个文档和问题一起给LLM
   → LLM回答: "根据公司员工手册，工作满1年享有5天年假。"
   ```

### RAG架构图
```
┌─────────────┐
│   用户提问   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │ → 将问题转为向量
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 向量数据库   │ → 搜索相似文档
│ (Chroma/    │
│  Pinecone)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  检索到的    │ → 相关文档片段
│  文档片段    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Prompt    │ → 问题 + 文档 → LLM
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   LLM回答   │ → 基于文档的回答
└─────────────┘
```

### 常用RAG工具
- **LangChain**: 最流行的RAG框架
- **LlamaIndex**: 专注数据索引和检索
- **ChromaDB**: 轻量级向量数据库
- **Pinecone**: 云端向量数据库

---

# Part 3: 第一个API调用 (30分钟)

## 3.1 安装必要的库

### 创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv llm-env

# 激活虚拟环境
# Windows:
llm-env\Scripts\activate
# Mac/Linux:
source llm-env/bin/activate

# 安装所需的库
pip install openai dashscope python-dotenv
```

### 各个库的作用
- `openai`: OpenAI官方Python SDK
- `dashscope`: 阿里云通义千问SDK
- `python-dotenv`: 管理环境变量（可选但推荐）

---

## 3.2 OpenAI API 调用

### 方法1: 使用环境变量
创建文件 `test_openai.py`:

```python
import os
from openai import OpenAI

# 从环境变量读取API Key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 调用GPT模型
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # 或 "gpt-4"
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "用一句话解释什么是LLM"}
    ],
    max_tokens=100,
    temperature=0.7
)

# 打印结果
print("回答:", response.choices[0].message.content)
print("使用的Token数:", response.usage.total_tokens)
```

### 方法2: 使用.env文件（推荐）
创建 `.env` 文件（注意不要上传到Git！）:
```
OPENAI_API_KEY=sk-你的key
```

创建文件 `test_openai_env.py`:

```python
from dotenv import load_dotenv
import os
from openai import OpenAI

# 加载.env文件
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "用一句话解释什么是LLM"}
    ]
)

print(response.choices[0].message.content)
```

### 参数解释
| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| model | 使用的模型 | "gpt-3.5-turbo" 或 "gpt-4" |
| messages | 对话历史 | 包含system/user/assistant |
| max_tokens | 最大输出token数 | 100-1000 |
| temperature | 创意程度 | 0.0-1.0，越高越有创意 |

### 运行测试
```bash
python test_openai.py
```

**预期输出**:
```
回答: LLM（大语言模型）是一种通过学习海量文本数据来理解和生成人类语言的人工智能模型。
使用的Token数: 45
```

---

## 3.3 通义千问 API 调用

### 安装SDK
```bash
pip install dashscope
```

### 调用代码
创建文件 `test_qwen.py`:

```python
import os
import dashscope

# 设置API Key
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

# 调用通义千问
response = dashscope.Generation.call(
    model="qwen-turbo",  # 或 qwen-plus, qwen-max
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "用一句话解释什么是LLM"}
    ],
    result_format='message'
)

if response.status_code == 200:
    print("回答:", response.output.choices[0].message.content)
    print("使用的Token数:", response.usage.total_tokens)
else:
    print("错误:", response.message)
```

### 运行测试
```bash
python test_qwen.py
```

### 通义千问模型对比
| 模型 | 特点 | 价格 |
|-----|------|-----|
| qwen-turbo | 快速、便宜 | 最低 |
| qwen-plus | 平衡性能和价格 | 中等 |
| qwen-max | 最强性能 | 最高 |

---

## 3.4 综合示例：封装一个通用的LLM调用函数

创建文件 `llm_helper.py`:

```python
import os
from openai import OpenAI
import dashscope

class LLMHelper:
    def __init__(self, provider="openai"):
        """
        初始化LLM助手
        
        Args:
            provider: "openai" 或 "qwen"
        """
        self.provider = provider
        
        if provider == "openai":
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        elif provider == "qwen":
            dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
        else:
            raise ValueError("不支持的provider，请选择 'openai' 或 'qwen'")
    
    def chat(self, message, system_prompt="你是一个有用的助手", temperature=0.7):
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 创意程度 0-1
            
        Returns:
            LLM的回复文本
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
            
        elif self.provider == "qwen":
            response = dashscope.Generation.call(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                result_format='message',
                temperature=temperature
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"错误: {response.message}"


# 使用示例
if __name__ == "__main__":
    # 使用OpenAI
    helper = LLMHelper(provider="openai")
    result = helper.chat("什么是LLM？用一句话解释")
    print("OpenAI回答:", result)
    
    # 使用通义千问
    # helper = LLMHelper(provider="qwen")
    # result = helper.chat("什么是LLM？用一句话解释")
    # print("通义千问回答:", result)
```

### 运行测试
```bash
python llm_helper.py
```

---

# Part 4: 阅读 & 记录 (15分钟)

## 4.1 今日学习笔记模板

复制以下模板，填写你的学习笔记：

```markdown
# Day 1 学习笔记 (2026-03-12)

## 今日完成情况
- [ ] 环境搭建（Python、VS Code、API Key）
- [ ] 核心概念学习（LLM、Token、Prompt、Embedding、RAG）
- [ ] 第一个API调用
- [ ] 整理笔记

## 学到的关键概念
### LLM
- 我的理解：


### Token
- 我的理解：


### Prompt
- 我的理解：


### Embedding
- 我的理解：


### RAG
- 我的理解：


## 遇到的问题
1. 

## 解决方案
1. 

## 今日收获
- 

## 明日计划
- 

## 代码仓库
- GitHub: https://github.com/liang8065/llm-learning-journey
```

---

## 4.2 今日总结

恭喜你完成Day 1的学习！🎉

### 你今天学到了：
1. ✅ 如何搭建Python + VS Code开发环境
2. ✅ 如何获取和配置API Key
3. ✅ LLM的核心概念：LLM、Token、Prompt、Embedding、RAG
4. ✅ 如何调用OpenAI和通义千问的API
5. ✅ 封装了一个通用的LLM调用工具类

### 下一步：
- 晚上汇报今日进展
- 明天我们将学习更深入的Prompt Engineering技巧
- 开始尝试用LLM解决实际问题

### 今日任务清单
- [ ] 完成环境搭建
- [ ] 理解所有核心概念
- [ ] 跑通API调用
- [ ] 填写学习笔记
- [ ] 向亮亮汇报进展

---

**Day 1 完成！明天见 🚀**
