# 📚 名著改写器 (Classic Literature Rewriter)

> 让经典文学焕发新生 —— 基于大语言模型的名著风格改写系统

## 🎯 项目简介

这是一个基于 LLM 的经典文学风格改写系统，用户可以上传或输入经典名著文本，选择多种改写风格，系统会保留章节结构并进行深度润色改写。

### ✨ 功能特点

- **多风格改写**：支持 8 种改写风格（诙谐、严肃、活泼、现代口语、简洁、戏剧化、儿童友好、学术风格）
- **章节结构保留**：自动识别章节，按章节进行改写
- **深度润色**：保留原意的同时进行深度改写
- **网页界面**：简洁易用的网页界面，支持文本粘贴和文件上传
- **API 集成**：基于通义千问 API，支持国产大模型

---

## 📖 支持的改写风格

| 风格 | Emoji | 说明 |
|------|-------|------|
| 诙谐幽默 | 🔥 | 加入吐槽、现代梗、轻松调侃 |
| 严肃庄重 | 🎩 | 保留古典韵味，简化晦涩表达 |
| 活泼生动 | 🌸 | 口语化、画面感强、节奏轻快 |
| 现代口语 | 🤖 | 完全用现代人说话方式改写 |
| 简洁明快 | 📖 | 去掉冗余，直击核心情节 |
| 戏剧化 | 🎭 | 增强冲突、对话、张力 |
| 儿童友好 | 🧒 | 适合小朋友阅读的简化版本 |
| 学术风格 | 📝 | 保留文学性，增加注释和分析 |

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目（如果还没有）
cd llm-learning-journey/classic-rewriter

# 安装后端依赖
cd backend
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
```

### 2. 获取 API Key

1. 访问 [阿里云 DashScope](https://dashscope.aliyun.com/)
2. 注册账号并获取 API Key
3. 将 Key 填入 `.env` 文件

### 3. 启动服务

```bash
# 启动后端服务
cd backend
python app.py

# 访问网页
# 打开浏览器访问 http://localhost:5000
```

---

## 📁 项目结构

```
classic-rewriter/
├── backend/                 # 后端服务
│   ├── app.py              # Flask 应用主入口
│   ├── rewriter.py         # LLM 改写逻辑
│   ├── prompts.py          # 风格 Prompt 模板
│   ├── requirements.txt    # Python 依赖
│   └── .env.example        # 环境变量示例
├── frontend/               # 前端界面
│   ├── index.html          # 主页面
│   ├── style.css           # 样式文件
│   └── script.js           # 前端逻辑
├── data/                   # 数据目录
│   ├── uploads/            # 上传的文件
│   └── results/            # 改写结果
└── README.md               # 本文件
```

---

## 🔧 API 接口

### 获取风格列表
```
GET /api/styles
```

### 改写文本
```
POST /api/rewrite
Content-Type: application/json

{
  "text": "要改写的原文",
  "style": "风格ID（如 humorous）",
  "title": "章节标题（可选）"
}
```

### 按章节改写
```
POST /api/rewrite/chapters
Content-Type: application/json

{
  "chapters": [
    {"title": "第一章", "content": "..."},
    {"title": "第二章", "content": "..."}
  ],
  "style": "风格ID"
}
```

### 上传文件
```
POST /api/upload
Content-Type: multipart/form-data

file: 文本文件（.txt）
```

---

## 💡 使用示例

### 示例 1：改写《三国演义》片段

**原文：**
```
第一回 宴桃园豪杰三结义 斩黄巾英雄首立功

话说天下大势，分久必合，合久必分...
```

**选择风格：** 诙谐幽默 🔥

**改写结果：**
（系统会生成带有现代梗和吐槽的幽默版本）

### 示例 2：改写《红楼梦》片段

**原文：**
```
第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀

此开卷第一回也...
```

**选择风格：** 现代口语 🤖

**改写结果：**
（系统会生成完全用现代人说话方式的版本）

---

## 🎓 学习价值

这个项目是 LLM 应用开发的绝佳实践：

1. **Prompt 工程**：为每种风格设计专门的 Prompt 模板
2. **API 集成**：调用通义千问 API 进行文本生成
3. **文本处理**：章节识别、分块处理、结果合并
4. **Web 开发**：前后端分离架构
5. **用户体验**：简洁直观的界面设计

---

## 📈 未来计划

- [ ] 支持更多文件格式（PDF、EPUB）
- [ ] 添加更多改写风格
- [ ] 支持批量处理多章节
- [ ] 添加改写质量评估
- [ ] 支持导出为 PDF/EPUB
- [ ] 添加历史记录和收藏功能

---

## 📝 许可证

本项目为 LLM 学习项目的一部分，仅供学习使用。

---

*项目地址：https://github.com/liang8065/llm-learning-journey/tree/main/classic-rewriter*
