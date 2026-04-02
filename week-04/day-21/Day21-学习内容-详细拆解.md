# Day 21 任务详细拆解

> 日期：2026-04-02 | 主题：RAG 实战 + 检索优化 | 预计 1.5-2 小时

---

## Part 0：复习 & 环境检查（10分钟）

### 输入
- 前几天的学习笔记
- `learning-progress.json`

### 操作
1. 快速回顾 Day 3-5 的 RAG 基础内容
2. 确认 Python 环境（建议 3.8+，可用 conda）
3. 确认 pip 依赖已安装：
   ```bash
   pip install -U langchain langchain-community langchain-openai faiss-cpu dashscope chromadb
   ```

### 输出
- 环境确认清单
- 前序知识点复习笔记

### ✅ 检查点
- [ ] Python 版本 ≥ 3.8
- [ ] 所有依赖安装成功
- [ ] API Key 可用

---

## Part 1：最小可运行 RAG（30分钟）

### 输入
- 通义千问 API Key
- 一个简单的知识文档目录

### 操作步骤

**步骤 1.1：准备测试文档**
```bash
mkdir -p knowledge_base
echo "RAG（检索增强生成）是一种结合检索和生成的AI技术..." > knowledge_base/rag_intro.txt
echo "向量数据库通过embedding将文本转换为数值向量..." > knowledge_base/vector_db.txt
```

**步骤 1.2：编写并运行核心代码**
- 复制 Day21 学习内容中的"最小可运行 RAG"代码
- 填入 API Key
- 运行 `python rag_demo.py`

**步骤 1.3：验证结果**
- 观察日志输出的文档数量和片段数量
- 测试提问，检查回答是否基于文档

### 输出
- `code/day21_rag_demo.py`
- 测试结果截图/记录

### ✅ 检查点
- [ ] 文档加载成功
- [ ] FAISS 索引构建成功
- [ ] 能回答关于文档内容的问题

---

## Part 2：检索质量对比实验（30分钟）

### 输入
- Part 1 已构建的 RAG 系统

### 操作步骤

**步骤 2.1：Chunk Size 对比**
- 分别用 chunk_size=200, 500, 800, 1000 运行
- 对同一个提问，记录每次检索到的结果
- 对比哪个 chunk_size 的回答最准确

**步骤 2.2：MMR vs 普通检索对比**
- 用普通检索：`search_kwargs={"k": 3}`
- 用 MMR 检索：`search_type="mmr"`
- 对比返回结果的相关性和多样性

**步骤 2.3：记录实验结果**
- 用表格记录每个配置的效果

### 输出
- 实验记录（推荐保存在学习笔记中）
- 对最佳 chunk_size 的结论

### ✅ 检查点
- [ ] 至少测试了 3 种 chunk_size
- [ ] 对比了 MMR 和普通检索
- [ ] 得出了最佳配置结论

---

## Part 3：封装 SimpleRAG 类（30分钟）

### 输入
- Part 1-2 的经验代码

### 操作步骤

**步骤 3.1：编写类文件**
- 创建 `code/rag_system.py`
- 实现 SimpleRAG 类（参考 Day21 学习内容中的完整代码）

**步骤 3.2：测试构建流程**
```python
rag = SimpleRAG(api_key="你的Key")
docs = rag.load_documents("./knowledge_base")
rag.build(docs)
print(rag.query("RAG是什么？"))
```

**步骤 3.3：测试保存/加载**
```python
rag.save("faiss_index")
# 重新加载
rag2 = SimpleRAG.load("你的Key", "faiss_index")
print(rag2.query("RAG是什么？"))
# 应该得到相同答案，无需重新构建
```

### 输出
- `code/rag_system.py`
- 测试运行结果

### ✅ 检查点
- [ ] 类可正常实例化和使用
- [ ] 保存/加载功能正常
- [ ] 代码结构清晰，有注释

---

## Part 4：GitHub 更新 & 笔记整理（10分钟）

### 输入
- 今天创建的所有文件

### 操作步骤
1. `git add week-04/day-21/ code/`
2. `git add learning-log.md learning-progress.json`
3. `git commit -m "Day 21: RAG complete pipeline + retrieval optimization"`
4. `git push origin main`

### 输出
- 已推送的 GitHub 仓库更新

### ✅ 检查点
- [ ] 所有新文件已提交
- [ ] 远程仓库已更新
- [ ] learning-log.md 已更新今日记录

---

## ⏰ 时间分配建议

| 时间段 | 任务 | 时长 |
|--------|------|------|
| 10:30-10:40 | Part 0 复习 | 10min |
| 10:40-11:10 | Part 1 最小 RAG | 30min |
| 11:10-11:40 | Part 2 检索对比 | 30min |
| 11:40-12:10 | Part 3 封装类 | 30min |
| 12:10-12:20 | Part 4 GitHub | 10min |
| **合计** | | **1小时50分钟** |

---

## 🎯 今日学习成果

完成后你应该能够：
1. ✅ 从头搭建一个完整的 RAG 问答系统
2. ✅ 知道如何调整 chunk_size 优化检索效果
3. ✅ 使用 MMR 检索平衡相关性和多样性
4. ✅ 封装自己的 RAG 工具类
5. ✅ 理解"索引只构建一次"的工程实践
