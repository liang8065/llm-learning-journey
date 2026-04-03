# 📚 Day 22 — 高级 RAG 技术（2026-04-03）

> **周期**：Day 22 / Week 4 / 阶段二：技能提升期
> **目标**：掌握 HyDE、Multi-query 检索、Self-RAG 等高级 RAG 技术
> **预计用时**：1.5 - 2 小时
> **LangChain 版本**：1.0+（全部使用 LCEL 表达式链）

---

## 📋 今日任务

| 时间段 | 任务 | 内容 | 时长 |
|--------|------|------|------|
| Part 1 | HyDE 假设性文档嵌入 | 原理 + 完整代码实现 | 30 分钟 |
| Part 2 | Multi-query 多路检索 | 原理 + 完整代码实现 | 30 分钟 |
| Part 3 | Self-RAG 自我反思 | 原理 + 完整代码实现 | 30 分钟 |
| Part 4 | 三种 RAG 策略效果对比 | 同一问题对比测试 | 15 分钟 |

---

## Part 1：HyDE 假设性文档嵌入 🎯

### 原理

HyDE（Hypothetical Document Embeddings）的核心思想：

- **问题**：用户问题和知识库文档的语义表达方式差异大，直接检索匹配度不高
- **解法**：让 LLM 先假设一个"理想答案"，用假设文档去检索，而不是用原始问题
- **效果**：假设文档的语义分布更接近知识库内容，检索准确度大幅提升

**流程**：问题 → LLM 生成假设答案 → 假设答案向量化 → 检索 → 生成最终回答

### 代码

见 `code/part1_hyde_rag.py`

**关键代码片段**：
```python
# HyDE 完整 LCEL 链
hyde_chain = (
    {
        "question": lambda x: x,
        "hypothetical": (hyde_prompt | llm | StrOutputParser()),
    }
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "hypothetical": x["hypothetical"],
        "context": retrieve_with_hyde(x["hypothetical"]),
    })
    | qa_prompt
    | llm
    | StrOutputParser()
)
```

---

## Part 2：Multi-query 多路检索 🔀

### 原理

- **问题**：单一查询可能遗漏相关内容
- **解法**：LLM 生成 3 条不同角度的等价查询，分别检索后合并去重
- **效果**：检索覆盖率大幅提升

**流程**：问题 → 生成多路查询 → 并行检索 → 合并去重 → 生成回答

### 代码

见 `code/part2_multi_query_rag.py`

---

## Part 3：Self-RAG 自我反思 🤔

### 原理

- **问题**：传统 RAG 检索质量不可控
- **解法**：引入"自我评估"步骤，判断检索文档是否足够，不足则改进查询重新检索
- **效果**：系统自动迭代优化检索质量

**流程**：检索 → 评估（相关性/完整性）→ 不足则改进查询 → 再检索 → 直到足够则生成回答

### 代码

见 `code/part3_self_rag.py`

---

## Part 4：三种策略对比 ⚔️

运行 `code/part1_hyde_rag.py`、`code/part2_multi_query_rag.py`、`code/part3_self_rag.py`，对比同一个问题下三种策略的检索文档数和回答质量。

**经验总结**：
- HyDE 适合问题表达模糊、与知识库语义差异大的场景
- Multi-query 适合需要全面覆盖、避免遗漏的场景
- Self-RAG 适合对回答质量要求高的场景，自动迭代提升

---

## 🔗 参考资料

- HyDE 论文: https://arxiv.org/abs/2212.10496
- Self-RAG 论文: https://arxiv.org/abs/2310.11511
- LangChain 多路检索: https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
- LangChain RAG: https://python.langchain.com/docs/tutorials/rag/
