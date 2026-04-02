# Day 21 - RAG 实战：完整串联 + 检索优化

> 日期：2026-04-02 | 预计耗时：1.5-2小时 | LangChain 1.0+ API

---

## 背景

之前 Week 1 的 RAG 实战没有彻底跑通，今天我们**一次性搞定**，从文档加载到向量检索到问答生成，完整串联一个知识库系统，并且加入检索优化技巧。

---

## Part 1：完整 RAG Pipeline 串联（45分钟）

### 1.1 环境准备

确认依赖已安装（Python 3.8+ 环境）：

```bash
pip install -U langchain langchain-community langchain-openai faiss-cpu dashscope chromadb PyPDF2
```

> ⚠️ 注意：你服务器上 Python 是 3.6 版本，如果跑 LangChain 1.0 需要升级到 3.8+。建议用 conda 创建新环境：
> ```bash
> conda create -n llm python=3.10
> conda activate llm
> ```

### 1.2 核心代码：最小可运行 RAG

```python
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 配置
os.environ["DASHSCOPE_API_KEY"] = "你的通义千问API Key"

# 2. 加载文档
loader = DirectoryLoader("./knowledge_base", glob="**/*.txt")
documents = loader.load()
print(f"加载了 {len(documents)} 个文档")

# 3. 分割文档（chunk）—— 这是RAG成败关键！
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个片段500字符
    chunk_overlap=80,    # 重叠80字符保持上下文连贯
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = splitter.split_documents(documents)
print(f"分割成 {len(chunks)} 个片段")

# 4. 向量化 + 构建 FAISS 索引
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 每次检索取3个最相关片段

# 5. 构建 LLM
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.3,
)

# 6. 构建 RAG Chain（LCEL 方式）
template = """你是一个知识问答助手。根据以下参考信息回答问题。
如果参考信息中没有相关内容，请诚实地说"我不知道"。

参考信息：
{context}

问题：{question}

回答："""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 测试
result = rag_chain.invoke("文档里讲了什么内容？")
print(f"\n回答：{result}")
```

### ✅ 检查点
- [ ] 成功加载文档并打印片段数量
- [ ] FAISS 索引构建成功
- [ ] 能回答问题并打印回答

---

## Part 2：检索质量优化（30分钟）

RAG 效果好坏，**70% 取决于检索质量**。以下是实战中常用的优化手段：

### 2.1 Chunk Size 调优

```python
# 测试不同 chunk_size 的检索效果
for chunk_size in [200, 500, 800, 1000]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5
    )
    chunks = splitter.split_documents(documents)
    print(f"chunk_size={chunk_size}: {len(chunks)} 个片段")
    
    # 用相同问题测试
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("你的测试问题")
    print(f"  检索到 {len(results)} 个结果\n")
```

> 📌 经验值：
> - 技术文档/代码：chunk_size=500-800
> - 普通文章：chunk_size=300-500
> - 对话记录：chunk_size=200-300

### 2.2 MMR（最大边界相关）检索

MMR 可以增加检索结果的**多样性**，减少重复内容：

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,              # 返回3个结果
        "fetch_k": 10,       # 先取10个候选
        "lambda_mult": 0.7   # 0=完全多样性, 1=完全相关性
    }
)
```

### 2.3 元数据过滤

给文档打标签，检索时限定范围：

```python
# 加载时添加元数据
for doc in documents:
    doc.metadata["source_type"] = "技术文档"  # 或其他分类
    doc.metadata["topic"] = "RAG"

# 检索时过滤
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"topic": "RAG"}
    }
)
```

### ✅ 检查点
- [ ] 尝试至少3种不同的 chunk_size
- [ ] 对比 MMR vs 普通检索的效果差异
- [ ] 理解元数据过滤的使用场景

---

## Part 3：构建可复使用的 RAG 类（30分钟）

把今天的代码封装成一个可复用的类：

```python
# rag_system.py
class SimpleRAG:
    """一个简单但完整的 RAG 系统"""
    
    def __init__(self, api_key, model="qwen-plus"):
        self.api_key = api_key
        os.environ["DASHSCOPE_API_KEY"] = api_key
        
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        self.llm = ChatOpenAI(
            model=model,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            temperature=0.3,
        )
        self.vectorstore = None
        self.rag_chain = None
    
    def load_documents(self, directory, glob_pattern="**/*.txt"):
        """从目录加载文档"""
        loader = DirectoryLoader(directory, glob=glob_pattern)
        return loader.load()
    
    def build(self, documents, chunk_size=500, chunk_overlap=80):
        """构建 RAG 系统"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " "]
        )
        chunks = splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        prompt = ChatPromptTemplate.from_template(
            "你是一个知识问答助手。根据以下参考信息回答问题。\n"
            "如果参考信息中没有相关内容，请诚实地说'我不知道'。\n\n"
            "参考信息：\n{context}\n\n"
            "问题：{question}\n\n回答："
        )
        
        self.rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print(f"✅ RAG 系统构建完成！共 {len(chunks)} 个片段")
    
    def _format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    def query(self, question):
        """提问"""
        if not self.rag_chain:
            raise ValueError("请先调用 build() 构建系统")
        return self.rag_chain.invoke(question)
    
    def save(self, path="faiss_index"):
        """保存向量索引"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"✅ 索引已保存到 {path}")
    
    @classmethod
    def load(cls, api_key, path="faiss_index"):
        """从文件加载"""
        rag = cls(api_key)
        rag.vectorstore = FAISS.load_local(
            path, 
            DashScopeEmbeddings(model="text-embedding-v3"),
            allow_dangerous_deserialization=True
        )
        return rag

# === 使用示例 ===
if __name__ == "__main__":
    rag = SimpleRAG(api_key="你的API Key")
    
    # 构建
    docs = rag.load_documents("./knowledge_base")
    rag.build(docs)
    
    # 提问
    print(rag.query("RAG是什么？"))
    
    # 保存 & 下次加载
    rag.save()
    # 下次直接：rag = SimpleRAG.load("API Key", "faiss_index")
```

### ✅ 检查点
- [ ] 能成功运行 `SimpleRAG` 类
- [ ] 能用 `load()` 方法恢复之前保存的索引
- [ ] 理解封装的价值（下次直接 `pip install` 级别调用）

---

## Part 4：GitHub 更新 & 笔记（15分钟）

### 今日提交清单：
- `week-04/day-21/Day21-学习内容.md`（本文件）
- `week-04/day-21/Day21-学习内容-详细拆解.md`（拆解文档）
- `code/day21_rag_demo.py`（上面的完整代码）
- `code/rag_system.py`（SimpleRAG 类）
- 更新 `learning-log.md`
- 更新 `learning-progress.json`

```bash
cd /path/to/llm-learning-journey
git add week-04/day-21/
git add code/
git add learning-log.md
git commit -m "Day 21: RAG complete pipeline + retrieval optimization + SimpleRAG class"
git push origin main
```

---

## 📚 参考资源

| 来源 | 链接 | 说明 |
|------|------|------|
| LangChain RAG 教程 | https://python.langchain.com/docs/tutorials/rag/ | 官方最新教程 |
| FAISS GitHub | https://github.com/facebookresearch/faiss | 向量检索库文档 |
| 通义千问 Embedding API | https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-api-details | DashScope 文档 |
| RAG 优化技巧总结 | https://zhuanlan.zhihu.com/p/668792362 | 知乎：Retrieval 优化 |
| Chunk Size 如何选择 | https://www.pinecone.io/learn/chunking-strategies/ | Pinecone 教程 |

---

## 💡 今日要点

1. **RAG 的核心是检索质量**，不是 LLM — chunk 分割决定了一切
2. **MMR 检索**可以在相关性和多样性之间取得平衡
3. **元数据过滤**让检索更精准，特别是有多个数据源时
4. **封装成类**不是花架子，是工程化思维的体现
5. **保存索引**很关键 — 文档不变的话不需要每次重新向量化

> "RAG 不是技术问题，是信息架构问题。" — 想清楚你的知识怎么组织，效果自然就好。
