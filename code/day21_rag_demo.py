"""
Day 21: RAG 完整演示
最小可运行的 RAG Pipeline
"""
import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def create_test_docs(directory="./knowledge_base"):
    """创建测试文档"""
    os.makedirs(directory, exist_ok=True)
    
    docs = {
        "rag_intro.txt": """RAG（检索增强生成，Retrieval-Augmented Generation）是一种结合信息检索和文本生成的AI技术。
它的工作流程是：当用户提问时，系统先从知识库中检索相关文档片段，然后将这些片段作为上下文一起交给大语言模型生成回答。

RAG 的优势：
1. 减少幻觉：基于检索到的事实回答，减少LLM编造
2. 知识更新方便：只需更新文档库，不需要重新训练模型
3. 可追溯：可以回答时列出参考来源
4. 成本更低：不需要微调，直接利用已有知识

RAG 的核心组件：
- Document Loader：加载各种格式的文档（PDF、TXT、Markdown等）
- Text Splitter：将文档切成小块（chunks）
- Embeddings：将文本转换为向量
- Vector Store：存储和检索向量（FAISS、ChromaDB等）
- Retriever：根据查询向量检索相关文档
- LLM：根据检索到的上下文生成回答""",
        
        "vector_db.txt": """向量数据库（Vector Database）是一种专门用于存储和检索向量数据的数据库。

在AI应用中，向量数据库主要用于：
1. 语义搜索：将文本转换为向量，通过向量相似度进行语义匹配
2. 推荐系统：将用户和物品的特征表示为向量，进行相似度匹配
3. RAG系统：存储文档的向量表示，快速检索相关上下文

常见的向量数据库：
- FAISS（Facebook）：轻量级，适合本地使用，速度快
- ChromaDB：Python友好，适合开发原型
- Milvus（Zilliz）：企业级，支持分布式部署
- Pinecone：云服务，开箱即用

向量相似度计算方式：
- 余弦相似度（Cosine Similarity）：最常用，计算两个向量的夹角余弦
- 欧氏距离（Euclidean Distance）：计算两个向量之间的直线距离
- 点积（Dot Product）：计算两个向量的内积

FAISS 使用步骤：
1. 准备文档和Embeddings模型
2. 将文档转换为向量
3. 构建 FAISS 索引
4. 查询时：将问题转为向量 → 在索引中查找最近邻 → 返回对应文档
5. 将返回的文档作为上下文交给LLM生成回答""",
        
        "chunk_strategy.txt": """文本分块策略（Chunking Strategy）是RAG系统中最重要的环节之一。

为什么需要分块：
- LLM有上下文窗口限制（通常4K-128K token）
- 检索需要精确到片段级别
- 减少无关信息干扰

分块关键参数：
- chunk_size：每个片段的大小（字符或token数）
- chunk_overlap：相邻片段的重叠部分，保持上下文连贯

常见分块策略：

1. 固定长度分块（Fixed-Size Chunking）
   - 最简单：按固定字符数或token数切分
   - 优点：实现简单，可控
   - 缺点：可能在语义边界处切断内容
   - 适用场景：结构均匀的文本
   
2. 递归字符分块（Recursive Character Splitting）
   - 按段落、句子、词语的顺序递归切分
   - 优先在段落边界切，其次是句子，最后是词语
   - 优点：尽量保持语义完整性
   - 缺点：实现稍复杂
   - 适用场景：大多数自然语言文本
   
3. 语义分块（Semantic Chunking）
   - 按语义相似度动态分块
   - 相邻段落语义差异大时就切分
   - 优点：语义边界最准确
   - 缺点：计算开销大
   - 适用场景：对检索质量要求极高的场景
   
4. 结构分块（Document-Specific Chunking）
   - 按文档结构分块（如Markdown的标题层级、HTML的标签）
   - 优点：保持文档结构
   - 适用场景：有明确结构的文档（Markdown、HTML、代码等）

Chunk Size 选择经验：
- 短文/代码：200-500 tokens
- 技术文档：500-1000 tokens
- 长文章：1000-2000 tokens
- 对话记录：100-300 tokens

Chunk Overlap 选择：
- 一般为 chunk_size 的 10-20%
- 太小：可能丢失上下文关联
- 太大：增加冗余，浪费token""",
    }
    
    for filename, content in docs.items():
        filepath = os.path.join(directory, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 创建 {filepath}")
    
    return directory


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


def demonstrate_chunk_size_comparison(documents, embeddings):
    """演示不同 chunk_size 的效果"""
    print("\n" + "="*60)
    print("Chunk Size 对比实验")
    print("="*60)
    
    for chunk_size in [200, 500, 800]:
        overlap = chunk_size // 5
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "，", " "]
        )
        chunks = splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        results = retriever.invoke("RAG的优缺点是什么？")
        
        print(f"\nchunk_size={chunk_size}, overlap={overlap}:")
        print(f"  文档被分成 {len(chunks)} 个片段")
        print(f"  检索到 {len(results)} 个结果")
        print(f"  第一个片段长度: {len(results[0].page_content)} 字符")
        if results:
            preview = results[0].page_content[:100].replace("\n", " ")
            print(f"  预览: {preview}...")


def demonstrate_mmr(documents, embeddings):
    """演示 MMR 检索"""
    print("\n" + "="*60)
    print("MMR vs 普通检索 对比")
    print("="*60)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 普通检索
    retriever_normal = vectorstore.as_retriever(search_kwargs={"k": 3})
    results_normal = retriever_normal.invoke("向量数据库")
    
    # MMR 检索
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )
    results_mmr = retriever_mmr.invoke("向量数据库")
    
    print(f"\n普通检索:")
    for i, doc in enumerate(results_normal):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  {i+1}. {preview}...")
    
    print(f"\nMMR 检索:")
    for i, doc in enumerate(results_mmr):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  {i+1}. {preview}...")


def main():
    # 获取 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY", input("请输入通义千问 API Key: "))
    if not api_key or api_key.startswith("请输入"):
        print("❌ 未提供 API Key，无法运行")
        print("请设置环境变量: export DASHSCOPE_API_KEY='your-key'")
        return
    
    # 1. 创建测试文档
    print("\n" + "="*60)
    print("Day 21: RAG 完整演示")
    print("="*60)
    
    doc_dir = create_test_docs()
    
    # 2. 初始化
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    
    # 3. 加载文档
    loader = DirectoryLoader(doc_dir, glob="**/*.txt")
    documents = loader.load()
    print(f"\n✅ 加载了 {len(documents)} 个文档")
    
    # 4. Chunk Size 对比
    demonstrate_chunk_size_comparison(documents, embeddings)
    
    # 5. MMR 对比
    demonstrate_mmr(documents, embeddings)
    
    # 6. 构建完整 RAG 系统
    print("\n" + "="*60)
    print("构建完整 RAG 系统")
    print("="*60)
    
    rag = SimpleRAG(api_key=api_key)
    rag.build(documents, chunk_size=500)
    
    # 7. 测试问答
    questions = [
        "RAG是什么？有什么优势？",
        "向量数据库在RAG中起什么作用？",
        "如何选择 chunk size？",
    ]
    
    for q in questions:
        print(f"\n❓ 问题: {q}")
        try:
            answer = rag.query(q)
            print(f"💡 回答: {answer[:500]}...")
        except Exception as e:
            if "API_KEY" in str(e) or "authentication" in str(e).lower():
                print(f"⚠️ API Key 无效: {e}")
                print("请检查你的通义千问 API Key 是否正确")
            else:
                print(f"⚠️ 出错: {e}")
            break
    
    # 8. 保存索引
    rag.save("faiss_index")
    print("\n✅ 演示完成！")
    print("\n下次可以直接加载: rag = SimpleRAG.load('your-key', 'faiss_index')")


if __name__ == "__main__":
    main()
