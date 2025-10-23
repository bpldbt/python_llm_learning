import os
from fastapi import FastAPI
from library import Library
from book import BookCreateModel
from typing import List
from fastapi import HTTPException

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough # 新增 import
from langchain_core.output_parsers import StrOutputParser # 新增 import
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


from api_mode import ResponseModel

# 创建 FastAPI 应用实例
app = FastAPI(title="个人藏书管理 API", description="一个用于管理个人书籍的简单 API")

# 1. 初始化 Ollama LLM 客户端，指向本地运行的 Llama3 模型
llm = OllamaLLM(model = "gemma3:1b")

# --- 新增：初始化嵌入模型 ---
# 这个模型专门用来将文本转换为向量
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# --- 新增：定义向量数据库的持久化路径 ---
# 我们将把向量数据存储在这个文件夹里
CHROMA_DB_PATH = "./chroma_db"


# 2. 创建一个提示词模板
#    模板中的 {topic} 是一个占位符，我们可以在运行时动态填充它
prompt_template_text = "你是一位博学的图书推荐官。请为主题为 '{topic}' 的书籍写一句吸引人的、不超过50字的推荐语。请使用中文回答！"
prompt_template = PromptTemplate.from_template(prompt_template_text)

# 3. 使用 LangChain 将 LLM 和 Prompt Template 链接起来，形成一个 Chain
#    这个 Chain 的作用是：接收一个 topic -> 格式化 prompt -> 调用 LLM -> 解析输出
recommendation_chain = prompt_template | llm


# --- 新增：定义 API 请求体模型 ---
class RecommendationRequest(BaseModel):
    topic: str

class RecommendationData(BaseModel):
    topic: str
    recommendation: str

# --- RAG 相关的 API 请求体模型 ---
class RagQueryRequest(BaseModel):
    question: str

# 创建一个全局的 Library 实例，在整个应用生命周期中共享
# 注意：这是一种简单的状态管理方式，适用于小型应用
library_manager = Library()

@app.get("/books", summary="获取所有书籍", response_model=ResponseModel[List[BookCreateModel]])
def get_all_books() -> ResponseModel:
    """返回图书馆中所有书籍的列表。"""
    books = library_manager.get_all_books()
    book_responses = [BookCreateModel(**book.to_dict()) for book in books]

    # FastAPI 会自动将 Book 对象转换为 JSON
    return ResponseModel(data=book_responses, code=0, message="成功获取所有书籍")


@app.get("/books/search", summary="搜索书籍", response_model=ResponseModel[List[BookCreateModel]])
def search_books(keyword: str):
    """根据关键词搜索书籍。"""
    found_books = library_manager.search_books(keyword)
    if not found_books:
        return ResponseModel(data=[], code=0, message="未找到匹配的书籍")
    return ResponseModel(data=[BookCreateModel(**book.to_dict()) for book in found_books], code=0, message="成功找到匹配的书籍")


@app.post("/books", summary="添加一本新书", response_model=ResponseModel[BookCreateModel])
def add_new_book(new_book: BookCreateModel):
    """接收书籍信息，创建一本新书并添加到图书馆。"""
    created_book = library_manager.add_book(new_book)
    return ResponseModel(data=BookCreateModel(**created_book.to_dict()), code=0, message="书籍添加成功！")

# --- 新增的 API 端点 ---
@app.post("/books/generate-recommendation", summary="生成书籍主题推荐语", response_model=ResponseModel[RecommendationData])
def generate_recommendation(request: RecommendationRequest):
    """
    接收一个书籍主题（例如 "科幻小说"），并使用大模型生成一句推荐语。
    """
    print(f"收到了生成推荐语的请求，主题是: {request.topic}")

    try:
        # 使用我们创建的 chain 来调用大模型
        # .invoke() 方法会执行整个链条，并将结果返回
        response_text = recommendation_chain.invoke({"topic": request.topic})
        
        print(f"从大模型收到的原始响应: {response_text}")
        
        return ResponseModel.success_response(data=RecommendationData(topic=request.topic, recommendation=response_text))

    except Exception as e:
        print(f"调用大模型时发生错误: {e}")
        # 在 FastAPI 中，可以抛出 HTTPException 来返回一个标准的错误响应
        return ResponseModel.error_response(code=500, message=f"调用大模型时发生错误: {str(e)}")
    

@app.post("/books/process-documents", summary="加载并分割知识库文档")
def process_documents():
    """
    演示 RAG 的第一步：从文件加载文档并将其分割成小块。
    """
    try:
        # 1. 加载文档
            loader = TextLoader("poems.txt", encoding="utf-8")
            documents = loader.load()
            print(f"加载了 {len(documents)} 个文档。")

            # 2. 创建文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=20,
                # separators=["\n\n", "\n", " ", ""]
            )

            # 3. 分割文档
            chunks = text_splitter.split_documents(documents)
            print(f"分割后得到 {len(chunks)} 个文档块。")

            if chunks:
                print("示例文档块内容：")
                print(chunks[0].page_content)

            # 返回分割后的信息给前端
            return {
            "message": "文档加载和分割成功！",
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "chunks_preview": [chunk.page_content for chunk in chunks[:3]] # 返回前3个块作为预览
            }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="知识库文件 'poems.txt' 未找到。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文档时发生错误: {str(e)}")
    

# --- 新-的 API 端点 ---
@app.post("/build-index", summary="构建或更新知识库的向量索引")
def build_knowledge_base_index():
    """
    执行完整的 RAG 索引构建流程：
    1. 加载文档 (Load)
    2. 分割文本 (Split)
    3. 文本嵌入 (Embed)
    4. 存入向量数据库 (Store)
    """
    knowledge_file = "poems.txt"

    if not os.path.exists(knowledge_file):
        raise HTTPException(status_code=404, detail=f"知识库文件 '{knowledge_file}' 未找到。")

    try:
        # 1. & 2. Load and Split
        print("--- 步骤 1 & 2: 加载并分割文档 ---")
        loader = TextLoader(knowledge_file, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            return {"message": "文档为空或分割后没有内容，未创建索引。"}

        print(f"文档被成功分割成 {len(chunks)} 个小块 (chunks)。")

        # 3. & 4. Embed and Store
        print("--- 步骤 3 & 4: 嵌入文本并存入 ChromaDB ---")
        
        # Chroma.from_documents 会自动处理嵌入和存储的过程
        # - chunks: 我们分割好的文本块
        # - embeddings: 我们初始化的嵌入模型实例
        # - persist_directory: 指定向量数据要持久化存储到的文件夹
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
        
        print(f"成功创建向量索引并持久化到 '{CHROMA_DB_PATH}'。")
        
        # 这个操作在服务器端完成，我们只需要返回一个成功的消息
        return {
            "message": "知识库向量索引构建成功！",
            "total_chunks_added": len(chunks)
        }

    except Exception as e:
        print(f"构建索引时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"构建索引时发生错误: {str(e)}")
    

# --- 最核心的新增 API 端点 ---
@app.post("/rag-query", summary="基于知识库的 RAG 问答")
def rag_query(request: RagQueryRequest):
    """
    接收一个问题，执行完整的 RAG 流程来生成答案：
    1. 从 ChromaDB 加载向量存储
    2. 创建检索器
    3. 构建 RAG 链
    4. 调用链来获取答案
    """
    if not os.path.exists(CHROMA_DB_PATH):
        raise HTTPException(status_code=404, detail=f"向量数据库 '{CHROMA_DB_PATH}' 不存在。请先调用 /build-index 端点来创建它。")

    try:
        # 1. 从持久化文件中加载向量存储
        print("--- 步骤 1: 加载已存在的 ChromaDB 向量存储 ---")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embeddings
        )

        # 2. 创建检索器 (Retriever)
        #    retriever 会根据问题，从 vectorstore 中找出最相关的文档
        print("--- 步骤 2: 创建检索器 ---")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # "k": 2 表示我们想检索回 2 个最相关的文档块


        # 在构建完整链之前，单独调用一次 retriever 来获取 context
        print("--- 步骤 2.5: 单独执行检索以进行调试 ---")
        retrieved_docs = retriever.invoke(request.question)
    
        # 打印检索到的上下文内容
        print("--- 检索到的上下文内容 ---")
        for doc in retrieved_docs:
            print(doc.page_content)
            print("-" * 20)

        # 3. 构建 RAG 提示词模板
        #    这个模板是 RAG 的核心，它指导 LLM 如何利用我们提供的上下文
        rag_prompt_template_text = """
        你是一个知识渊博的诗词助手。请根据下面提供的上下文信息来回答用户的问题。
        如果上下文中没有足够的信息来回答问题，就说你不知道。请不要编造答案。

        上下文:
        {context}

        问题:
        {question}

        回答:
        """
        rag_prompt = PromptTemplate.from_template(rag_prompt_template_text)

        # 4. 构建完整的 RAG 链 (Chain)
        #    这是 LangChain 表达式语言 (LCEL) 的强大之处
        print("--- 步骤 4: 构建并执行 RAG 链 ---")
        
        # 定义一个函数，用于将检索到的文档列表格式化为单一的字符串
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        # 5. 调用 RAG 链
        #    .invoke() 的输入会作为 RunnablePassthrough() 的值，即用户的原始问题
        answer = rag_chain.invoke(request.question)
        
        print(f"RAG 链成功执行，生成的回答是: {answer}")

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        print(f"RAG 查询时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"RAG 查询时发生错误: {str(e)}")
