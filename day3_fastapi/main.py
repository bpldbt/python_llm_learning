from fastapi import FastAPI
from library import Library
from book import BookCreateModel
from typing import List

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# 创建 FastAPI 应用实例
app = FastAPI(title="个人藏书管理 API", description="一个用于管理个人书籍的简单 API")

# 1. 初始化 Ollama LLM 客户端，指向本地运行的 Llama3 模型
llm = OllamaLLM(model = "llama3:8b")

# 2. 创建一个提示词模板
#    模板中的 {topic} 是一个占位符，我们可以在运行时动态填充它
prompt_template_text = "你是一位博学的图书推荐官。请为主题为 '{topic}' 的书籍写一句吸引人的、不超过50字的推荐语。"
prompt_template = PromptTemplate.from_template(prompt_template_text)

# 3. 使用 LangChain 将 LLM 和 Prompt Template 链接起来，形成一个 Chain
#    这个 Chain 的作用是：接收一个 topic -> 格式化 prompt -> 调用 LLM -> 解析输出
recommendation_chain = prompt_template | llm


# --- 新增：定义 API 请求体模型 ---
class RecommendationRequest(BaseModel):
    topic: str

# 创建一个全局的 Library 实例，在整个应用生命周期中共享
# 注意：这是一种简单的状态管理方式，适用于小型应用
library_manager = Library()

@app.get("/books", summary="获取所有书籍")
def get_all_books():
    """返回图书馆中所有书籍的列表。"""
    books = library_manager.get_all_books()
    # FastAPI 会自动将 Book 对象转换为 JSON
    return {"books": [book.to_dict() for book in books]}

@app.get("/books/search", summary="搜索书籍")
def search_books(keyword: str):
    """根据关键词搜索书籍。"""
    found_books = library_manager.search_books(keyword)
    if not found_books:
        return {"message": f"未找到与 '{keyword}' 相关的书籍。"}
    return {"results": [book.to_dict() for book in found_books]}

@app.post("/books", summary="添加一本新书")
def add_new_book(new_book: BookCreateModel):
    """接收书籍信息，创建一本新书并添加到图书馆。"""
    created_book = library_manager.add_book(new_book)
    return {"message": "书籍添加成功！", "book": created_book.to_dict()}

# --- 新增的 API 端点 ---
@app.post("/books/generate-recommendation", summary="生成书籍主题推荐语")
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
        
        return {"topic": request.topic, "recommendation": response_text}

    except Exception as e:
        print(f"调用大模型时发生错误: {e}")
        # 在 FastAPI 中，可以抛出 HTTPException 来返回一个标准的错误响应
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"调用大模型时发生错误: {str(e)}")