# book.py
from pydantic import BaseModel
from typing import List, Optional # 导入类型提示工具

# 这个类用于 API 创建书籍时接收请求体 (Request Body)
class BookCreateModel(BaseModel):
    title: str
    author: str
    year: int
    tags: Optional[List[str]] = [] # tags 是可选的，默认为空列表

# 这是我们原有的数据模型类，保持不变
class Book:
    """表示一本书的数据模型。"""
    def __init__(self, title, author, year, tags=None):
        self.title = title
        self.author = author
        self.year = year
        self.tags = tags if tags is not None else []

    def __str__(self):
        tags_str = ", ".join(self.tags)
        return f"《{self.title}》 by {self.author} ({self.year}) [标签: {tags_str}]"

    def to_dict(self):
        return {
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "tags": self.tags
        }