# library.py
import json
from book import Book # 注意这里导入的是我们自己的 Book 类

class Library:
    def __init__(self, filename="library_data.json"):
        self.filename = filename
        self.books = self._load_books()

    def _load_books(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Book(b['title'], b['author'], b['year'], b['tags']) for b in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_books(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump([book.to_dict() for book in self.books], f, indent=4, ensure_ascii=False)

    def add_book(self, book_data):
        """接收一个 BookCreateModel 对象，创建 Book 实例并保存。"""
        new_book = Book(
            title=book_data.title,
            author=book_data.author,
            year=book_data.year,
            tags=book_data.tags
        )
        self.books.append(new_book)
        self._save_books()
        return new_book # 返回新创建的书籍对象

    def get_all_books(self):
        """不再打印，而是返回所有书籍的列表。"""
        return self.books

    def search_books(self, keyword):
        """不再打印，而是返回搜索结果列表。"""
        keyword_lower = keyword.lower()
        found_books = [
            book for book in self.books
            if keyword_lower in book.title.lower()
            or keyword_lower in book.author.lower()
            or any(keyword_lower in tag.lower() for tag in book.tags)
        ]
        return found_books

