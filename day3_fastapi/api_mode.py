from pydantic import BaseModel
from typing import TypeVar, Generic, Optional, Any

T = TypeVar('T')

class ResponseModel(BaseModel, Generic[T]):
    code: int
    message: str
    data: Optional[T] = None

    @property
    def is_success(self) -> bool:
        return self.code == 0
    
    @classmethod
    def success_response(cls, data: T, message: str = "成功"):
        """创建一个成功的响应。"""
        return cls(code=0, message=message, data=data)

    @classmethod
    def error_response(cls, code: int, message: str):
        """创建一个失败的响应。"""
        return cls(code=code, message=message, data=None)
