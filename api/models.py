from pydantic import BaseModel
from typing import List


class Query(BaseModel):
    documents: str
    questions: List[str]


class Response(BaseModel):
    answers: List[str]