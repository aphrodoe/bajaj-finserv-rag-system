from pydantic import BaseModel

class Query(BaseModel):
    documents: str
    questions: list[str]

class Response(BaseModel):
    answers: list[str]
