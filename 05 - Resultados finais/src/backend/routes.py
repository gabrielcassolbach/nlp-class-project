from fastapi import APIRouter
from pydantic import BaseModel
from rag_service import RAGService

class Routes:
    def __init__(self):
        self.router = APIRouter()
        self.rag = RAGService(False)

        @self.router.post("/message")
        def ask_route(req: Query):
            answer = self.rag.ask(req.query, req.top_k)
            return {"answer": answer}

class Query(BaseModel):
    query: str
    top_k: int = 3