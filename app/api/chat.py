# api/chat.py
from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.services.agent import DocuRAGAgent

router = APIRouter()
agent = DocuRAGAgent()

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "thread-12"

@router.post("/chat")
async def chat(request: ChatRequest):
    reply = agent.run(request.message, request.thread_id)
    return {"response": reply}
