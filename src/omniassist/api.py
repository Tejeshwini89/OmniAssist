from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.omniassist.generator import generate_answer


app = FastAPI(
    title="OmniAssist",
    description="Enterprise GenAI Assistant using RAG",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask OmniAssist",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "OmniAssist",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = generate_answer(
        request.question,
    )

    return result