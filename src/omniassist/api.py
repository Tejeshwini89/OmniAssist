from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.omniassist.generator import generate_answer


app = FastAPI(
    title="OmniAssist",
    description="Enterprise GenAI Assistant using RAG",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask OmniAssist",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


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
    try:
        return generate_answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The answer generation service is temporarily unavailable.",
        ) from exc
