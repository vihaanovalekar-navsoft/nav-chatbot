"""FastAPI app for the NavAI assistant."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from navai.pipeline import NavAIChat
from navai.settings import ROOT

MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 12000


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[ChatMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )

    @field_validator("history")
    @classmethod
    def check_history_size(cls, history):
        if sum(len(message.content) for message in history) > MAX_HISTORY_CHARS:
            raise ValueError(
                f"History must be at most {MAX_HISTORY_CHARS} characters."
            )
        return history


@asynccontextmanager
async def lifespan(app):
    app.state.chat = NavAIChat()
    yield


app = FastAPI(title="NavAI Assistant", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return app.state.chat.answer(
        question,
        [message.model_dump() for message in request.history],
    )


if __name__ == "__main__":
    from navai.server import main

    main(app)