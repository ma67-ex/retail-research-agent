"""Minimal web UI: a chat box over the existing agent, no build step."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import run, TRACE_PATH

app = FastAPI()
STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    task: str


def _trace_len() -> int:
    return len(TRACE_PATH.read_text().splitlines()) if TRACE_PATH.exists() else 0


def _entries_since(start_len: int) -> list:
    lines = TRACE_PATH.read_text().splitlines() if TRACE_PATH.exists() else []
    return [json.loads(l) for l in lines[start_len:]]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    start_len = _trace_len()
    answer = run(req.task)
    trace = [e for e in _entries_since(start_len) if e.get("type") == "tool_call"]
    return {"answer": answer, "trace": trace}
