"""Tool-calling agent over a local Ollama model. No paid API, no card, no account."""
import json
import time
from pathlib import Path

import requests

from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
TRACE_PATH = Path(__file__).parent / "trace.jsonl"

SYSTEM_PROMPT = (
    "You are a research assistant for retail investors who can't afford a Bloomberg "
    "terminal. Use the tools available to answer with real numbers, not guesses. "
    "Always call a tool before stating a price, ratio, or filing date. Be concise."
)


def _log(event: dict):
    event["ts"] = time.time()
    with open(TRACE_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def run(task: str, max_turns: int = 6) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    _log({"type": "task_start", "task": task})

    for _ in range(max_turns):
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        message = resp.json()["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            _log({"type": "final_answer", "content": message.get("content", "")})
            return message.get("content", "")

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            fn = TOOL_FUNCTIONS.get(name)
            result = fn(**args) if fn else {"error": f"unknown tool {name}"}
            _log({"type": "tool_call", "tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "content": json.dumps(result)})

    _log({"type": "max_turns_exceeded"})
    return "[agent gave up after max turns without a final answer]"


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) or "What's AAPL's trailing P/E ratio?"
    print(run(task))
