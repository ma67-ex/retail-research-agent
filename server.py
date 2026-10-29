"""Minimal web UI + JSON API over the agent. Stateless: conversation memory lives in the browser."""
import datetime
import hmac
import math
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent import CALC_TOOL, FIRM_NAME, MODEL, NEED_COMPANY, PRODUCT_NAME, run
from tools import TOOL_FUNCTIONS

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
STATIC_DIR = Path(__file__).parent / "static"


def _brand_color(raw: str) -> str | None:
    return raw if re.fullmatch(r"#[0-9a-fA-F]{6}", raw) else None


def _embed_origins(raw: str) -> list:
    return [o for o in re.split(r"[,\s]+", raw) if re.fullmatch(r"https?://[A-Za-z0-9.-]+(:\d+)?", o)]


def _minutes_saved(raw: str) -> float:
    try:
        v = float(raw)
    except ValueError:
        return 3.0
    return v if 0 < v <= 60 else 3.0


BRAND_COLOR = _brand_color(os.environ.get("BRAND_COLOR", "").strip())
STATS_TOKEN = os.environ.get("STATS_TOKEN", "").strip()
MINUTES_PER_QUESTION = _minutes_saved(os.environ.get("MINUTES_SAVED_PER_QUESTION", "3").strip())
_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; object-src 'none'; "
        "frame-ancestors " + " ".join(["'self'", *_embed_origins(os.environ.get("EMBED_ORIGINS", ""))]))
_SECURITY_HEADERS = [
    (b"content-security-policy", _CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]
_LOOPBACK = {"127.0.0.1", "::1"}
_PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")

RATE_LIMIT = 20
RATE_WINDOW = 60.0
# Default: in memory, per process. Set RATE_LIMIT_DB to a file path and every worker process shares one limit.
_hits: dict = defaultdict(deque)
_hits_lock = threading.Lock()
RATE_LIMIT_DB = os.environ.get("RATE_LIMIT_DB", "").strip()
# Optional shared secret for /chat. Unset: open, as before (run on localhost or behind a proxy with access control).
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "").strip()


class _SharedLimiter:
    """Sliding-window limit kept in a SQLite file, so several uvicorn workers count the same requests."""

    def __init__(self, path: str):
        self.path, self._local = path, threading.local()

    def _conn(self):
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS hits (ip TEXT NOT NULL, t REAL NOT NULL)")
            c.execute("CREATE INDEX IF NOT EXISTS hits_ip_t ON hits (ip, t)")
            self._local.c = c
        return c

    def allow(self, ip: str, limit: int, window: float, now: float) -> bool:
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")  # take the write lock first so two workers cannot both see room for one more
        try:
            c.execute("DELETE FROM hits WHERE t < ?", (now - window,))
            (n,) = c.execute("SELECT COUNT(*) FROM hits WHERE ip = ?", (ip,)).fetchone()
            if n < limit:
                c.execute("INSERT INTO hits (ip, t) VALUES (?, ?)", (ip, now))
            c.execute("COMMIT")
            return n < limit
        except Exception:
            c.execute("ROLLBACK")
            raise


_shared = _SharedLimiter(RATE_LIMIT_DB) if RATE_LIMIT_DB else None

# ponytail: in-memory, resets on restart, single-process
_stats_lock = threading.Lock()
_started = datetime.datetime.now(datetime.timezone.utc)
_COUNTERS = ("questions", "answered", "answered_with_data", "fully_verified", "flagged_numbers",
             "advice_deflected", "clarifying_questions", "model_errors", "served_from_cache",
             "rejected_invalid", "rate_limited")
_counts = dict.fromkeys(_COUNTERS, 0)
_tool_calls: dict = defaultdict(int)
_tool_errors: dict = defaultdict(int)
_latencies: deque = deque(maxlen=1000)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    task: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    history: list[Message] | None = Field(default=None, max_length=10)


def _bump(*keys: str):
    with _stats_lock:
        for k in keys:
            _counts[k] += 1


def _record(result: dict):
    # aggregate-only: reads shapes and tool names, never copies question, answer, args or result text
    try:
        calls = [c for c in result.get("tool_calls") or [] if isinstance(c, dict)]
        seen = []
        for c in calls:
            name = c.get("tool") if c.get("tool") in TOOL_FUNCTIONS else "other"  # model-supplied names are untrusted
            res = c.get("result") if isinstance(c.get("result"), dict) else {"error": True}
            seen.append((name, c.get("tool"), res))
        answered = result.get("error") is None
        data = answered and any(raw != CALC_TOOL and "error" not in r for _, raw, r in seen)
        answer = result.get("answer")
        answer = answer.strip() if isinstance(answer, str) else ""
        keys = ["questions"]
        keys += ["answered"] * answered
        keys += ["answered_with_data"] * data
        flagged = bool(result.get("unverified_numbers") or result.get("unverified_quotes"))
        keys += ["fully_verified"] * (data and not flagged)
        keys += ["flagged_numbers"] * flagged
        keys += ["advice_deflected"] * bool(result.get("advice_refused"))
        keys += ["clarifying_questions"] * (answered and not calls and (answer.endswith("?") or answer == NEED_COMPANY))
        keys += ["model_errors"] * (not answered)
        keys += ["served_from_cache"] * any(r.get("stale") for _, _, r in seen)
        ms = result.get("latency_ms")
        with _stats_lock:
            for k in keys:
                _counts[k] += 1
            for name, _, r in seen:
                _tool_calls[name] += 1
                _tool_errors[name] += "error" in r
            if answered and isinstance(ms, (int, float)) and not isinstance(ms, bool):
                _latencies.append(ms)
    except Exception:
        pass


def _pct(sorted_ms: list, p: float):
    return sorted_ms[max(0, math.ceil(p * len(sorted_ms)) - 1)] if sorted_ms else None


def _ratio(n: int, d: int):
    return n / d if d else None


def _snapshot() -> dict:
    with _stats_lock:
        c, calls, errors, lat = dict(_counts), dict(_tool_calls), dict(_tool_errors), sorted(_latencies)
    minutes = f"{MINUTES_PER_QUESTION:g}"
    return {
        "since": _started.isoformat(timespec="seconds"),
        **c,
        "rates": {
            "answered": _ratio(c["answered"], c["questions"]),
            "live_data": _ratio(c["answered_with_data"], c["answered"]),
            "fully_verified": _ratio(c["fully_verified"], c["answered"]),
            "model_errors": _ratio(c["model_errors"], c["questions"]),
        },
        "tools": {n: {"calls": calls[n], "errors": errors.get(n, 0)} for n in sorted(calls)},
        "latency_ms": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "max": lat[-1] if lat else None,
                       "samples": len(lat)},
        "estimated_minutes_saved": round(c["answered"] * MINUTES_PER_QUESTION, 1),
        "assumption": f"{minutes} min of support/analyst time per answered question (set MINUTES_SAVED_PER_QUESTION)",
    }


def _error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=headers)


def stats_access(request: Request):
    if STATS_TOKEN:
        if not hmac.compare_digest(request.headers.get("x-stats-token", "").encode(), STATS_TOKEN.encode()):
            raise HTTPException(401, "Missing or wrong stats token.")
    elif (request.client.host if request.client else None) not in _LOOPBACK \
            or any(h in request.headers for h in _PROXY_HEADERS):
        raise HTTPException(403, "Stats are limited to direct local access. Set STATS_TOKEN to read them remotely.")


def chat_access(request: Request):
    if ACCESS_TOKEN and not hmac.compare_digest(request.headers.get("x-access-token", "").encode(), ACCESS_TOKEN.encode()):
        raise HTTPException(401, "An access token is needed to use this assistant.")


def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    if _shared:
        if not _shared.allow(ip, RATE_LIMIT, RATE_WINDOW, time.time()):  # wall clock: processes do not share a monotonic one
            _bump("rate_limited")
            raise HTTPException(429, "Too many requests. Wait a minute and try again.",
                                headers={"Retry-After": str(int(RATE_WINDOW))})
        return
    now = time.monotonic()
    with _hits_lock:
        hits = _hits[ip]
        while hits and now - hits[0] > RATE_WINDOW:
            hits.popleft()
        if len(hits) >= RATE_LIMIT:
            _bump("rate_limited")
            raise HTTPException(429, "Too many requests. Wait a minute and try again.",
                                headers={"Retry-After": str(int(RATE_WINDOW))})
        hits.append(now)


@app.exception_handler(RequestValidationError)
async def _invalid(request: Request, exc: RequestValidationError):
    _bump("rejected_invalid")
    # loc + msg only: pydantic's default payload echoes the offending input back
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    return _error(422, f"Invalid request: {field + ': ' if field else ''}{first.get('msg', 'bad input')}")


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, str(exc.detail), getattr(exc, "headers", None))


@app.exception_handler(Exception)
async def _crash(request: Request, exc: Exception):
    return _error(500, "Something went wrong on our side. Please try again.")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers=_REVALIDATE)


@app.get("/config")
def config():
    return JSONResponse({"product_name": PRODUCT_NAME, "firm_name": FIRM_NAME or None,
                         "brand_color": BRAND_COLOR, "model": MODEL, "requires_token": bool(ACCESS_TOKEN)},
                        headers={"Cache-Control": "no-store"})


@app.get("/stats", dependencies=[Depends(stats_access)])
def stats():
    return JSONResponse(_snapshot(), headers={"Cache-Control": "no-store"})


@app.get("/stats/dashboard")
def stats_dashboard():
    return FileResponse(STATIC_DIR / "stats.html", headers=_REVALIDATE)


# Unversioned files: let browsers keep them but re-check the ETag on every load.
_REVALIDATE = {"Cache-Control": "no-cache"}
_ASSETS = {"theme.css": "text/css; charset=utf-8", "theme.js": "text/javascript; charset=utf-8"}


@app.get("/assets/{name}")
def asset(name: str):
    if name not in _ASSETS:
        raise HTTPException(404, "Not found")
    return FileResponse(STATIC_DIR / name, media_type=_ASSETS[name], headers=_REVALIDATE)


@app.post("/chat", dependencies=[Depends(rate_limit), Depends(chat_access)])  # limit first, so token guessing is throttled too
def chat(req: ChatRequest):
    try:
        result = run(req.task, [m.model_dump() for m in req.history or []])
    except Exception:
        _record({"error": "crash"})
        raise
    _record(result)
    return JSONResponse(result, status_code=503 if result["error"] else 200,
                        headers={"Cache-Control": "no-store"})


class _SecurityHeaders:
    # outermost ASGI wrapper: add_middleware sits inside ServerErrorMiddleware, which builds the 500 response
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *_SECURITY_HEADERS]}
            await send(message)

        await self.inner(scope, receive, send_with_headers)


app = _SecurityHeaders(app)
