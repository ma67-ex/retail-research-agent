"""Tests for the white-label, usage-stats and security layer in server.py. No model, no network, no GPU.

Each scenario runs server.py in its own process (env differs per scenario) with run() replaced by a script.
Run: ./.venv/bin/python tests/test_business.py
"""
import re
import datetime
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

SPEC_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; object-src 'none'; "
    "frame-ancestors "
)
ASSUMPTION = "{} min of support/analyst time per answered question (set MINUTES_SAVED_PER_QUESTION)"
SECRETS = ["SECRETQ-9f3a", "ZEBRA-ANSWER", "AAPL", "secret-ticker", "yfinance-ZZ", "127.0.0.1"]
ENV_KEYS = ["FIRM_NAME", "BRAND_COLOR", "STATS_TOKEN", "EMBED_ORIGINS", "MINUTES_SAVED_PER_QUESTION", "TRACE_LOG"]


def _quote(error=None, stale=False):
    res = {"error": "boom", "source": "yfinance-ZZ"} if error else {"price": 1.0, "source": "yfinance-ZZ", "as_of": "2026-10-21"}
    if stale:
        res["stale"] = True
    return {"tool": "get_quote", "args": {"ticker": "AAPL"}, "result": res, "ms": 5}


def _calc():
    return {"tool": "calculator", "args": {"expression": "1+1"}, "result": {"result": 2, "source": "calculator"}, "ms": 1}


def _out(answer="ZEBRA-ANSWER AAPL is $1.00", calls=(), bad=(), advice=False, error=None, ms=100):
    return {"answer": answer, "tool_calls": list(calls), "unverified_numbers": list(bad),
            "advice_refused": advice, "model": "fake", "latency_ms": ms, "error": error}


def _scenarios():
    from agent import NEED_COMPANY
    return {
        "verified": _out(calls=[_quote(), _calc()], ms=100),
        "flagged": _out(calls=[_quote()], bad=["42.5"], ms=200),
        "advice": _out(calls=[_quote()], advice=True, ms=300),
        "clarify": _out("Which company do you mean?", ms=50),
        "need": _out(NEED_COMPANY, ms=60),
        "calc": _out(calls=[_calc()], ms=400),
        "modelerr": _out("", error="The local model isn't running.", ms=9999),
        "stale": _out(calls=[_quote(stale=True)], ms=20),
        "toolerr": _out(calls=[_quote(error=True)], ms=500),
        "weird": _out(calls=[{"tool": "AAPL secret-ticker", "args": {}, "result": {"error": "unknown tool"}, "ms": 1}], ms=600),
    }


def _serve(port):
    import uvicorn
    import server

    scripts = _scenarios()

    def fake_run(task, history=None):
        name = task.split("]")[0].lstrip("[")
        if name == "crash":
            raise ValueError("secret detail")
        return scripts.get(name) or _out(calls=[_quote()])

    server.run = fake_run
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="critical")


def _free_port():
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        if port != 8010:
            return port


S = requests.Session()
S.trust_env = False
_procs = {}


def _start(name, **env):
    if name in _procs:
        return _procs[name][1]
    port = _free_port()
    child_env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    child_env.update(env, OLLAMA_URL="http://127.0.0.1:9/api/chat")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve", str(port)],
                            env=child_env, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    _procs[name] = (proc, base)
    for _ in range(300):
        assert proc.poll() is None, f"server {name} exited early"
        try:
            if S.get(base + "/config", timeout=2).status_code == 200:
                return base
        except requests.RequestException:
            time.sleep(0.2)
    raise AssertionError(f"server {name} did not start")


def _plain():
    return _start("plain")


def _branded():
    return _start("branded", FIRM_NAME="Demo Brokerage", BRAND_COLOR="#1A2b3C", STATS_TOKEN="s3cr3t-token",
                  EMBED_ORIGINS="https://broker.example.com https://app.partner.io:8443 https://ok.com; script-src *",
                  MINUTES_SAVED_PER_QUESTION="5")


def _hostile():
    return _start("hostile", BRAND_COLOR="red;}</style><script>", EMBED_ORIGINS="javascript:alert(1) https://ok.com;x *",
                  MINUTES_SAVED_PER_QUESTION="abc", FIRM_NAME="   ")


def _chat(base, name, text="what is the price"):
    return S.post(base + "/chat", json={"task": f"[{name}] SECRETQ-9f3a {text} AAPL"}, timeout=30)


def _stats(base, **headers):
    return S.get(base + "/stats", headers=headers, timeout=10)


def _json_error(r, status):
    assert r.status_code == status, (r.status_code, r.text[:200])
    assert r.headers["content-type"].startswith("application/json")
    assert isinstance(r.json()["error"], str) and r.json()["error"]
    return r.json()


def _assert_secure_headers(r, frame="'self'"):
    h = r.headers
    assert h["x-content-type-options"] == "nosniff", (r.url, dict(h))
    assert h["referrer-policy"] == "no-referrer", r.url
    assert h["permissions-policy"] == "camera=(), microphone=(), geolocation=()", r.url
    assert h["content-security-policy"] == SPEC_CSP + frame, (r.url, h["content-security-policy"])
    assert "x-frame-options" not in h, r.url


def test_helper_validation():
    import server
    for ok in ["#000000", "#FFFFFF", "#1a2B3c"]:
        assert server._brand_color(ok) == ok
    for bad in ["", "#12345", "#1234567", "#GGGGGG", "abcdef", "#abcdef\n", " #abcdef", "red", "#abc", "#abcdef;x"]:
        assert server._brand_color(bad) is None, bad
    assert server._embed_origins("https://a.com http://b.io:80,https://c-d.example.org") == \
        ["https://a.com", "http://b.io:80", "https://c-d.example.org"]
    for bad in ["https://ok.com; script-src *", "javascript:alert(1)", "https://a.com/path", "https://a.com:x",
                "*", "https://", "ftp://a.com", "https://a.com\r\nX-Evil: 1", "https://exa mple.com"]:
        got = server._embed_origins(bad)
        assert not any(";" in o or "*" in o or "/path" in o or "evil" in o.lower() for o in got), (bad, got)
    assert server._embed_origins("https://a.com\nhttps://b.com") == ["https://a.com", "https://b.com"]
    for raw, want in [("3", 3.0), ("0.5", 0.5), ("60", 60.0), ("7", 7.0), ("1e1", 10.0), ("0", 3.0), ("-1", 3.0),
                      ("61", 3.0), ("nan", 3.0), ("inf", 3.0), ("abc", 3.0), ("", 3.0)]:
        assert server._minutes_saved(raw) == want, raw


def test_config_defaults():
    base = _plain()
    r = S.get(base + "/config", timeout=10)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert set(body) == {"product_name", "firm_name", "brand_color", "model", "requires_token"}
    assert body["requires_token"] is False  # no ACCESS_TOKEN set: the chat stays open
    assert body["product_name"] == "Retail Research Agent" and body["firm_name"] is None
    assert body["brand_color"] is None and isinstance(body["model"], str) and body["model"]


def test_config_branded():
    r = S.get(_branded() + "/config", timeout=10)
    assert r.json()["firm_name"] == "Demo Brokerage" and r.json()["brand_color"] == "#1A2b3C"
    assert r.headers["cache-control"] == "no-store"


def test_config_hostile_env_not_echoed():
    base = _hostile()
    r = S.get(base + "/config", timeout=10)
    assert r.json()["brand_color"] is None and r.json()["firm_name"] is None
    assert "red" not in r.text and "script" not in r.text
    _assert_secure_headers(r)


def test_stats_counts_and_privacy():
    base = _plain()
    for name in ["verified", "flagged", "advice", "clarify", "need", "calc", "modelerr", "stale", "toolerr", "weird"]:
        r = _chat(base, name)
        assert r.status_code == (503 if name == "modelerr" else 200), (name, r.status_code, r.text[:200])
    r = _chat(base, "crash")
    assert _json_error(r, 500) and "secret" not in r.text
    r = _stats(base)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    s = r.json()
    want = {"questions": 11, "answered": 9, "answered_with_data": 4, "fully_verified": 3, "flagged_numbers": 1,
            "advice_deflected": 1, "clarifying_questions": 2, "model_errors": 2, "served_from_cache": 1,
            "rejected_invalid": 0, "rate_limited": 0}
    assert {k: s[k] for k in want} == want, s
    assert s["fully_verified"] <= s["answered_with_data"] <= s["answered"] <= s["questions"]
    assert s["tools"] == {"calculator": {"calls": 2, "errors": 0}, "get_quote": {"calls": 5, "errors": 1},
                          "other": {"calls": 1, "errors": 1}}, s["tools"]
    assert s["latency_ms"] == {"p50": 200, "p95": 600, "max": 600, "samples": 9}, s["latency_ms"]
    assert s["estimated_minutes_saved"] == 27.0 and s["assumption"] == ASSUMPTION.format(3)
    rates = s["rates"]
    assert abs(rates["answered"] - 9 / 11) < 1e-9 and abs(rates["live_data"] - 4 / 9) < 1e-9
    assert abs(rates["fully_verified"] - 3 / 9) < 1e-9 and abs(rates["model_errors"] - 2 / 11) < 1e-9
    since = datetime.datetime.fromisoformat(s["since"])
    assert since.utcoffset() == datetime.timedelta(0)
    assert 0 <= (datetime.datetime.now(datetime.timezone.utc) - since).total_seconds() < 600
    assert set(s) == set(want) | {"since", "rates", "tools", "latency_ms", "estimated_minutes_saved", "assumption"}
    for secret in SECRETS:
        assert secret not in r.text, secret


def test_stats_rates_null_when_empty():
    r = _stats(_start("empty"))
    s = r.json()
    assert s["questions"] == 0 and s["rates"] == {"answered": None, "live_data": None, "fully_verified": None,
                                                   "model_errors": None}
    assert s["latency_ms"] == {"p50": None, "p95": None, "max": None, "samples": 0} and s["tools"] == {}
    assert s["estimated_minutes_saved"] == 0.0


def test_stats_access_without_token():
    base = _plain()
    assert _stats(base).status_code == 200
    cases = [{"X-Forwarded-For": "203.0.113.9"}, {"X-Forwarded-For": "127.0.0.1"}, {"X-Real-IP": "127.0.0.1"},
             {"X-Real-IP": "203.0.113.9"}, {"Forwarded": "for=203.0.113.9"}, {"Forwarded": "for=127.0.0.1"}]
    for h in cases:
        r = _stats(base, **h)
        body = _json_error(r, 403)
        assert "STATS_TOKEN" in body["error"] and "questions" not in r.text, h
        _assert_secure_headers(r)
    assert S.get(base + "/stats/dashboard", headers={"X-Forwarded-For": "203.0.113.9"}, timeout=10).status_code == 200


def test_stats_access_with_token():
    base = _branded()
    _json_error(_stats(base), 401)
    _json_error(_stats(base, **{"X-Stats-Token": "wrong"}), 401)
    _json_error(_stats(base, **{"X-Stats-Token": ""}), 401)
    _json_error(_stats(base, **{"X-Stats-Token": "s3cr3t-tök"}), 401)
    _json_error(_stats(base, **{"X-Stats-Token": "s3cr3t-token-extra"}), 401)
    assert _stats(base, **{"X-Stats-Token": "s3cr3t-token"}).status_code == 200
    r = _stats(base, **{"X-Stats-Token": "s3cr3t-token", "X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert S.get(base + "/stats/dashboard", timeout=10).status_code == 200


def test_branded_minutes_and_hostile_default():
    base = _branded()
    for _ in range(2):
        assert _chat(base, "verified").status_code == 200
    s = _stats(base, **{"X-Stats-Token": "s3cr3t-token"}).json()
    assert s["answered"] == 2 and s["estimated_minutes_saved"] == 10.0 and s["assumption"] == ASSUMPTION.format(5)
    assert _stats(_hostile()).json()["assumption"] == ASSUMPTION.format(3)


def test_security_headers_every_path():
    base = _plain()
    for r in [S.get(base + "/", timeout=10), S.get(base + "/config", timeout=10), _stats(base),
              S.get(base + "/stats/dashboard", timeout=10), _chat(base, "verified"), _chat(base, "modelerr"),
              S.post(base + "/chat", json={"task": ""}, timeout=10), S.get(base + "/nope", timeout=10),
              S.get(base + "/chat", timeout=10), _chat(base, "crash")]:
        _assert_secure_headers(r)
    assert _chat(base, "crash").status_code == 500


def test_embed_origins_in_frame_ancestors():
    r = S.get(_branded() + "/", timeout=10)
    _assert_secure_headers(r, frame="'self' https://broker.example.com https://app.partner.io:8443")
    csp = r.headers["content-security-policy"]
    assert "ok.com" not in csp and "script-src *" not in csp and "script-src 'self' 'unsafe-inline'" in csp
    assert csp.count("script-src") == 1
    _assert_secure_headers(S.get(_hostile() + "/", timeout=10))


def test_dashboard_served():
    r = S.get(_plain() + "/stats/dashboard", timeout=10)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert "<title>Usage</title>" in html and "X-Stats-Token" in html and "location.hash" in html
    assert "innerHTML" not in html and "location.search" not in html
    assert not re.search(r"""(?:src|href)\s*=\s*["']?https?://|url\(\s*["']?https?://|fetch\(\s*["']https?://""", html)


def test_theme_assets_served_strictly():
    base = _plain()
    for name, ctype in (("theme.css", "text/css"), ("theme.js", "text/javascript")):
        r = S.get(base + "/assets/" + name, timeout=10)
        assert r.status_code == 200 and r.headers["content-type"].startswith(ctype), name
        assert "content-security-policy" in r.headers and r.headers.get("cache-control") == "no-cache"
    for bad in ("/assets/index.html", "/assets/../server.py", "/assets/%2e%2e%2fserver.py", "/assets/stats.html"):
        assert S.get(base + bad, timeout=10).status_code == 404, bad
    page = S.get(base + "/", timeout=10).text
    assert '/assets/theme.css' in page and '/assets/theme.js' in page


def test_rate_limit_counted_and_stats_exempt():
    base = _plain()
    before = _stats(base).json()
    n422 = n429 = 0
    for _ in range(40):
        r = S.post(base + "/chat", json={"task": ""}, timeout=10)
        if r.status_code == 429:
            break
        assert r.status_code == 422
        n422 += 1
    for _ in range(3):
        r = S.post(base + "/chat", json={"task": "[verified] x"}, timeout=10)
        assert _json_error(r, 429) and r.headers["retry-after"] == "60"
        n429 += 1
    for _ in range(30):
        assert _stats(base).status_code == 200
    after = _stats(base).json()
    assert after["rejected_invalid"] - before["rejected_invalid"] == n422 > 0
    assert after["rate_limited"] - before["rate_limited"] == n429 + 1
    assert after["questions"] == before["questions"]


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    try:
        for name, fn in tests:
            try:
                fn()
                print(f"ok    {name}")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    finally:
        for proc, _ in _procs.values():
            proc.terminate()
        for proc, _ in _procs.values():
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        _serve(int(sys.argv[2]))
    else:
        main()
