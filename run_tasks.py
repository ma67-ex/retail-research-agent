"""5 scripted tasks with independently-computed expected answers.
Each check re-derives ground truth via a fresh direct tool call (not the
agent's own prior output), then verifies the agent's trace used the right
tool with the right args and got a matching result.
"""
import json
from pathlib import Path

from agent import run, TRACE_PATH
import tools

TOL = 0.05  # 5% tolerance on live numeric data pulled seconds apart


def _new_trace_entries(path: Path, start_len: int) -> list:
    lines = path.read_text().splitlines() if path.exists() else []
    return [json.loads(l) for l in lines[start_len:]]


def _tool_calls(entries, name=None):
    calls = [e for e in entries if e.get("type") == "tool_call"]
    return [c for c in calls if name is None or c["tool"] == name]


def _close(a, b, tol=TOL):
    if a is None or b is None:
        return a == b
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def check_pe(entries):
    calls = _tool_calls(entries, "get_quote")
    if not any(c["args"].get("ticker", "").upper() == "AAPL" for c in calls):
        return False, "no get_quote(AAPL) call"
    agent_val = next(c["result"]["trailing_pe"] for c in calls if c["args"]["ticker"].upper() == "AAPL")
    truth = tools.get_quote("AAPL")["trailing_pe"]
    return _close(agent_val, truth), f"agent={agent_val} truth={truth}"


def check_sma(entries):
    calls = _tool_calls(entries, "get_quote")
    if not any(c["args"].get("ticker", "").upper() == "MSFT" for c in calls):
        return False, "no get_quote(MSFT) call"
    agent_val = next(c["result"]["above_sma_200d"] for c in calls if c["args"]["ticker"].upper() == "MSFT")
    truth = tools.get_quote("MSFT")["above_sma_200d"]
    return agent_val == truth, f"agent={agent_val} truth={truth}"


def check_calculator(entries):
    calls = _tool_calls(entries, "calculator")
    if not calls:
        return False, "no calculator call"
    agent_val = calls[0]["result"].get("result")
    truth = 0.15 * 2_900_000_000_000
    return _close(agent_val, truth, tol=0.01), f"agent={agent_val} truth={truth}"


def check_filing(entries):
    calls = _tool_calls(entries, "get_latest_filing")
    if not any(c["args"].get("ticker", "").upper() == "TSLA" for c in calls):
        return False, "no get_latest_filing(TSLA) call"
    agent_val = next(c["result"].get("url") for c in calls if c["args"]["ticker"].upper() == "TSLA")
    truth = tools.get_latest_filing("TSLA")["url"]
    return agent_val == truth and agent_val is not None, f"agent={agent_val} truth={truth}"


def check_pe_comparison(entries):
    calls = _tool_calls(entries, "get_quote")
    tickers_seen = {c["args"]["ticker"].upper() for c in calls}
    if not {"NVDA", "AMD"} <= tickers_seen:
        return False, f"missing a get_quote call, saw {tickers_seen}"
    pe = {c["args"]["ticker"].upper(): c["result"]["trailing_pe"] for c in calls}
    agent_higher = "NVDA" if pe["NVDA"] >= pe["AMD"] else "AMD"
    truth_pe = {t: tools.get_quote(t)["trailing_pe"] for t in ("NVDA", "AMD")}
    truth_higher = "NVDA" if truth_pe["NVDA"] >= truth_pe["AMD"] else "AMD"
    return agent_higher == truth_higher, f"agent says {agent_higher} higher, truth is {truth_higher} ({pe} vs {truth_pe})"


TASKS = [
    ("What's AAPL's trailing P/E ratio?", check_pe),
    ("Is MSFT trading above its 200-day moving average?", check_sma),
    ("Calculate 15% of 2,900,000,000,000.", check_calculator),
    ("Find TSLA's most recent 10-K filing and give me the URL.", check_filing),
    ("Which has a higher trailing P/E right now: NVDA or AMD?", check_pe_comparison),
]


def main():
    passed = 0
    for i, (task, checker) in enumerate(TASKS, 1):
        start_len = len(TRACE_PATH.read_text().splitlines()) if TRACE_PATH.exists() else 0
        answer = run(task)
        entries = _new_trace_entries(TRACE_PATH, start_len)
        ok, detail = checker(entries)
        status = "PASS" if ok else "FAIL"
        passed += ok
        print(f"[{status}] task {i}: {task}\n  answer: {answer}\n  check: {detail}\n")
    print(f"{passed}/{len(TASKS)} passed")


if __name__ == "__main__":
    main()
