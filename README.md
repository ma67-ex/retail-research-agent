# Retail Research Agent

A tool-calling LLM agent that answers retail-investor research questions
(price, moving averages, P/E, SEC filings) using real data — for people
priced out of a Bloomberg terminal.

## Why

Professional market-data terminals cost tens of thousands of dollars a year.
Retail investors are left with scattered free sites and no single assistant
that can look things up and reason about them. This agent runs entirely
local and free: no paid API, no card, no account.

## Stack

- **Model:** `llama3.2` via [Ollama](https://ollama.com), local, free
- **Tools:** `yfinance` (price/SMA/P/E/market cap), SEC EDGAR (filings), a
  restricted arithmetic calculator
- **Trace log:** every tool call + final answer logged to `trace.jsonl`

## Layout

```
tools.py       market-data, SEC-filing, and calculator tools + their schemas
agent.py       chat loop: sends tools to Ollama, executes tool calls, logs trace
run_tasks.py   5 scripted tasks, each checked against an independently
               recomputed ground truth (not the agent's own prior output)
```

## Run it

```bash
ollama serve &              # if not already running as a service
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python agent.py "What's AAPL's trailing P/E ratio?"
./.venv/bin/python run_tasks.py     # runs all 5 scripted checks
```

## Status

5/5 scripted tasks passing. Known limitation: the model's restated prose can
occasionally misstate a number in plain text even when the tool call itself
returned the correct value (seen once on the percentage-calculation task) —
the tool result is correct and checked directly; the prose isn't.

## Roadmap

- [ ] Retrieval tool over a small local corpus (10-Ks, research notes)
- [ ] Eval harness: cost + latency logged per task, separate from the trace
- [ ] Simple web UI (watchlist + chat) for a demo video
- [ ] Pitch deck + business case (Dubai AI Agents Championship submission)

## Competition compliance (Dubai AI Agents Championship — Best Business AI Agent)

Checked against the rules at dubaichamberdigital.com/en/create-ai-agents:

- [x] Original agent, not copied from an existing repository — written from scratch
- [x] Idea/prototype/early-MVP stage, not a fully commercialized product
- [x] Protects personal/confidential/organizational data — no personal data
      collected; only public market data and public SEC filings are touched.
      **Revisit once a UI with user accounts is added** — that's where real
      data-protection obligations start.
- [x] Submission materials in English
- [ ] Registration — done by the founder individually, not something this repo covers
- [ ] Submission package: prototype (in progress) + pitch deck (not started) +
      demo video (not started) — due Jan 2027
- [ ] Grand Final needs a fully functional agent live on stage (not just an
      MVP) — current prototype already makes live tool calls, not canned
      responses, which is a head start on that bar
