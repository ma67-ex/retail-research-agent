"""Tools the agent can call: market data, SEC filings and fundamentals, calculator."""
import ast
import datetime as dt
import functools
import hashlib
import inspect
import json
import math
import operator
import os
import re
import threading
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import requests
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"
YF_SOURCE = "Yahoo Finance via yfinance — prototype only, not licensed for commercial use"
SEC_SOURCE = "SEC EDGAR"
FACTS_SOURCE = "SEC EDGAR XBRL companyfacts"
DEFAULT_UA = "retail-research-agent/0.2 (set SEC_USER_AGENT to 'Name contact@domain')"
LIMIT = 1e100
REVENUE_CONCEPTS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")
PERIODS = ("1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y")
MAX_POINTS = 130
MAX_DOC = 20_000_000
PASSAGES, PASSAGE_CHARS = 6, 700


def _num(x):
    try:
        f = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return None
    return int(f) if f.is_integer() and abs(f) >= 1e6 else f


def _scaled(x):
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= size:
            return f"{x / size:,.2f}{suffix}"
    return f"{x:,.2f}"


def _money(x, cur="USD"):
    if x is None:
        return None
    return f"{'-' if x < 0 else ''}{'$' if cur == 'USD' else cur + ' '}{_scaled(abs(x))}"


def _calc_fmt(x):
    s = f"{x:,.2f}".removesuffix(".00") if abs(x) >= 1000 else f"{x:.6g}"
    if abs(x) >= 1e18:
        return f"{s} ({x:.3e})"
    for size, word in ((1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(x) >= size:
            return f"{s} ({x / size:,.2f} {word})"
    return s


def _norm_ticker(t):
    t = str(t or "").strip().lstrip("$").strip().upper()
    return re.sub(r"\.([A-Z])$", r"-\1", t)  # share class BRK.B -> BRK-B; keeps exchange suffixes like .NS


def _err(e):
    if isinstance(e, requests.Timeout):
        return "request timed out"
    if isinstance(e, requests.ConnectionError):
        return "network unavailable"
    if isinstance(e, requests.HTTPError):
        return f"HTTP {getattr(e.response, 'status_code', '?')} from data provider"
    return (str(e) or type(e).__name__)[:150]


def _read_cache(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_cache(path, body):
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(body)
        os.replace(tmp, path)
    except OSError:
        pass


def _tool(source):
    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError as e:
                return {"error": f"bad arguments: {e}", "source": source}
            bound.apply_defaults()
            norm = {k: _norm_ticker(v) if k == "ticker" else ("" if v is None else str(v).strip().upper())
                    for k, v in bound.arguments.items()}
            if "ticker" in norm and not TICKER_RE.match(norm["ticker"]):
                msg = "ticker is required" if not norm["ticker"] else f"invalid ticker {norm['ticker'][:20]!r}"
                return {"error": msg, "source": source}
            path = CACHE_DIR / (hashlib.sha1(json.dumps([fn.__name__, norm]).encode()).hexdigest() + ".json")
            offline = os.environ.get("OFFLINE") == "1"
            if not offline:
                try:
                    res = fn(**norm)
                    res.setdefault("source", source)
                    body = json.dumps(res, allow_nan=False)
                except Exception as e:
                    res = {"error": _err(e), "source": source}
                if "error" not in res:
                    _write_cache(path, body)
                    return res
            cached = _read_cache(path)
            if cached:
                return {**cached, "stale": True}
            return {"error": "OFFLINE=1 and no cached result", "source": source} if offline else res
        return wrapper
    return deco


_sec_lock = threading.Lock()
_sec_last = 0.0


def _sec_fetch(url, **kw):
    global _sec_last
    with _sec_lock:
        time.sleep(max(0.0, _sec_last + 0.11 - time.monotonic()))  # SEC allows 10 req/s
        _sec_last = time.monotonic()
    r = requests.get(url, headers={"User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_UA)}, timeout=15, **kw)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r


def _sec_get(url):
    r = _sec_fetch(url)
    return None if r is None else r.json()


@functools.lru_cache(maxsize=1)
def _cik_map():
    rows = _sec_get("https://www.sec.gov/files/company_tickers.json")
    return {_norm_ticker(r["ticker"]): str(r["cik_str"]).zfill(10) for r in rows.values()}


def _not_in_sec(ticker):
    return {"error": f"{ticker} not found in SEC's company list (funds/ETFs, foreign or delisted tickers may be absent)"}


@_tool(YF_SOURCE)
def get_quote(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    hist = t.history(period="2y", auto_adjust=False)  # 2y covers the dividend window; t.dividends would cost a second download
    close = hist["Close"].dropna()
    price = _num(close.iloc[-1]) if len(close) else None
    if price is None:
        return {"error": f"no price history for {ticker} (unknown ticker or data source unavailable)"}
    info = t.info
    name = info.get("shortName") or info.get("longName")
    if not name and not info.get("marketCap"):
        raise RuntimeError("quote metadata unavailable")
    cur = t.history_metadata.get("currency") or info.get("currency") or "USD"
    sma = _num(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    pe = _num(info.get("trailingPE"))
    pe = round(pe, 2) if pe and pe > 0 else None
    cap = _num(info.get("marketCap"))
    divs = hist["Dividends"]
    dy = float(divs[divs.index > close.index[-1] - dt.timedelta(days=365)].sum()) / price * 100
    out = {
        "ticker": ticker, "name": name, "currency": cur,
        "price": round(price, 2), "price_fmt": _money(price, cur),
        "sma_200d": round(sma, 2) if sma is not None else None,
        "sma_200d_fmt": _money(sma, cur),
        "above_sma_200d": price > sma if sma is not None else None,
        "trailing_pe": pe,
        "market_cap": cap, "market_cap_fmt": _money(cap, cur),
        "dividend_yield_pct": round(dy, 2), "dividend_yield_fmt": f"{dy:.2f}%",
        "as_of": close.index[-1].date().isoformat(),
    }
    if pe is None:
        out["pe_note"] = "no trailing P/E (negative or missing earnings)"
    return out


@_tool(YF_SOURCE)
def get_price_history(ticker: str, date: str) -> dict:
    try:
        d = dt.date.fromisoformat(str(date)[:10])
    except ValueError:
        return {"error": "date must be YYYY-MM-DD"}
    if d > dt.date.today():
        return {"error": f"{d} is in the future"}
    t = yf.Ticker(ticker)
    close = t.history(start=d - dt.timedelta(days=10), end=d + dt.timedelta(days=1), auto_adjust=False)["Close"].dropna()
    close = close[close.index.date <= d]
    price = _num(close.iloc[-1]) if len(close) else None
    if price is None:
        return {"error": f"no trading data for {ticker} on or before {d}"}
    cur = t.history_metadata.get("currency") or "USD"
    used = close.index[-1].date().isoformat()
    return {
        "ticker": ticker, "requested_date": d.isoformat(), "date_used": used,
        "close": round(price, 2), "close_fmt": _money(price, cur), "currency": cur,
        "note": "split-adjusted close, not dividend-adjusted",
        "as_of": used,
    }


def _annual(facts, concept, unit):
    best = None
    for r in facts.get("us-gaap", {}).get(concept, {}).get("units", {}).get(unit, []):
        if r.get("form") not in ("10-K", "10-K/A") or r.get("fp") != "FY" or not r.get("start"):
            continue
        # 10-Ks also carry quarterly rows tagged FY; only ~1-year spans are annual figures
        if not 350 <= (dt.date.fromisoformat(r["end"]) - dt.date.fromisoformat(r["start"])).days <= 380:
            continue
        if best is None or (r["end"], r.get("filed", "")) > (best["end"], best.get("filed", "")):
            best = r
    return best


def _pick(facts, concepts, unit):
    found = [(r, c) for c in concepts if (r := _annual(facts, c, unit))]
    # newest year wins, then larger value (total revenue beats contract-only revenue)
    return max(found, key=lambda x: (x[0]["end"], x[0]["val"]), default=(None, None))


@_tool(FACTS_SOURCE)
def get_fundamentals(ticker: str) -> dict:
    cik = _cik_map().get(ticker)
    if cik is None:
        return _not_in_sec(ticker)
    data = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    if data is None:
        return {"error": f"no SEC XBRL financial data for {ticker} (funds/ETFs and some foreign issuers don't have it)"}
    facts = data.get("facts", {})
    out = {"ticker": ticker, "company": data.get("entityName")}
    missing, ends = [], []
    for key, concepts, unit in (
        ("revenue", REVENUE_CONCEPTS, "USD"),
        ("net_income", ("NetIncomeLoss",), "USD"),
        ("eps_diluted", ("EarningsPerShareDiluted",), "USD/shares"),
        ("dividends_per_share_declared", ("CommonStockDividendsPerShareDeclared",), "USD/shares"),
    ):
        row, concept = _pick(facts, concepts, unit)
        if row is None and key.startswith("dividends"):
            continue
        val = _num(row["val"]) if row else None
        out[key], out[f"{key}_fmt"], out[f"{key}_fy_end"] = val, _money(val), row["end"] if row else None
        if key == "revenue":
            out["revenue_concept"] = concept
        if row:
            ends.append(row["end"])
        else:
            missing.append(key)
    if len(missing) == 3:
        return {"error": f"no annual 10-K XBRL data for {ticker} (funds/ETFs and some foreign issuers don't have it)"}
    rows = facts.get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    latest = max(rows, key=lambda r: (r["end"], r.get("filed", "")), default=None)
    newest = max(ends)
    # multi-class issuers (GOOGL, BRK) stop reporting a usable cover-page total
    fresh = latest and latest["end"] >= str(dt.date.fromisoformat(newest) - dt.timedelta(days=400))
    shares = _num(latest["val"]) if fresh else None
    out["shares_outstanding"] = shares
    out["shares_outstanding_fmt"] = f"{_scaled(shares)} shares" if shares is not None else None
    out["shares_outstanding_date"] = latest["end"] if shares is not None else None
    if shares is None:
        missing.append("shares_outstanding")
    if missing:
        out["missing"] = missing
    out["as_of"] = newest
    return out


def _filing_pages(sub):
    yield sub["filings"]["recent"]
    for f in sorted(sub["filings"].get("files", []), key=lambda f: f.get("filingTo", ""), reverse=True):
        page = _sec_get(f"https://data.sec.gov/submissions/{f['name']}")  # heavy filers push older forms out of "recent"
        if page:
            yield page


@_tool(SEC_SOURCE)
def get_latest_filing(ticker: str, form_type: str = "10-K") -> dict:
    cik = _cik_map().get(ticker)
    if cik is None:
        return _not_in_sec(ticker)
    sub = _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if sub is None:
        return _not_in_sec(ticker)
    form_type = form_type or "10-K"
    seen = set()
    for page in _filing_pages(sub):
        seen.update(page["form"])
        for i, form in enumerate(page["form"]):
            if form == form_type:
                accession = page["accessionNumber"][i].replace("-", "")
                date = page["filingDate"][i]
                return {
                    "ticker": ticker, "company": sub.get("name"), "form_type": form_type,
                    "filing_date": date, "period_of_report": page["reportDate"][i] or None,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{page['primaryDocument'][i]}",
                    "as_of": date,
                }
    forms = ", ".join(list(dict.fromkeys(sub["filings"]["recent"]["form"]))[:6])
    fund = "10-K" not in seen and sub.get("entityType") != "operating"
    hint = "; funds/ETFs generally don't file 10-Ks (try form_type N-CSR)" if fund else ""
    return {"error": f"no {form_type} found for {ticker}; recent forms: {forms}{hint}"}


SECTIONS = {
    "10-K": {
        "risk_factors": ("1A", r"risk\s+factors", "Item 1A. Risk Factors"),
        "mdna": ("7", r"management\W{0,3}s?\s*discussion", "Item 7. Management's Discussion and Analysis"),
        "business": ("1", r"business\b", "Item 1. Business"),
        "legal_proceedings": ("3", r"legal\s+proceedings", "Item 3. Legal Proceedings"),
    },
    "10-Q": {
        "risk_factors": ("1A", r"risk\s+factors", "Part II, Item 1A. Risk Factors"),
        "mdna": ("2", r"management\W{0,3}s?\s*discussion", "Part I, Item 2. Management's Discussion and Analysis"),
    },
}
_VOID = {"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr"}
_SKIP = {"script", "style", "title", "ix:header"}
_BLOCK = {"p", "div", "tr", "li", "br", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
_ITEM = re.compile(r"(?:part\s+[iv]+\b\W*)?item\s*(\d{1,2}[a-c]?)(?![a-z])\W*(.*)", re.I | re.S)
_BARE_ITEM = re.compile(r"(?:part\s+[iv]+\b\W*)?item\s*\d{1,2}[a-c]?\W*", re.I)
_NOISE = re.compile(r"[-–—\s]*(?:page\s+)?\d{1,3}[-–—\s]*|table of contents|index|part\s+[iv]+|[•·▪●◦■*]|.{0,60}\|\s*(?:\d{4}\s+)?form 10-[kq]\s*\|\s*\d+", re.I)
_TERMINAL = re.compile(r"[.!?:;][\"')”’\]]*$")
_SENTENCE_END = re.compile(r"[.!?][\"')”’]*(?=\s)")
_STOP = set("a an the and or of to in on for with by at from as is are was were be been it its this that these those "
            "what which who how why does do did about say says said company any".split())
_ALIASES = {"ai": ["artificial", "intelligence"], "artificial": ["ai"]}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines, self.cur, self.skip, self.depth = [], [], None, 0

    def flush(self):
        line = re.sub(r"\s+", " ", "".join(self.cur).replace("​", "").replace("﻿", "")).strip()
        self.cur = []
        if line:
            self.lines.append(line)

    def handle_starttag(self, tag, attrs):
        if self.skip:
            self.depth += tag == self.skip
        elif tag in _SKIP or (tag not in _VOID and _HIDDEN.search(dict(attrs).get("style") or "")):
            self.skip, self.depth = tag, 1
        elif tag in _BLOCK:
            self.flush()
        elif tag in ("td", "th"):
            self.cur.append(" ")

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip:
                self.depth -= 1
                if not self.depth:
                    self.skip = None
        elif tag in _BLOCK:
            self.flush()
        elif tag in ("td", "th"):
            self.cur.append(" ")

    def handle_data(self, data):
        if not self.skip:
            self.cur.append(data)


def _paragraphs(html):
    p = _Text()
    p.feed(html)
    p.close()
    p.flush()
    out = []
    for line in p.lines:
        if _NOISE.fullmatch(line):
            continue
        if out and _BARE_ITEM.fullmatch(out[-1]) and len(line) <= 120 and not _ITEM.match(line):
            out[-1] += " " + line  # filers often put "Item 1A." and its title in separate cells
        elif out and line[:1].islower() and not _TERMINAL.search(out[-1]) and not _heading(out[-1]):
            out[-1] += " " + line  # sentence split by a page break
        else:
            out.append(line)
    return out


def _heading(p):
    m = _ITEM.match(p)
    if m and len(p) <= 250 and not m[2][:1].islower():  # lowercase title = in-text cross-reference
        return m[1].upper(), m[2]


def _find_section(paras, item, title_re):
    heads = [(i, *h) for i, p in enumerate(paras) if (h := _heading(p))]
    best = None
    for k, (i, iid, title) in enumerate(heads):
        if iid == item and re.match(title_re, title, re.I):
            # the same item repeats as a running page header; only a different item ends the section
            end = next((j for j, jid, _ in heads[k + 1:] if jid != item), len(paras))
            body = paras[i + 1:end]
            if best is None or sum(map(len, body)) > sum(map(len, best)):
                best = body
    return best


def _stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _terms(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def _rank(paras, terms):
    docs = [Counter(_terms(p)) for p in paras]
    lens = [sum(d.values()) for d in docs]
    avg = sum(lens) / len(lens) or 1
    idf = {t: math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
           for t in set(terms) if (df := sum(t in d for d in docs))}
    score = [sum(w * d[t] * 2.5 / (d[t] + 1.5 * (0.25 + 0.75 * n / avg)) for t, w in idf.items())
             for d, n in zip(docs, lens)]
    return sorted((i for i, s in enumerate(score) if s > 0), key=lambda i: (-score[i], i))[:PASSAGES]


def _is_title(p):
    return 40 <= len(p) <= 300 and not re.search(r"[.!?][\"')”’]*\s+[A-Z]", p)


def _clip(text, cap=PASSAGE_CHARS):
    if len(text) <= cap:
        return text
    cut = text[:cap - 1]
    ends = [m.end() for m in _SENTENCE_END.finditer(cut)]
    cut = cut[:ends[-1]] if ends and ends[-1] >= cap // 2 else cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


def _filing_html(url):
    path = CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    try:
        return path.read_text()
    except (OSError, ValueError):
        pass
    r = _sec_fetch(url, stream=True)
    if r is None:
        raise RuntimeError("filing document not found (HTTP 404)")
    buf = bytearray()
    try:
        for chunk in r.iter_content(1 << 16):
            buf += chunk
            if len(buf) > MAX_DOC:
                raise ValueError(f"filing document is larger than {MAX_DOC // 1_000_000} MB")
    finally:
        r.close()
    try:
        html = buf.decode("utf-8")
    except UnicodeDecodeError:
        html = buf.decode("cp1252", "replace")
    _write_cache(path, html)
    return html


@_tool(SEC_SOURCE)
def read_filing(ticker: str, section: str = "risk_factors", query: str = None, form_type: str = "10-K") -> dict:
    form_type, section, query = form_type or "10-K", section.lower() or "risk_factors", query.lower()
    sections = SECTIONS.get(form_type)
    if sections is None:
        return {"error": f"form_type must be one of: {', '.join(SECTIONS)}"}
    if section not in sections:
        return {"error": f"section must be one of: {', '.join(sections)} (for {form_type})"}
    filing = get_latest_filing(ticker, form_type)
    if "error" in filing:
        return filing
    item, title_re, title = sections[section]
    body = _find_section(_paragraphs(_filing_html(filing["url"])), item, title_re)
    if not body:
        return {"error": f"could not locate {title} in {ticker}'s {form_type} filed {filing['filing_date']} "
                         f"(unusual layout, or the section is incorporated by reference); open {filing['url']}"}
    terms = _terms(query)
    terms += [a for t in terms for a in _ALIASES.get(t, [])]
    if terms:
        picked = _rank(body, terms)
    else:
        picked = [i for i, p in enumerate(body) if _is_title(p)][:PASSAGES] or list(range(min(PASSAGES, len(body))))
    out = {
        "ticker": ticker, "company": filing["company"], "form_type": form_type,
        "filing_date": filing["filing_date"], "period_of_report": filing["period_of_report"], "url": filing["url"],
        "section": section, "section_title": title, "query": query or None,
        "passages": [{"id": f"p{i + 1}", "text": _clip(body[i])} for i in picked],
        "passage_count": len(body), "as_of": filing["filing_date"],
    }
    if terms and not picked:
        out["note"] = "no paragraph in this section matched the query terms"
    return out


@_tool(YF_SOURCE)
def get_price_series(ticker: str, period: str = "1y") -> dict:
    period = period.lower() or "1y"
    if period not in PERIODS:
        return {"error": f"period must be one of: {', '.join(PERIODS)}"}
    t = yf.Ticker(ticker)
    close = t.history(period=period, auto_adjust=False)["Close"].dropna()
    if not len(close):
        return {"error": f"no price history for {ticker} (unknown ticker or data source unavailable)"}
    cur = t.history_metadata.get("currency") or "USD"
    try:
        info = t.info
        name = info.get("shortName") or info.get("longName")
    except Exception:  # metadata is a second Yahoo call; the series is still useful without a name
        name = None
    n = len(close)
    k = min(n, MAX_POINTS)
    keep = [round(i * (n - 1) / max(k - 1, 1)) for i in range(k)]
    day = lambda i: close.index[i].date().isoformat()
    start, end, hi, lo = (round(float(v), 2) for v in (close.iloc[0], close.iloc[-1], close.max(), close.min()))
    change = round((end / start - 1) * 100, 2) + 0.0  # + 0.0 turns -0.0 into 0.0
    return {
        "ticker": ticker, "name": name, "currency": cur, "period": period,
        "points": [[day(i), round(float(close.iloc[i]), 2)] for i in keep],
        "start_date": day(0), "end_date": day(n - 1),
        "start_close": start, "start_close_fmt": _money(start, cur),
        "end_close": end, "end_close_fmt": _money(end, cur),
        "change_pct": change, "change_fmt": f"{change:+.2f}%",
        "high": hi, "high_fmt": _money(hi, cur), "high_date": day(int(close.values.argmax())),
        "low": lo, "low_fmt": _money(lo, cur), "low_date": day(int(close.values.argmin())),
        "as_of": day(n - 1),
    }


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _check(v):
    if isinstance(v, complex) or v != v or abs(v) > LIMIT:
        raise ValueError("result out of range or not a real number")
    return v


def _eval_node(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):  # type() excludes bool
        return _check(float(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        a, b = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 1000:
            raise ValueError("exponent too large (max 1000)")
        return _check(_BINARY[type(node.op)](a, b))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check(_UNARY[type(node.op)](_eval_node(node.operand)))
    raise ValueError("unsupported expression: only numbers, + - * / ** and parentheses")


def calculator(expression: str = None, **_) -> dict:  # **_ so a wrongly named argument returns an error instead of raising
    expr = str(expression)
    out = {"expression": expr, "source": "calculator", "as_of": None}
    if expression is None:
        return {**out, "error": "missing required argument: expression"}
    try:
        if len(expr) > 300:
            raise ValueError("expression too long")
        s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)
        s = re.sub(r"(\d*\.?\d+)\s*%\s*of\s+", r"(\1/100) * ", s, flags=re.I)
        s = re.sub(r"(\d*\.?\d+)\s*%", r"(\1/100)", s)
        result = _eval_node(ast.parse(s.strip(), mode="eval").body)
        return {**out, "result": result, "result_fmt": _calc_fmt(result)}
    except SyntaxError:
        return {**out, "error": "invalid expression syntax"}
    except OverflowError:
        return {**out, "error": "result out of range"}
    except Exception as e:
        return {**out, "error": str(e)[:150] or type(e).__name__}


def _schema(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


_TICKER = {"type": "string", "description": "Stock or ETF ticker symbol, e.g. AAPL or BRK.B"}

TOOL_SCHEMAS = [
    _schema(
        "get_quote",
        "Current market snapshot for a ticker: latest price, 200-day moving average (SMA) and whether price is "
        "above it, trailing P/E, market cap, trailing-12-month dividend yield, company name. Use for any "
        "question about the price or valuation NOW. Not for past dates and not for revenue or earnings.",
        {"ticker": _TICKER}, ["ticker"],
    ),
    _schema(
        "get_price_history",
        "Closing price of a ticker on a specific past date (or the most recent trading day before it). "
        "Use whenever the question names a date.",
        {"ticker": _TICKER, "date": {"type": "string", "description": "Date as YYYY-MM-DD, e.g. 2020-01-02"}},
        ["ticker", "date"],
    ),
    _schema(
        "get_fundamentals",
        "Latest annual financials from SEC filings: revenue, net income, diluted EPS, shares outstanding, "
        "dividends per share declared. Use for revenue, profit, earnings, EPS or share-count questions. "
        "Not for prices or P/E.",
        {"ticker": _TICKER}, ["ticker"],
    ),
    _schema(
        "get_latest_filing",
        "Date and URL of a company's most recent SEC filing of a given form type (10-K annual report, "
        "10-Q quarterly report, 8-K). Use when asked for a filing, report or its link. Funds and ETFs "
        "do not file 10-Ks.",
        {"ticker": _TICKER,
         "form_type": {"type": "string", "description": "e.g. 10-K, 10-Q, 8-K, N-CSR", "default": "10-K"}},
        ["ticker"],
    ),
    _schema(
        "read_filing",
        "Read the actual text of a company's latest SEC 10-K (annual) or 10-Q (quarterly) and return numbered "
        "passages (ids like p12) from one section, to quote and cite. Use for 'what does the company say about "
        "X', its risk factors, management's discussion (MD&A), business description or legal proceedings. "
        "Put the topic's key words in query to get the best-matching passages; omit query for an overview. "
        "Not for revenue/EPS numbers (use get_fundamentals) and not for just the filing link (use get_latest_filing).",
        {"ticker": _TICKER,
         "section": {"type": "string", "enum": list(SECTIONS["10-K"]), "default": "risk_factors",
                     "description": "risk_factors, mdna (management's discussion), business or legal_proceedings; "
                                    "10-Q supports only risk_factors and mdna"},
         "query": {"type": "string", "description": "Topic key words, e.g. 'supply chain China' or 'artificial intelligence'"},
         "form_type": {"type": "string", "enum": list(SECTIONS), "default": "10-K"}},
        ["ticker"],
    ),
    _schema(
        "get_price_series",
        "Daily closing prices of a ticker over a recent period, for charts and performance questions: start and "
        "end close, percent change, period high and low with dates, plus a downsampled list of [date, close] "
        "points. Use for 'how has X done over the past year', 'is X up this year', trends or a chart. For one "
        "specific past date use get_price_history; for the price now use get_quote.",
        {"ticker": _TICKER,
         "period": {"type": "string", "enum": list(PERIODS), "default": "1y",
                    "description": "Look-back window; ytd = since the start of this calendar year"}},
        ["ticker"],
    ),
    _schema(
        "calculator",
        "Evaluate arithmetic (+ - * / ** and parentheses; commas and % allowed) on numbers already obtained "
        "from other tools or given by the user, e.g. '0.15 * 2,900,000,000,000'. Never use it to look up "
        "data. When stating large results, copy the result_fmt text.",
        {"expression": {"type": "string"}}, ["expression"],
    ),
]

TOOL_FUNCTIONS = {
    "get_quote": get_quote,
    "get_price_history": get_price_history,
    "get_fundamentals": get_fundamentals,
    "get_latest_filing": get_latest_filing,
    "read_filing": read_filing,
    "get_price_series": get_price_series,
    "calculator": calculator,
}
