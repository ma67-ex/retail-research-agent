import contextlib
import datetime as dt
import html as htmllib
import json
import os
import re
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

import tools


@contextlib.contextmanager
def patched(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@contextlib.contextmanager
def temp_cache():
    with tempfile.TemporaryDirectory() as d, patched(tools, CACHE_DIR=Path(d) / "cache"):
        yield tools.CACHE_DIR


def is_json_safe(x):
    json.dumps(x, allow_nan=False)
    return True


def test_calculator_math_and_formats():
    r = tools.calculator("0.15 * 2,900,000,000,000")
    assert r["result"] == 435e9 and isinstance(r["result"], float), r
    assert r["result_fmt"] == "435,000,000,000 (435.00 billion)", r
    assert r["source"] == "calculator" and r["as_of"] is None and r["expression"] == "0.15 * 2,900,000,000,000"
    assert tools.calculator("15% * 2900000000000")["result"] == 435e9
    assert tools.calculator("15% of 2,900,000,000,000")["result"] == 435e9
    assert tools.calculator("+5 - -2")["result"] == 7.0
    assert tools.calculator("2 ** 10 / 4")["result"] == 256.0
    assert tools.calculator("1,000 + 2,500.5")["result"] == 3500.5


def test_calculator_rejects_unsafe():
    t0 = time.perf_counter()
    r = tools.calculator("9**9**9")
    assert time.perf_counter() - t0 < 0.5 and "error" in r, r
    for bad in ["10**5000", "2**1001", "9**14000000", "(-8)**(1/3)", "True+True", "True", "1e999", "1e100*10",
                "1/0", "1j", "__import__('os')", "abs(1)", "x+1", "(1).real", "[1]", "'a'*3", "10 % 3", "1+", "",
                "(" * 100 + "1" + ")" * 100 + "+" * 400]:
        r = tools.calculator(bad)
        assert "error" in r and "result" not in r, (bad, r)
        assert r["source"] == "calculator" and is_json_safe(r)
    assert "error" in tools.calculator(None)
    assert tools.calculator(5)["result"] == 5.0
    for r in (tools.calculator(), tools.calculator(expr="1+1")):
        assert "expression" in r["error"] and r["source"] == "calculator", r


def test_ticker_normalization_and_validation():
    assert tools._norm_ticker(" $brk.b ") == "BRK-B"
    assert tools._norm_ticker("aapl") == "AAPL"
    assert tools._norm_ticker("RELIANCE.NS") == "RELIANCE.NS"
    assert tools._norm_ticker(None) == ""
    for fn in (tools.get_quote, tools.get_fundamentals, tools.get_latest_filing):
        for bad in ("", "   ", None, "A B", "../etc"):
            r = fn(bad)
            assert "error" in r and "source" in r, (fn.__name__, bad, r)
    assert "error" in tools.get_quote(wrong_arg="AAPL")


def test_formatters():
    assert tools._money(4.95e12) == "$4.95T"
    assert tools._money(339.5) == "$339.50"
    assert tools._money(416.16e9) == "$416.16B"
    assert tools._money(-1.2e9) == "-$1.20B"
    assert tools._money(1234.5, "INR") == "INR 1,234.50"
    assert tools._money(None) is None
    assert tools._calc_fmt(4.95e12) == "4,950,000,000,000 (4.95 trillion)"
    assert tools._calc_fmt(1.5e6) == "1,500,000 (1.50 million)"
    assert tools._calc_fmt(0.15) == "0.15"
    assert tools._calc_fmt(-3.0) == "-3"
    assert tools._num(float("nan")) is None and tools._num("Infinity") is None and tools._num(None) is None
    assert tools._num(4.95e12) == 4950000000000 and isinstance(tools._num(4.95e12), int)


def _bdays(n):
    end = pd.Timestamp.today().normalize()
    while end.weekday() > 4:
        end -= pd.Timedelta(days=1)
    end -= pd.Timedelta(days=1)
    return pd.bdate_range(end=end, periods=n, tz="America/New_York")


N = 400


class FakeTicker:
    mode = "ok"
    pe = 30.0
    calls = 0

    def __init__(self, symbol):
        FakeTicker.calls += 1
        if FakeTicker.mode == "raise":
            raise ConnectionError("Too Many Requests")
        self.idx = _bdays(N)
        self.close = pd.Series([100 + i * 0.1 for i in range(N)], index=self.idx, name="Close")
        self.history_metadata = {"currency": "USD"}
        self.info = {"shortName": "Fake Corp", "trailingPE": FakeTicker.pe, "marketCap": 4.95e12}
        self.divs = pd.Series(0.0, index=self.idx)
        self.divs.iloc[[0, -130, -60]] = 0.25  # first one is >365 days old

    def history(self, period=None, start=None, end=None, auto_adjust=True):
        assert auto_adjust is False
        if FakeTicker.mode == "empty":
            return pd.DataFrame({"Close": [], "Dividends": []})
        c = self.close
        if start is not None:
            c = c[(c.index.date >= start) & (c.index.date < end)]
        return pd.DataFrame({"Close": c, "Dividends": self.divs.reindex(c.index)})


@contextlib.contextmanager
def fake_yf(**attrs):
    for k, v in {"mode": "ok", "pe": 30.0, "calls": 0, **attrs}.items():
        setattr(FakeTicker, k, v)
    with patched(tools, yf=types.SimpleNamespace(Ticker=FakeTicker)):
        yield


def test_get_quote_fields():
    with temp_cache(), fake_yf():
        r = tools.get_quote("fake")
        close = [100 + i * 0.1 for i in range(N)]
        assert r["ticker"] == "FAKE" and r["name"] == "Fake Corp" and r["currency"] == "USD"
        assert r["price"] == round(close[-1], 2) and r["price_fmt"] == f"${close[-1]:,.2f}"
        assert r["sma_200d"] == round(sum(close[-200:]) / 200, 2) and r["above_sma_200d"] is True
        assert r["trailing_pe"] == 30.0 and "pe_note" not in r
        assert r["market_cap"] == 4950000000000 and r["market_cap_fmt"] == "$4.95T"
        assert r["dividend_yield_pct"] == round(0.5 / close[-1] * 100, 2), r
        assert r["dividend_yield_fmt"].endswith("%")
        assert r["as_of"] == _bdays(N)[-1].date().isoformat() and "Yahoo Finance" in r["source"]
        assert is_json_safe(r)
    with temp_cache(), fake_yf(pe="Infinity"):
        r = tools.get_quote("LOSS")
        assert r["trailing_pe"] is None and r["pe_note"] == "no trailing P/E (negative or missing earnings)", r
    with temp_cache(), fake_yf(mode="empty"):
        r = tools.get_quote("NOPE")
        assert "error" in r and "source" in r and not list(tools.CACHE_DIR.glob("*")), r


def test_get_price_history_uses_prior_trading_day():
    with temp_cache(), fake_yf():
        idx = _bdays(N)
        friday = next(d for d in reversed(idx) if d.weekday() == 4 and d.date() + dt.timedelta(days=2) < dt.date.today())
        sunday = friday.date() + dt.timedelta(days=2)
        r = tools.get_price_history("fake", sunday.isoformat())
        assert r["date_used"] == friday.date().isoformat() and r["requested_date"] == sunday.isoformat(), r
        assert r["close"] == round(100 + list(idx).index(friday) * 0.1, 2) and r["close_fmt"].startswith("$")
        assert r["as_of"] == r["date_used"] and is_json_safe(r)
        assert "YYYY-MM-DD" in tools.get_price_history("FAKE", "last tuesday")["error"]
        assert "future" in tools.get_price_history("FAKE", "2999-01-01")["error"]


def test_cache_fallback_and_offline():
    with temp_cache(), fake_yf():
        fresh = tools.get_quote("FAKE")
        assert "stale" not in fresh and len(list(tools.CACHE_DIR.glob("*.json"))) == 1
        FakeTicker.mode = "raise"
        stale = tools.get_quote("$fake")
        assert stale["stale"] is True and stale["as_of"] == fresh["as_of"] and stale["price"] == fresh["price"], stale
        err = tools.get_quote("OTHER")
        assert "error" in err and "stale" not in err and "Too Many Requests" in err["error"], err
        FakeTicker.mode, FakeTicker.calls = "ok", 0
        with env(OFFLINE="1"):
            off = tools.get_quote("FAKE")
            assert off["stale"] is True and off["price"] == fresh["price"] and FakeTicker.calls == 0
            miss = tools.get_quote("OTHER")
            assert "OFFLINE" in miss["error"] and FakeTicker.calls == 0
        with open(next(tools.CACHE_DIR.glob("*.json")), "w") as f:
            f.write("{corrupt")
        FakeTicker.mode = "raise"
        assert "error" in tools.get_quote("FAKE")


FACTS = {
    "entityName": "Fake Inc",
    "facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2011-02-17", "val": 941481, "form": "10-K", "filed": "2011-02-28"}]}}},
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 350, "form": "10-K", "fp": "FY", "filed": "2025-02-05"},
                {"start": "2025-01-01", "end": "2025-12-31", "val": 247, "form": "10-K", "fp": "FY", "filed": "2026-02-05"},
            ]}},
            "Revenues": {"units": {"USD": [
                {"start": "2025-01-01", "end": "2025-12-31", "val": 371, "form": "10-K", "fp": "FY", "filed": "2026-02-05"},
                {"start": "2025-10-01", "end": "2025-12-31", "val": 999, "form": "10-K", "fp": "FY", "filed": "2026-02-05"},
            ]}},
            "NetIncomeLoss": {"units": {"USD": [
                {"start": "2025-01-01", "end": "2025-12-31", "val": -5e9, "form": "10-K", "fp": "FY", "filed": "2026-02-05"},
                {"start": "2026-01-01", "end": "2026-12-31", "val": 1, "form": "DEF 14A", "fp": None, "filed": "2026-04-24"},
            ]}},
        },
    },
}


def fake_sec(routes, seen):
    def get(url):
        seen.append(url)
        for part, payload in routes.items():
            if part in url:
                return payload
        raise AssertionError(f"unexpected url {url}")
    return get


def test_get_fundamentals_selection_and_shape():
    tickers = {"0": {"ticker": "FAKE", "cik_str": 1}, "1": {"ticker": "BRK-B", "cik_str": 2}}
    seen = []
    routes = {"company_tickers": tickers, "CIK0000000001": FACTS, "CIK0000000002": None}
    tools._cik_map.cache_clear()
    with temp_cache(), patched(tools, _sec_get=fake_sec(routes, seen)):
        r = tools.get_fundamentals("fake")
        assert r["revenue"] == 371 and r["revenue_concept"] == "Revenues" and r["revenue_fy_end"] == "2025-12-31", r
        assert r["net_income"] == -5e9 and r["net_income_fmt"] == "-$5.00B", r
        assert r["eps_diluted"] is None and r["shares_outstanding"] is None, r
        assert set(r["missing"]) == {"eps_diluted", "shares_outstanding"} and "dividends_per_share_declared" not in r
        assert r["as_of"] == "2025-12-31" and r["source"] == "SEC EDGAR XBRL companyfacts" and is_json_safe(r)
        fund = tools.get_fundamentals("BRK.B")
        assert "error" in fund and "ETFs" in fund["error"], fund
        tools.get_fundamentals("NOSUCH")
        assert sum("company_tickers" in u for u in seen) == 1, seen
    tools._cik_map.cache_clear()


def test_get_latest_filing_and_sec_plumbing():
    tickers = {"0": {"ticker": "FAKE", "cik_str": 1}, "1": {"ticker": "FUND", "cik_str": 2}}
    fund = {"name": "Fake Fund", "filings": {"recent": {"form": ["N-CSR", "N-PORT"]}}}
    sub = {"name": "Fake Inc", "filings": {"recent": {
        "form": ["8-K", "10-K"], "accessionNumber": ["0-1", "0000000001-26-000001"],
        "primaryDocument": ["a.htm", "fake-10k.htm"], "filingDate": ["2026-05-01", "2026-02-01"],
        "reportDate": ["", "2025-12-31"]}}}
    seen = []
    tools._cik_map.cache_clear()
    with temp_cache(), patched(tools, _sec_get=fake_sec({"company_tickers": tickers, "CIK0000000001": sub, "CIK0000000002": fund}, seen)):
        r = tools.get_latest_filing("fake")
        assert r["url"] == "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/fake-10k.htm", r
        assert r["filing_date"] == "2026-02-01" and r["as_of"] == "2026-02-01" and r["period_of_report"] == "2025-12-31"
        miss = tools.get_latest_filing("FAKE", "10Q")
        assert "recent forms: 8-K, 10-K" in miss["error"] and "ETFs" not in miss["error"], miss
        miss = tools.get_latest_filing("FUND")
        assert "recent forms: N-CSR, N-PORT" in miss["error"] and "N-CSR" in miss["error"] and "ETFs" in miss["error"], miss
        assert not any("browse-edgar" in u for u in seen)
    tools._cik_map.cache_clear()

    calls = []

    def fake_get(url, headers, timeout):
        calls.append(headers["User-Agent"])
        return types.SimpleNamespace(status_code=404, json=lambda: {}, raise_for_status=lambda: None)

    with patched(tools.requests, get=fake_get):
        with env(SEC_USER_AGENT="tester/1.0 qa"):
            assert tools._sec_get("https://data.sec.gov/x") is None
        with env(SEC_USER_AGENT=None):
            tools._sec_get("https://data.sec.gov/x")
    assert calls[0] == "tester/1.0 qa" and calls[1] == tools.DEFAULT_UA and "@" not in calls[0]


def test_get_latest_filing_scans_older_submission_pages():
    def page(forms, doc, year):
        n = len(forms)
        return {"form": forms, "accessionNumber": [f"0000000005-{year - i}-00000{i}" for i in range(n)],
                "primaryDocument": [f"{doc}{i}.htm" for i in range(n)],
                "filingDate": [f"{year - i}-02-01" for i in range(n)], "reportDate": [f"{year - i - 1}-12-31" for i in range(n)]}

    tickers = {"0": {"ticker": "HEAVY", "cik_str": 5}, "1": {"ticker": "OPCO", "cik_str": 6}, "2": {"ticker": "FUND", "cik_str": 7}}
    heavy = {"name": "Heavy Inc", "entityType": "operating", "filings": {
        "recent": {"form": ["424B2", "8-K"]},
        "files": [{"name": "CIK0000000005-submissions-002.json", "filingTo": "2015-01-01"},
                  {"name": "CIK0000000005-submissions-001.json", "filingTo": "2024-06-30"}]}}
    opco = {"name": "Op Co", "entityType": "operating", "filings": {
        "recent": {"form": ["8-K"]}, "files": [{"name": "CIK0000000006-submissions-001.json", "filingTo": "2024-01-01"}]}}
    fund = {"name": "Fake Fund", "entityType": "other", "filings": {"recent": {"form": ["N-CSR"]}}}
    routes = {"company_tickers": tickers, "CIK0000000005.json": heavy, "CIK0000000006.json": opco, "CIK0000000007.json": fund,
              "5-submissions-001": page(["8-K", "10-K", "10-K"], "new", 2025), "5-submissions-002": page(["10-K", "10-Q"], "old", 2012),
              "6-submissions-001": None}
    seen = []
    tools._cik_map.cache_clear()
    with temp_cache(), patched(tools, _sec_get=fake_sec(routes, seen)):
        r = tools.get_latest_filing("heavy")
        assert r["url"] == "https://www.sec.gov/Archives/edgar/data/5/00000000052024000001/new1.htm", r
        assert r["filing_date"] == "2024-02-01" and r["period_of_report"] == "2023-12-31" and r["company"] == "Heavy Inc"
        assert any("5-submissions-001" in u for u in seen) and not any("5-submissions-002" in u for u in seen), seen
        r = tools.get_latest_filing("HEAVY", "10-Q")
        assert r["url"].endswith("/old1.htm") and r["filing_date"] == "2011-02-01", r
        miss = tools.get_latest_filing("HEAVY", "S-1")
        assert "error" in miss and "recent forms: 424B2, 8-K" in miss["error"] and "ETFs" not in miss["error"], miss
        op = tools.get_latest_filing("OPCO")
        assert "error" in op and "no 10-K found for OPCO" in op["error"] and "ETFs" not in op["error"], op
        assert "ETFs" in tools.get_latest_filing("FUND")["error"]
    tools._cik_map.cache_clear()


def test_registry_matches_schemas():
    names = [s["function"]["name"] for s in tools.TOOL_SCHEMAS]
    assert names == list(tools.TOOL_FUNCTIONS) and len(names) == 7, names
    for s in tools.TOOL_SCHEMAS:
        p = s["function"]["parameters"]
        assert set(p["required"]) <= set(p["properties"]) and s["function"]["description"]
    by = {s["function"]["name"]: s["function"]["parameters"] for s in tools.TOOL_SCHEMAS}
    assert by["read_filing"]["required"] == ["ticker"] and set(by["read_filing"]["properties"]) == {
        "ticker", "section", "query", "form_type"}
    assert by["read_filing"]["properties"]["section"]["enum"] == ["risk_factors", "mdna", "business", "legal_proceedings"]
    assert by["get_price_series"]["required"] == ["ticker"] and by["get_price_series"]["properties"]["period"]["enum"] == list(tools.PERIODS)


RISKS = [
    "The following risk factors could materially affect our results. They are not the only risks we face.",
    "Our supply chain depends on a single contract manufacturer in Asia.",
    "A disruption at our sole contract manufacturer, whether from a natural disaster, labor dispute or export "
    "restriction, would delay shipments of widgets to customers. We have no qualified second source for our core "
    "assembly, so supply chain recovery could take two quarters or longer.",
    "Cyber attacks could compromise customer data and harm our reputation.",
    "We store payment details and personal information. A breach could trigger regulatory fines and notification costs.",
    "Currency fluctuations may reduce the value of our international sales.",
    "Roughly 60% of revenue is earned outside the United States. A stronger dollar reduces reported revenue.",
    "Item 7 of this report describes our liquidity position.",
    "We face intense competition from larger companies with greater resources.",
    " ".join(f"Competitor number {i} may undercut our prices on widgets." for i in range(30)),
    "Changes in tax law could increase our effective tax rate.",
    "A change in corporate tax rates or the loss of a tax holiday could reduce net income and cash flow.",
    "Climate related events could disrupt our facilities and logistics networks.",
]
MDNA = [
    "Revenue grew 12% to $4.2 billion, driven by demand for widgets in Europe and Asia.",
    "Operating margin was 18%, up from 16%, as gross margin improved on lower freight costs.",
    "Liquidity: we ended the year with $900 million of cash and no outstanding borrowings under our credit facility.",
    "Artificial intelligence features lifted subscription renewals, and we plan to expand AI spending next year.",
]
BUSINESS = ["Fake Corp designs and sells widgets worldwide through retailers and online.",
            "We employ 4,100 people and operate assembly lines in three countries."]
LEGAL = "We are party to a patent infringement suit filed in Delaware in 2024 and believe the claims are without merit."


SPLIT_HEAD, _, SPLIT_TAIL = RISKS[4].partition(" regulatory")


def _paras(items):
    return "".join(f"<p><span>{htmllib.escape(t)}</span></p>" for t in items)


def _toc(rows):
    return "<table>" + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{n}</td></tr>" for a, b, n in rows) + "</table>"


TEN_K = f"""<html><head><title>fake-20251231</title><style>.x{{color:red}}</style></head><body>
<div style="display: none"><ix:header><ix:hidden><ix:nonnumeric>HIDDENFACT 0000001</ix:nonnumeric></ix:hidden></ix:header></div>
<script>var track = "SCRIPTNOISE";</script>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>Table of Contents</p>
{_toc([("Item&#160;1.", "Business", 3), ("Item 1A.", "Risk&#160;Factors", 5), ("Item 1B.", "Unresolved Staff Comments", 9),
       ("Item 3.", "Legal Proceedings", 10), ("Item 7.", "Management&#8217;s Discussion and Analysis", 11),
       ("Item 7A.", "Quantitative and Qualitative Disclosures About Market Risk", 14)])}
<div>PART I</div>
<div><span>Item&#160;1.</span><span>&#160;Business</span></div>
{_paras(BUSINESS)}<p>3</p>
<div>ITEM 1A.</div><div>RISK&nbsp;FACTORS</div>
{_paras(RISKS[:4])}{_paras([SPLIT_HEAD])}<p>- 7 -</p><div>Table of Contents</div>{_paras(["regulatory" + SPLIT_TAIL])}<div>Item 1A. Risk Factors (continued)</div>{_paras(RISKS[5:])}
<div>Item 1B. Unresolved Staff Comments</div><p>None.</p>
<div>Item 3. Legal Proceedings</div><p>{LEGAL}</p>
<div>PART II</div>
<div>ITEM 7.</div><div>MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</div>
{_paras(MDNA)}
<div>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</div><p>We hold no derivatives.</p>
</body></html>"""

QRISK = "There were no material changes to the risk factors in our Annual Report; supply chain constraints persist."
QMDNA = ["Quarterly revenue grew 9% on stronger widget demand.", "We repurchased no shares during the quarter."]
TEN_Q = f"""<html><body>
{_toc([("Item 1.", "Financial Statements", 2), ("Item 2.", "Management&#8217;s Discussion and Analysis", 8),
       ("Item 1A.", "Risk Factors", 12), ("Item 2.", "Unregistered Sales of Equity Securities", 13)])}
<div>PART I</div><div>Item 1. Financial Statements</div><p>Condensed statements follow.</p>
<div>Item 2. Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</div>{_paras(QMDNA)}
<div>Item 3. Quantitative and Qualitative Disclosures About Market Risk</div><p>Not material.</p>
<div>PART II</div><div>Item 1. Legal Proceedings</div><p>None.</p>
<div>Item 1A. Risk Factors</div><p>{QRISK}</p>
<div>Item 2. Unregistered Sales of Equity Securities and Use of Proceeds</div>{_paras([f"Repurchase table row {i} for the quarter." for i in range(12)])}
</body></html>"""


@contextlib.contextmanager
def fake_filings(seen_docs):
    tickers = {"0": {"ticker": "FAKE", "cik_str": 1}, "1": {"ticker": "ETF", "cik_str": 2}}
    sub = {"name": "Fake Inc", "filings": {"recent": {
        "form": ["10-Q", "10-K"], "accessionNumber": ["0000000001-26-000002", "0000000001-26-000001"],
        "primaryDocument": ["fake-10q.htm", "fake-10k.htm"], "filingDate": ["2026-05-01", "2026-02-01"],
        "reportDate": ["2026-03-31", "2025-12-31"]}}}
    etf = {"name": "Fake ETF", "filings": {"recent": {"form": ["N-CSR"]}}}
    docs = {"fake-10k.htm": TEN_K, "fake-10q.htm": TEN_Q}
    sec_urls = []

    def html(url):
        seen_docs.append(url)
        return docs[url.rsplit("/", 1)[1]]

    tools._cik_map.cache_clear()
    try:
        with patched(tools, _sec_get=fake_sec({"company_tickers": tickers, "CIK0000000001": sub, "CIK0000000002": etf}, sec_urls),
                     _filing_html=html):
            yield
    finally:
        tools._cik_map.cache_clear()


def test_paragraphs_strip_markup_and_noise():
    paras = tools._paragraphs(TEN_K)
    blob = "\n".join(paras)
    assert "HIDDENFACT" not in blob and "SCRIPTNOISE" not in blob and "fake-20251231" not in blob and "color:red" not in blob
    assert "Item 1A. Risk Factors 5" in paras and "Item 7. Management’s Discussion and Analysis 11" in paras, paras[:8]
    assert "\xa0" not in blob and "&amp;" not in blob
    assert not {"Table of Contents", "3", "- 7 -", "PART I", "PART II"} & set(paras), paras
    assert "ITEM 1A. RISK FACTORS" in paras and "Item 1. Business" in paras
    ent = tools._paragraphs("<p>Tom &amp; Jerry&#8217;s&nbsp;<b>widgets</b></p><p>A.<br>B.</p><div hidden style='DISPLAY:NONE'><div>x</div>y</div>Z")
    assert ent == ["Tom & Jerry’s widgets", "A.", "B.", "Z"], ent


def test_paragraphs_merge_page_break_fragments_only():
    split = tools._paragraphs("<p>Our sales could fall and</p><p>12</p><div>Table of Contents</div><p>our margins would shrink. In addition,</p>"
                              "<p>we may lose (or fail to renew)</p><p>key customers.</p>")
    assert split == ["Our sales could fall and our margins would shrink. In addition, we may lose (or fail to renew) key customers."], split
    keep = ["Risks include:", "higher costs;", "lower demand.", "“Done.”", "and then", "Macro Risks", "The text.",
            "Item 1A. Risk Factors", "and so on follows", "$5 million"]
    assert tools._paragraphs("".join(f"<p>{t}</p>" for t in keep)) == keep


def test_find_section_skips_toc_and_respects_boundaries():
    paras = tools._paragraphs(TEN_K)
    risk = tools._find_section(paras, "1A", r"risk\s+factors")
    assert risk[0] == RISKS[0] and risk[-1] == RISKS[-1] and len(risk) == len(RISKS) + 1, (len(risk), risk[:2])
    assert "Item 1A. Risk Factors (continued)" in risk and "None." not in risk
    assert "Item 7 of this report describes our liquidity position." in risk
    assert tools._find_section(paras, "7", r"management\W{0,3}s?\s*discussion") == MDNA
    assert tools._find_section(paras, "1", r"business\b") == BUSINESS
    assert tools._find_section(paras, "3", r"legal\s+proceedings") == [LEGAL]
    assert not tools._find_section(tools._paragraphs("<p>Item 1A. Risk Factors</p><p>Item 1B. Unresolved</p>"), "1A", r"risk\s+factors")
    q = tools._paragraphs(TEN_Q)
    assert tools._find_section(q, "2", r"management\W{0,3}s?\s*discussion") == QMDNA
    assert tools._find_section(q, "1A", r"risk\s+factors") == [QRISK]


def test_read_filing_ids_stable_ranking_and_caps():
    seen = []
    with temp_cache(), fake_filings(seen):
        overview = tools.read_filing("fake")
        assert "error" not in overview, overview
        keys = ["ticker", "company", "form_type", "filing_date", "period_of_report", "url", "section", "section_title",
                "query", "passages", "passage_count", "as_of", "source"]
        assert list(overview) == keys, list(overview)
        assert overview["ticker"] == "FAKE" and overview["company"] == "Fake Inc" and overview["form_type"] == "10-K"
        assert overview["filing_date"] == overview["as_of"] == "2026-02-01" and overview["period_of_report"] == "2025-12-31"
        assert overview["url"].endswith("/000000000126000001/fake-10k.htm") and overview["source"] == "SEC EDGAR"
        assert overview["section"] == "risk_factors" and overview["section_title"] == "Item 1A. Risk Factors"
        assert overview["query"] is None and overview["passage_count"] == len(RISKS) + 1 and is_json_safe(overview)
        titles = [p["text"] for p in overview["passages"]]
        assert titles[:3] == [RISKS[1], RISKS[3], RISKS[5]] and len(titles) == 6, titles
        assert [p["id"] for p in overview["passages"]] == ["p2", "p4", "p7", "p9", "p10", "p12"], overview["passages"]

        by_id = {p["id"]: p["text"] for p in overview["passages"]}
        for q in ("contract manufacturer asia", "Cyber attacks customer data", "currencies fluctuation", "tax", "competitor widgets"):
            r = tools.read_filing("FAKE", "RISK_FACTORS", q)
            assert 1 <= len(r["passages"]) <= 6 and r["query"] == q.lower() and "note" not in r, (q, r)
            for p in r["passages"]:
                assert by_id.setdefault(p["id"], p["text"]) == p["text"], (q, p)
                assert 0 < len(p["text"]) <= 700
            assert sum(len(p["text"]) for p in r["passages"]) <= 4200
        assert tools.read_filing("FAKE", "risk_factors", "contract manufacturer asia")["passages"][0] == {"id": "p2", "text": RISKS[1]}
        assert tools.read_filing("FAKE", "risk_factors", "cyber attacks customer data")["passages"][0]["id"] == "p4"
        assert tools.read_filing("FAKE", "risk_factors", "payment details notification costs")["passages"][0] == {"id": "p5", "text": RISKS[4]}
        assert tools.read_filing("FAKE", "risk_factors", "currencies fluctuations")["passages"][0]["id"] == "p7"
        tax = [p["id"] for p in tools.read_filing("FAKE", "risk_factors", "tax rate")["passages"]]
        assert tax[:2] == ["p12", "p13"], tax
        assert by_id["p2"] == RISKS[1] and by_id["p4"] == RISKS[3]

        long = tools.read_filing("FAKE", "risk_factors", "competitor widgets undercut")["passages"][0]
        assert long["id"] == "p11" and len(long["text"]) <= 700 and long["text"].endswith(".…"), long
        assert RISKS[9].startswith(long["text"][:-1])
        short = tools.read_filing("FAKE", "risk_factors", "cyber attacks")["passages"][0]["text"]
        assert short == RISKS[3] and not short.endswith("…")

        none = tools.read_filing("FAKE", "risk_factors", "zzqq")
        assert none["passages"] == [] and "no paragraph" in none["note"] and none["passage_count"] == len(RISKS) + 1
        assert tools.read_filing("FAKE", "risk_factors", "the of and")["passages"] == overview["passages"]

        ai = tools.read_filing("FAKE", "mdna", "AI")
        assert ai["passages"][0]["text"] == MDNA[3] and ai["section_title"].startswith("Item 7."), ai
        liq = tools.read_filing("fake", "mdna", "liquidity cash borrowings")
        assert liq["passages"][0]["text"] == MDNA[2] and liq["passage_count"] == len(MDNA)
        assert tools.read_filing("FAKE", "business")["passages"][0]["text"] == BUSINESS[0]
        assert tools.read_filing("FAKE", "legal_proceedings")["passages"] == [{"id": "p1", "text": LEGAL}]
        n_docs = len(seen)
        with env(OFFLINE="1"):
            cached = tools.read_filing("FAKE", "risk_factors", "tax rate")
        assert cached["stale"] is True and cached["passages"] and len(seen) == n_docs


def test_read_filing_10q_and_errors():
    seen = []
    with temp_cache(), fake_filings(seen), env(OFFLINE=None):
        q = tools.read_filing("FAKE", "mdna", form_type="10-q")
        assert q["form_type"] == "10-Q" and q["filing_date"] == "2026-05-01" and q["section_title"].startswith("Part I, Item 2")
        assert [p["text"] for p in q["passages"]] == QMDNA, q
        r = tools.read_filing("FAKE", "risk_factors", "supply chain", "10-Q")
        assert r["section_title"].startswith("Part II, Item 1A") and r["passages"] == [{"id": "p1", "text": QRISK}], r
        n = len(seen)
        for kw, needle in (({"section": "business", "form_type": "10-Q"}, "risk_factors, mdna"),
                           ({"section": "bogus"}, "risk_factors, mdna, business, legal_proceedings"),
                           ({"form_type": "8-K"}, "10-K, 10-Q")):
            err = tools.read_filing("FAKE", **kw)
            assert needle in err["error"] and err["source"] == "SEC EDGAR" and "passages" not in err, (kw, err)
        assert len(seen) == n
        miss = tools.read_filing("FAKE", "legal_proceedings", form_type="10-Q")
        assert "section must be" in miss["error"]
        etf = tools.read_filing("ETF")
        assert "error" in etf and "ETFs" in etf["error"] and "N-CSR" in etf["error"], etf
        for bad in ("", None, "A B"):
            assert "error" in tools.read_filing(bad)
        assert "error" in tools.read_filing("FAKE", unknown="x")
        seen.clear()
        with patched(tools, _filing_html=lambda url: "<html><body><p>Item 1A. Risk Factors</p><p>text</p></body></html>"):
            assert "could not locate Item 7" in tools.read_filing("FAKE", "mdna")["error"]
        with patched(tools, _filing_html=lambda url: (_ for _ in ()).throw(ConnectionError("down"))):
            err = tools.read_filing("FAKE", "business")
            assert "error" in err and err["source"] == "SEC EDGAR", err
        with env(OFFLINE="1"):
            assert "OFFLINE" in tools.read_filing("FAKE", "mdna", "never fetched")["error"] and not seen


def test_filing_html_download_cache_and_size_cap():
    class Resp:
        def __init__(self, body):
            self.body, self.closed = body, False

        def iter_content(self, n):
            for i in range(0, len(self.body), n):
                yield self.body[i:i + n]

        def close(self):
            self.closed = True

    calls = []
    resp = [Resp("<p>café ’</p>".encode("utf-8"))]

    def fetch(url, **kw):
        calls.append((url, kw))
        return resp[0]

    with temp_cache(), patched(tools, _sec_fetch=fetch):
        assert tools._filing_html("https://www.sec.gov/Archives/x/a.htm") == "<p>café ’</p>"
        assert tools._filing_html("https://www.sec.gov/Archives/x/a.htm") == "<p>café ’</p>"
        assert len(calls) == 1 and calls[0][1] == {"stream": True} and resp[0].closed
        assert len(list(tools.CACHE_DIR.glob("*.html"))) == 1
        resp[0] = Resp("<p>’</p>".encode("cp1252"))
        assert tools._filing_html("https://www.sec.gov/Archives/x/legacy.htm") == "<p>’</p>"
        resp[0] = Resp(b"x" * 5000)
        with patched(tools, MAX_DOC=4096):
            try:
                tools._filing_html("https://www.sec.gov/Archives/x/huge.htm")
                raise AssertionError("oversize filing should be refused")
            except ValueError as e:
                assert "larger than" in str(e)
        assert resp[0].closed and len(list(tools.CACHE_DIR.glob("*.html"))) == 2
    with patched(tools.requests, get=lambda url, headers, timeout, **kw: types.SimpleNamespace(status_code=404)):
        assert tools._sec_fetch("https://www.sec.gov/x", stream=True) is None


class SeriesTicker:
    closes = None
    period_seen = None
    info_fails = False

    def __init__(self, symbol):
        self.history_metadata = {"currency": "USD"}

    @property
    def info(self):
        if SeriesTicker.info_fails:
            raise ConnectionError("Too Many Requests")
        return {"shortName": "Series Corp"}

    def history(self, period=None, auto_adjust=True):
        assert auto_adjust is False
        SeriesTicker.period_seen = period
        c = SeriesTicker.closes
        if period == "ytd":
            c = c[c.index.year == c.index[-1].year]
        return pd.DataFrame({"Close": c})


@contextlib.contextmanager
def series_yf(values, start="2025-01-02", info_fails=False):
    idx = pd.bdate_range(start, periods=len(values), tz="America/New_York")
    SeriesTicker.closes = pd.Series(values, index=idx, name="Close")
    SeriesTicker.period_seen, SeriesTicker.info_fails = None, info_fails
    with temp_cache(), patched(tools, yf=types.SimpleNamespace(Ticker=SeriesTicker)):
        yield idx


def test_get_price_series_downsamples_and_computes_stats():
    n = 300
    keep = {round(i * (n - 1) / 129) for i in range(130)}
    spike = next(i for i in range(10, n - 10) if i not in keep)
    dip = next(i for i in range(spike + 1, n - 10) if i not in keep)
    vals = [100.0 + i * 0.5 for i in range(n)]
    vals[spike], vals[dip] = 500.0, 5.0
    with series_yf(vals) as idx:
        r = tools.get_price_series("series", "1Y")
        assert SeriesTicker.period_seen == "1y" and r["period"] == "1y"
        assert r["ticker"] == "SERIES" and r["name"] == "Series Corp" and r["currency"] == "USD"
        pts = r["points"]
        assert len(pts) == 130 and pts[0] == [idx[0].date().isoformat(), 100.0] and pts[-1] == [idx[-1].date().isoformat(), 249.5]
        dates = [d for d, _ in pts]
        assert dates == sorted(set(dates)) and all(isinstance(c, float) for _, c in pts)
        assert r["start_date"] == dates[0] and r["end_date"] == dates[-1] == r["as_of"]
        assert r["start_close"] == 100.0 and r["start_close_fmt"] == "$100.00" and r["end_close"] == 249.5 and r["end_close_fmt"] == "$249.50"
        assert r["change_pct"] == 149.5 and r["change_fmt"] == "+149.50%"
        assert r["high"] == 500.0 and r["high_fmt"] == "$500.00" and r["high_date"] == idx[spike].date().isoformat()
        assert r["low"] == 5.0 and r["low_fmt"] == "$5.00" and r["low_date"] == idx[dip].date().isoformat()
        assert r["high_date"] not in dates and r["low_date"] not in dates, "high/low must come from the full daily series"
        assert "Yahoo Finance" in r["source"] and is_json_safe(r)
    with series_yf([100.0, 101.0, 99.0, 96.9]):
        r = tools.get_price_series("SERIES", "1mo")
        assert len(r["points"]) == 4 and r["change_pct"] == -3.1 and r["change_fmt"] == "-3.10%", r
        assert r["high_date"] == r["points"][1][0] and r["low_date"] == r["points"][3][0]
    with series_yf([50.0, 52.0, 50.0]):
        r = tools.get_price_series("SERIES", "3mo")
        assert r["change_fmt"] == "+0.00%" and r["change_pct"] == 0.0 and str(r["change_pct"]) == "0.0"
    with series_yf([42.5]):
        r = tools.get_price_series("SERIES", "6mo")
        assert len(r["points"]) == 1 and r["start_date"] == r["end_date"] and r["change_fmt"] == "+0.00%"
    with series_yf([float(i + 1) for i in range(131)]):
        pts = tools.get_price_series("SERIES", "2y")["points"]
        assert len(pts) == 130 and pts[0][1] == 1.0 and pts[-1][1] == 131.0


def test_get_price_series_ytd_periods_and_errors():
    with series_yf([100.0 + i for i in range(400)], start="2025-01-02") as idx:
        r = tools.get_price_series("SERIES", "YTD")
        assert SeriesTicker.period_seen == "ytd" and r["period"] == "ytd"
        year = idx[-1].year
        assert r["start_date"].startswith(str(year)) and r["end_date"] == idx[-1].date().isoformat(), r
        first = next(i for i, d in enumerate(idx) if d.year == year)
        assert r["start_close"] == 100.0 + first and r["change_pct"] == round((499.0 / (100.0 + first) - 1) * 100, 2)
        assert tools.get_price_series("SERIES")["period"] == "1y" and SeriesTicker.period_seen == "1y"
        for p in tools.PERIODS:
            assert tools.get_price_series("SERIES", p)["period"] == p and SeriesTicker.period_seen == p
        for bad in ("10y", "max", "week", "1 year"):
            err = tools.get_price_series("SERIES", bad)
            assert err["error"] == "period must be one of: 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y" and "Yahoo" in err["source"], err
        assert "error" in tools.get_price_series("", "1y") and "error" in tools.get_price_series("SERIES", foo=1)
    with series_yf([100.0, 110.0], info_fails=True):
        r = tools.get_price_series("series")
        assert r["name"] is None and r["change_fmt"] == "+10.00%", r
    with series_yf([]):
        r = tools.get_price_series("NOPE")
        assert "no price history" in r["error"] and not list(tools.CACHE_DIR.glob("*.json")), r
    with series_yf([1.0, float("nan"), 2.0]):
        r = tools.get_price_series("SERIES")
        assert len(r["points"]) == 2 and is_json_safe(r)
        SeriesTicker.closes = pd.Series([1.0, 2.0], index=SeriesTicker.closes.index[[0, 2]])
        with env(OFFLINE="1"):
            assert tools.get_price_series("SERIES")["stale"] is True


def live_tools():
    def common(r, ok_keys):
        assert "error" not in r and not r.get("stale"), r
        assert all(k in r for k in ok_keys), (ok_keys, r)
        assert r["source"] and is_json_safe(r)

    for tk in ("AAPL", "MSFT", "BRK.B", "SPY"):
        q = tools.get_quote(tk)
        common(q, ["ticker", "name", "currency", "price", "price_fmt", "sma_200d", "above_sma_200d", "trailing_pe",
                   "market_cap_fmt", "dividend_yield_pct", "as_of"])
        assert q["price"] > 0 and q["price_fmt"].startswith("$"), q
    assert tools.get_quote("brk.b")["ticker"] == "BRK-B"
    assert tools.get_quote("AAPL")["market_cap_fmt"].endswith("T"), "AAPL cap should be in trillions"

    h = tools.get_price_history("AAPL", "2020-01-02")
    common(h, ["date_used", "close", "close_fmt"])
    assert h["date_used"] == "2020-01-02" and abs(h["close"] - 75.09) < 0.02, h
    assert tools.get_price_history("AAPL", "2020-01-04")["date_used"] == "2020-01-03"

    for tk, lo in (("AAPL", 3e11), ("MSFT", 2e11), ("BRK.B", 3e11)):
        f = tools.get_fundamentals(tk)
        common(f, ["revenue", "revenue_fmt", "net_income", "revenue_fy_end", "as_of"])
        assert f["revenue"] > lo, f
    assert tools.get_fundamentals("AAPL")["shares_outstanding"] > 1e10
    assert "error" in tools.get_fundamentals("SPY")

    for tk in ("AAPL", "MSFT", "BRK.B"):
        common(tools.get_latest_filing(tk), ["filing_date", "url", "form_type", "as_of"])
    spy = tools.get_latest_filing("SPY")
    assert "error" in spy and "N-CSR" in spy["error"], spy
    assert tools.get_latest_filing("SPY", "N-CSR").get("url"), "SPY N-CSR should exist"


def live_filings_and_series():
    def flat(url):
        raw = re.sub(r"<[^>]+>", "", tools._filing_html(url))
        return re.sub(r"\s+", " ", htmllib.unescape(raw))

    a = tools.read_filing("AAPL")
    assert "error" not in a and not a.get("stale") and a["form_type"] == "10-K" and is_json_safe(a), a
    assert a["section_title"] == "Item 1A. Risk Factors" and a["passage_count"] > 30 and len(a["passages"]) >= 3, a
    text = flat(a["url"])
    low = text.lower()
    for p in a["passages"]:
        assert re.fullmatch(r"p\d+", p["id"]) and len(p["text"]) <= 700, p
        at = text.find(p["text"].rstrip("…")[:60])
        assert at > 0, ("passage not in the filing", p)
        head = low.rfind("item 1a", 0, at)
        assert head > 0 and "item 1b" not in low[head:at], ("passage outside Item 1A", p)

    m = tools.read_filing("MSFT", "mdna", "artificial intelligence")
    assert "error" not in m and m["section_title"].startswith("Item 7."), m
    assert re.search(r"\bAI\b|artificial intelligence", m["passages"][0]["text"], re.I), m["passages"][0]
    assert m["query"] == "artificial intelligence" and m["passage_count"] > 50

    t = tools.read_filing("TSLA", "risk_factors", form_type="10-Q")
    assert "error" not in t and t["form_type"] == "10-Q" and t["passages"] and t["section_title"].startswith("Part II"), t

    spy = tools.read_filing("SPY")
    assert "error" in spy and "passages" not in spy and spy["source"] == "SEC EDGAR", spy

    old = tools.get_latest_filing("AAPL", "10-K405")
    assert "error" not in old and old["filing_date"] == "2001-12-21" and "/320193/" in old["url"], old
    assert "ETFs" not in tools.get_latest_filing("XOM").get("error", ""), "operating companies must not get the fund hint"

    n = tools.get_price_series("NVDA", "1y")
    assert "error" not in n and not n.get("stale") and is_json_safe(n), n
    assert 100 <= len(n["points"]) <= 130 and n["points"][0][0] == n["start_date"] and n["points"][-1][0] == n["end_date"] == n["as_of"]
    assert n["points"][0][1] == n["start_close"] and n["points"][-1][1] == n["end_close"] and n["currency"] == "USD"
    assert re.fullmatch(r"[+-]\d+\.\d{2}%", n["change_fmt"]) and n["change_fmt"] == f"{n['change_pct']:+.2f}%"
    assert n["low"] <= min(n["start_close"], n["end_close"]) <= max(n["start_close"], n["end_close"]) <= n["high"]
    assert n["start_close_fmt"].startswith("$") and n["name"] and 360 <= (dt.date.fromisoformat(n["end_date"]) - dt.date.fromisoformat(n["start_date"])).days <= 372
    assert tools.get_price_series("AAPL", "ytd")["start_date"].startswith(str(dt.date.today().year))


def main():
    live = os.environ.get("LIVE") == "1"
    tests = [(n, f) for n, f in sorted(globals().items()) if callable(f) and n.startswith("test_")]
    if live:
        tests += [("live_tools", live_tools), ("live_filings_and_series", live_filings_and_series)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed" + ("" if live else " (set LIVE=1 for network tests)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
