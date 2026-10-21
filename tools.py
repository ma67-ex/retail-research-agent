"""Tools the agent can call: market data, SEC filings lookup, calculator."""
import ast
import operator
import requests
import yfinance as yf

EDGAR_HEADERS = {"User-Agent": "retail-research-agent contact@example.com"}


def get_quote(ticker: str) -> dict:
    """Latest price, 200-day SMA, trailing P/E, market cap for a ticker."""
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    if hist.empty:
        return {"error": f"no price history for {ticker}"}
    price = float(hist["Close"].iloc[-1])
    sma200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else None
    info = t.info
    return {
        "ticker": ticker.upper(),
        "price": round(price, 2),
        "sma_200d": round(sma200, 2) if sma200 is not None else None,
        "above_sma_200d": price > sma200 if sma200 is not None else None,
        "trailing_pe": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
    }


def get_latest_filing(ticker: str, form_type: str = "10-K") -> dict:
    """Most recent filing of a given type for a ticker, via SEC EDGAR (free, no key)."""
    lookup = requests.get(
        "https://www.sec.gov/cgi-bin/browse-edgar",
        params={"action": "getcompany", "company": ticker, "type": form_type,
                "dateb": "", "owner": "include", "count": "1", "output": "atom"},
        headers=EDGAR_HEADERS, timeout=10,
    )
    tickers_map = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=EDGAR_HEADERS, timeout=10
    ).json()
    cik = None
    for row in tickers_map.values():
        if row["ticker"].upper() == ticker.upper():
            cik = str(row["cik_str"]).zfill(10)
            break
    if cik is None:
        return {"error": f"no CIK found for {ticker}"}

    submissions = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=EDGAR_HEADERS, timeout=10
    ).json()
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            accession = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            date = recent["filingDate"][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            return {"ticker": ticker.upper(), "form_type": form_type, "filing_date": date, "url": url}
    return {"error": f"no {form_type} found for {ticker}"}


_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> dict:
    """Evaluate a numeric arithmetic expression (+ - * / ** only, no names/calls)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return {"expression": expression, "result": _eval_node(tree.body)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get latest price, 200-day SMA, trailing P/E, and market cap for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "e.g. AAPL"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_filing",
            "description": "Get the most recent SEC filing (e.g. 10-K) for a ticker, with its URL and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "form_type": {"type": "string", "description": "e.g. 10-K, 10-Q", "default": "10-K"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a numeric arithmetic expression, e.g. '0.15 * 2900000000000'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_quote": get_quote,
    "get_latest_filing": get_latest_filing,
    "calculator": calculator,
}
