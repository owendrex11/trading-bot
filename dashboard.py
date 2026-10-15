# /// script
# requires-python = ">=3.14"
# dependencies = ["alpaca-py", "cryptography", "requests", "yfinance"]
# ///
"""Builds the portfolio dashboard website from the Alpaca PAPER account.

    uv run --python 3.14 dashboard.py                  # writes site/ with password-locked data (needs DASHBOARD_PASSWORD)
    uv run --python 3.14 dashboard.py --no-password    # writes unlocked data, for looking at it on your own PC only

It also runs the bot's crossover check (same settings and code as crossover_bot.py, but it never trades)
so the page can list the stocks that crossed recently.

The page itself is dashboard/index.html. This script copies it to site/ next to a data.json holding the
account numbers. GitHub Pages sites can be opened by anyone with the link, so data.json is encrypted with
the DASHBOARD_PASSWORD secret and the page asks for that password before it shows anything.
"""
import argparse
import base64
import hashlib
import json
import os
import secrets
import shutil
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import crossover_bot as bot

HERE = Path(__file__).parent
PAGE = HERE / "dashboard" / "index.html"
PBKDF2_ITERATIONS = 310_000  # the page reads this from data.json
CHART_DAYS = 400               # calendar days of prices kept for each stock's chart (about a year of trading days)
LOGO_SOURCES = [               # tried in order; BRK.B is written BRK-B
    "https://financialmodelingprep.com/image-stock/{}.png",
    "https://assets.parqet.com/logos/symbol/{}?format=png",
]


def num(x):
    return None if x is None else float(x)


def all_filled_orders(client):
    """Every filled order, oldest first (Alpaca returns at most 500 per request)."""
    orders, after = [], datetime(2000, 1, 1, tzinfo=timezone.utc)
    while True:
        page = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500,
                                                  direction="asc", after=after))
        orders += [o for o in page if o.filled_at and float(o.filled_qty or 0) > 0]
        if len(page) < 500:
            break
        after = page[-1].submitted_at
    orders.sort(key=lambda o: o.filled_at)
    return orders


def closed_trades(orders):
    """Match each sell against the oldest shares bought before it (first in, first out).

    Each matched lot is one finished trade: a win if it sold for more than it cost.
    """
    lots = defaultdict(deque)  # symbol -> [qty, price] still held
    trades = []
    for o in orders:
        qty, price = float(o.filled_qty), float(o.filled_avg_price)
        if o.side.value == "buy":
            lots[o.symbol].append([qty, price, o.filled_at])
            continue
        while qty > 1e-9 and lots[o.symbol]:
            lot = lots[o.symbol][0]
            take = min(qty, lot[0])
            trades.append({
                "symbol": o.symbol, "qty": take, "buy_price": lot[1], "sell_price": price,
                "opened": lot[2].isoformat(), "closed": o.filled_at.isoformat(),
                "pl": (price - lot[1]) * take, "pl_pct": (price / lot[1] - 1) * 100,
            })
            lot[0] -= take
            qty -= take
            if lot[0] <= 1e-9:
                lots[o.symbol].popleft()
    return trades


def history(client, period, timeframe):
    h = client.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe=timeframe))
    points = [(t, e) for t, e in zip(h.timestamp, h.equity) if e]  # 0/None = before the account existed
    return [{"t": t, "equity": float(e)} for t, e in points]


def crossover_scan(key, secret, held, lookback=5):
    """Stocks whose short average crossed the long one in the last `lookback` trading days."""
    cfg = bot.load_config()
    symbols, universe = None, None
    if cfg["top_market_cap"]:
        try:
            symbols = bot.top_by_market_cap(cfg["top_market_cap"])
            universe = f"top {len(symbols)} US stocks by market cap"
        except Exception as e:
            print(f"Could not download the market cap list ({e}). Using the symbols list instead.")
    if not symbols:
        symbols = list(dict.fromkeys(sym.strip().upper() for sym in cfg["symbols"]))
        universe = f"{len(symbols)} stock(s) from config.toml"
    short_n, long_n = cfg["short_period"], cfg["long_period"]
    closes = bot.get_all_closes(symbols, cfg["history_days"], key, secret)

    signals, checked, above, as_of = [], 0, 0, None
    for sym in symbols:
        c = closes.get(sym)
        if c is None or len(c) < long_n + 1:
            continue
        checked += 1
        short_ma, long_ma = c.rolling(short_n).mean(), c.rolling(long_n).mean()
        diff = (short_ma - long_ma).iloc[-(lookback + 1):]
        above += diff.iloc[-1] > 0
        as_of = max(as_of or "", c.index[-1].strftime("%Y-%m-%d"))
        for i in range(len(diff) - 1, 0, -1):  # newest crossover first
            prev, cur = diff.iloc[i - 1], diff.iloc[i]
            if prev != prev:  # NaN: not enough history this far back
                break
            if prev <= 0 < cur or prev >= 0 > cur:
                signals.append({
                    "symbol": sym, "signal": "BUY" if cur > 0 else "SELL",
                    "date": diff.index[i].strftime("%Y-%m-%d"), "days_ago": len(diff) - 1 - i,
                    "close": float(c.iloc[-1]), "short": float(short_ma.iloc[-1]), "long": float(long_ma.iloc[-1]),
                    "gap_pct": float((short_ma.iloc[-1] / long_ma.iloc[-1] - 1) * 100),
                    "held": sym in held,
                })
                break
    signals.sort(key=lambda s: (s["days_ago"], s["symbol"]))
    print(f"Crossover scan: {checked} stock(s) checked, {len(signals)} crossed in the last {lookback} trading days.")
    return {"universe": universe, "checked": checked, "above": int(above), "short": short_n, "long": long_n,
            "lookback": lookback, "as_of": as_of, "signals": signals}


def price_history(symbols, key, secret):
    """Daily closes for each stock's chart, on one shared list of dates (missing days are null)."""
    closes = bot.get_all_closes(sorted(symbols), CHART_DAYS, key, secret)
    day = lambda d: d.strftime("%Y-%m-%d")
    dates = sorted({day(d) for c in closes.values() for d in c.index})
    out = {}
    for sym, c in closes.items():
        by_day = {day(d): round(float(v), 2) for d, v in c.items()}
        out[sym] = [by_day.get(d) for d in dates]
    return {"dates": dates, "closes": out}


def fetch_logos(symbols, folder, password):
    """Save each company logo next to the page, so the page never asks a logo site for your tickers.

    File names are scrambled with the password, so nobody can tell which stocks are shown by guessing names.
    """
    folder.mkdir(parents=True, exist_ok=True)

    def one(sym):
        for url in LOGO_SOURCES:
            try:
                r = requests.get(url.format(sym.replace(".", "-")), timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            except requests.RequestException:
                continue
            if r.ok and r.headers.get("content-type", "").startswith("image/") and len(r.content) > 200:
                name = hashlib.sha256(f"{password}:{sym}".encode()).hexdigest()[:20] + ".png"
                (folder / name).write_bytes(r.content)
                return sym, name
        return sym, None

    with ThreadPoolExecutor(16) as pool:
        logos = {sym: name for sym, name in pool.map(one, sorted(symbols)) if name}
    print(f"Logos: found {len(logos)} of {len(symbols)}.")
    return logos


def build_data(client, key, secret):
    acct = client.get_account()
    positions = client.get_all_positions()
    orders = all_filled_orders(client)
    trades = closed_trades(orders)
    daily = history(client, "1A", "1D")
    intraday = history(client, "1D", "5Min")

    try:  # the dashboard still publishes if the price download fails
        scan = crossover_scan(key, secret, {p.symbol for p in positions})
    except Exception as e:
        print(f"Crossover scan failed: {e}")
        scan = {"error": str(e)}

    shown = {p.symbol for p in positions} | {o.symbol for o in orders[-50:]} | {t["symbol"] for t in trades[-50:]}
    shown |= {s["symbol"] for s in scan.get("signals", [])}
    try:
        charts = price_history(shown, key, secret)
    except Exception as e:
        print(f"Chart prices failed: {e}")
        charts = {"error": str(e)}

    equity, last_equity = float(acct.equity), float(acct.last_equity)
    start = daily[0]["equity"] if daily else equity
    wins = [t for t in trades if t["pl"] > 0]
    losses = [t for t in trades if t["pl"] <= 0]
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "account": {
            "equity": equity, "cash": float(acct.cash), "buying_power": float(acct.buying_power),
            "long_market_value": float(acct.long_market_value or 0),
            "last_equity": last_equity, "start_equity": start,
            "start_date": datetime.fromtimestamp(daily[0]["t"], timezone.utc).date().isoformat() if daily else None,
        },
        "positions": sorted(({
            "symbol": p.symbol, "qty": float(p.qty), "avg_entry": float(p.avg_entry_price),
            "price": num(p.current_price), "market_value": num(p.market_value), "cost_basis": float(p.cost_basis),
            "pl": num(p.unrealized_pl), "pl_pct": None if p.unrealized_plpc is None else float(p.unrealized_plpc) * 100,
            "today_pct": None if p.change_today is None else float(p.change_today) * 100,
        } for p in positions), key=lambda p: -(p["market_value"] or 0)),
        "trades": {
            "count": len(trades), "wins": len(wins), "losses": len(losses),
            "realized_pl": sum(t["pl"] for t in trades),
            "avg_win_pct": sum(t["pl_pct"] for t in wins) / len(wins) if wins else None,
            "avg_loss_pct": sum(t["pl_pct"] for t in losses) / len(losses) if losses else None,
            "recent": trades[::-1][:50],
        },
        "orders": [{
            "symbol": o.symbol, "side": o.side.value, "qty": float(o.filled_qty),
            "price": float(o.filled_avg_price), "time": o.filled_at.isoformat(),
        } for o in orders[::-1][:50]],
        "history": {"daily": daily, "intraday": intraday},
        "crossovers": scan,
        "charts": charts,
        "shown": sorted(shown),
    }


def encrypt(data, password):
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=PBKDF2_ITERATIONS).derive(password.encode())
    ciphertext = AESGCM(key).encrypt(iv, json.dumps(data).encode(), None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"encrypted": True, "iterations": PBKDF2_ITERATIONS, "salt": b64(salt), "iv": b64(iv), "data": b64(ciphertext)}


def main():
    p = argparse.ArgumentParser(description="Build the portfolio dashboard site")
    p.add_argument("--out", default=str(HERE / "site"), help="folder to write (default: site)")
    p.add_argument("--no-password", action="store_true", help="write the data unlocked (only for viewing on your own PC)")
    args = p.parse_args()

    password = os.environ.get("DASHBOARD_PASSWORD", "")
    if not args.no_password and not password:
        raise SystemExit("Missing DASHBOARD_PASSWORD. Add it as a repository secret (see README).")

    key, secret = bot.load_keys()
    data = build_data(TradingClient(key, secret, paper=True), key, secret)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(PAGE, out / "index.html")
    try:
        data["logos"] = fetch_logos(data["shown"], out / "logos", password)
    except Exception as e:
        print(f"Logos failed: {e}")
        data["logos"] = {}
    payload = {"encrypted": False, "data": data} if args.no_password else encrypt(data, password)
    (out / "data.json").write_text(json.dumps(payload))
    print("Dashboard built.")  # no account numbers here: run logs are public if the repository is


if __name__ == "__main__":
    main()
