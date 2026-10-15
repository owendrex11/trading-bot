# /// script
# requires-python = ">=3.14"
# dependencies = ["alpaca-py", "cryptography"]
# ///
"""Builds the portfolio dashboard website from the Alpaca PAPER account.

    uv run --python 3.14 dashboard.py                  # writes site/ with password-locked data (needs DASHBOARD_PASSWORD)
    uv run --python 3.14 dashboard.py --no-password    # writes unlocked data, for looking at it on your own PC only

The page itself is dashboard/index.html. This script copies it to site/ next to a data.json holding the
account numbers. GitHub Pages sites can be opened by anyone with the link, so data.json is encrypted with
the DASHBOARD_PASSWORD secret and the page asks for that password before it shows anything.
"""
import argparse
import base64
import json
import os
import secrets
import shutil
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

HERE = Path(__file__).parent
ENV_FILE = Path.home() / ".alpaca" / "alpaca.env"
PAGE = HERE / "dashboard" / "index.html"
PBKDF2_ITERATIONS = 310_000  # the page reads this from data.json


def load_keys():
    """Same as crossover_bot.py: GitHub secrets, or ~/.alpaca/alpaca.env on your own PC."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")
    return key, secret


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


def build_data(client):
    acct = client.get_account()
    positions = client.get_all_positions()
    orders = all_filled_orders(client)
    trades = closed_trades(orders)
    daily = history(client, "1A", "1D")
    intraday = history(client, "1D", "5Min")

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

    key, secret = load_keys()
    data = build_data(TradingClient(key, secret, paper=True))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(PAGE, out / "index.html")
    payload = {"encrypted": False, "data": data} if args.no_password else encrypt(data, password)
    (out / "data.json").write_text(json.dumps(payload))
    print("Dashboard built.")  # no account numbers here: run logs are public if the repository is


if __name__ == "__main__":
    main()
