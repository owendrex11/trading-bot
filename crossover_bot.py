# /// script
# requires-python = ">=3.14"
# dependencies = ["alpaca-py", "yfinance"]
# ///
"""Moving average crossover bot for an Alpaca PAPER account.

    uv run --python 3.14 crossover_bot.py                       # check and trade if there is a crossover
    uv run --python 3.14 crossover_bot.py --dry-run             # check only, never place an order
    uv run --python 3.14 crossover_bot.py --market-hours-only   # skip unless the market is open (used by the schedule)

Settings live in config.toml (list as many symbols as you like). Keys come from the ALPACA_API_KEY / ALPACA_SECRET_KEY
environment variables (GitHub secrets), or from ~/.alpaca/alpaca.env on your own PC.

Rules:
- BUY  when the short average moves from at/below the long average yesterday to above it today.
- SELL when it moves from at/above yesterday to below it today.
- A sell only happens if you already hold enough shares (so it never short-sells).
- Each crossover is traded once: if an order on the same side was already placed today, it does nothing.
"""
import argparse
import os
import tomllib
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

HERE = Path(__file__).parent
CONFIG_FILE = HERE / "config.toml"
ENV_FILE = Path.home() / ".alpaca" / "alpaca.env"
EASTERN = ZoneInfo("America/New_York")


def load_keys():
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


def load_config():
    cfg = tomllib.loads(CONFIG_FILE.read_text())
    if "symbols" not in cfg:
        cfg["symbols"] = [cfg["symbol"]]  # older config files had a single symbol
    if not cfg["symbols"]:
        raise SystemExit("config.toml: symbols list is empty")
    if cfg["short_period"] >= cfg["long_period"]:
        raise SystemExit("config.toml: short_period must be smaller than long_period")
    if cfg["history_days"] < cfg["long_period"] + 1:
        raise SystemExit("config.toml: history_days must be more than long_period")
    return cfg


def get_closes(symbol, days, key, secret):
    """Daily closes from yfinance; falls back to Alpaca's free data if Yahoo refuses."""
    try:
        data = yf.Ticker(symbol).history(period=f"{days}d", interval="1d", auto_adjust=True)
        if not data.empty:
            return data["Close"]
        print("yfinance returned no data, using Alpaca price data instead.")
    except Exception as e:
        print(f"yfinance failed ({e}), using Alpaca price data instead.")
    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                           start=datetime.now(EASTERN) - timedelta(days=days), feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise SystemExit(f"No price data returned for {symbol}")
    closes = bars.xs(symbol)["close"]
    closes.index = closes.index.tz_convert(EASTERN)
    return closes


def find_signal(closes, short_n, long_n):
    """Return ("BUY"|"SELL"|None, details) by comparing the last two days."""
    short_ma = closes.rolling(short_n).mean()
    long_ma = closes.rolling(long_n).mean()
    if long_ma.iloc[-2:].isna().any():
        raise SystemExit(f"Not enough data: need at least {long_n + 1} days, got {len(closes)}")
    prev_diff = short_ma.iloc[-2] - long_ma.iloc[-2]
    curr_diff = short_ma.iloc[-1] - long_ma.iloc[-1]
    details = {
        "date": closes.index[-1].strftime("%Y-%m-%d"),
        "prev_date": closes.index[-2].strftime("%Y-%m-%d"),
        "close": float(closes.iloc[-1]),
        "short_prev": float(short_ma.iloc[-2]), "long_prev": float(long_ma.iloc[-2]),
        "short_now": float(short_ma.iloc[-1]), "long_now": float(long_ma.iloc[-1]),
    }
    if prev_diff <= 0 < curr_diff:
        return "BUY", details
    if prev_diff >= 0 > curr_diff:
        return "SELL", details
    return None, details


def already_traded_today(client, symbol, side):
    start = datetime.combine(datetime.now(EASTERN).date(), time(0), EASTERN)
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[symbol], after=start))
    return any(o.side == side and o.status.value not in ("canceled", "expired", "rejected") for o in orders)


def main():
    p = argparse.ArgumentParser(description="Moving average crossover bot (paper trading)")
    p.add_argument("--dry-run", action="store_true", help="show the signal but never place an order")
    p.add_argument("--market-hours-only", action="store_true",
                   help="do nothing unless the stock market is open right now (holidays included)")
    args = p.parse_args()

    key, secret = load_keys()
    client = TradingClient(key, secret, paper=True)  # paper=True: never real money

    if args.market_hours_only:
        clock = client.get_clock()
        if not clock.is_open:
            print(f"Market is closed. Next open: {clock.next_open.astimezone(EASTERN):%Y-%m-%d %H:%M} Eastern. Nothing to do.")
            return

    cfg = load_config()
    failed = []
    for symbol in dict.fromkeys(sym.strip().upper() for sym in cfg["symbols"]):
        print(f"\n=== {symbol} ===")
        try:
            check_symbol(client, symbol, cfg, key, secret, args.dry_run)
        except (Exception, SystemExit) as e:  # one bad ticker shouldn't stop the others
            print(f"Error checking {symbol}: {e}")
            failed.append(symbol)
    if failed:
        raise SystemExit(f"\nCould not check: {', '.join(failed)}")


def check_symbol(client, symbol, cfg, key, secret, dry_run):
    short_n, long_n, qty = cfg["short_period"], cfg["long_period"], cfg["trade_size"]

    closes = get_closes(symbol, cfg["history_days"], key, secret)
    signal, d = find_signal(closes, short_n, long_n)

    print(f"{symbol} close on {d['date']}: ${d['close']:,.2f}")
    print(f"  {d['prev_date']}: {short_n}-day ${d['short_prev']:,.2f} vs {long_n}-day ${d['long_prev']:,.2f}")
    print(f"  {d['date']}: {short_n}-day ${d['short_now']:,.2f} vs {long_n}-day ${d['long_now']:,.2f}")

    if signal is None:
        trend = "above" if d["short_now"] > d["long_now"] else "below"
        print(f"No crossover. The {short_n}-day average is still {trend} the {long_n}-day. No trade.")
        return

    print(f"Crossover found: {signal} signal.")
    side = OrderSide.BUY if signal == "BUY" else OrderSide.SELL

    if already_traded_today(client, symbol, side):
        print(f"Already placed a {signal} order for {symbol} today. No new order.")
        return

    if dry_run:
        print(f"Dry run: would {signal} {qty} {symbol} at market. No order placed.")
        return

    if signal == "SELL":
        try:
            held = float(client.get_open_position(symbol).qty)
        except APIError:
            held = 0
        if held < qty:
            print(f"You hold {held:g} {symbol}, need {qty}. Skipping the sell (no short selling).")
            return

    order = client.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY))
    print(f"Submitted: {order.side.value} {order.qty} {order.symbol} (market) status={order.status.value} id={order.id}")


if __name__ == "__main__":
    main()
