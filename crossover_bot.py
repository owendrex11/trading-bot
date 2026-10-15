# /// script
# requires-python = ">=3.14"
# dependencies = ["alpaca-py", "requests", "yfinance"]
# ///
"""Moving average crossover bot for an Alpaca PAPER account.

    uv run --python 3.14 crossover_bot.py                       # check and trade if there is a crossover
    uv run --python 3.14 crossover_bot.py --dry-run             # check only, never place an order
    uv run --python 3.14 crossover_bot.py --market-hours-only   # skip unless the market is open (used by the schedule)

Settings live in config.toml: either your own list of symbols, or the N largest US stocks by market cap. Keys come from the ALPACA_API_KEY / ALPACA_SECRET_KEY
environment variables (GitHub secrets), or from ~/.alpaca/alpaca.env on your own PC.

Rules:
- BUY  when the short average moves from at/below the long average yesterday to above it today.
- SELL when it moves from at/above yesterday to below it today.
- A buy is about position_dollars worth of whole shares (at least 1 share).
- A sell sells every share you hold of that stock, and is skipped if you hold none (it never short-sells).
- Each crossover is traded once: if an order on the same side was already placed today, it does nothing.
"""
import argparse
import math
import os
import tomllib
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
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
    cfg.setdefault("top_market_cap", 0)
    if not cfg["symbols"] and not cfg["top_market_cap"]:
        raise SystemExit("config.toml: symbols list is empty")
    if cfg["short_period"] >= cfg["long_period"]:
        raise SystemExit("config.toml: short_period must be smaller than long_period")
    if cfg["history_days"] < cfg["long_period"] + 1:
        raise SystemExit("config.toml: history_days must be more than long_period")
    return cfg


def top_by_market_cap(n):
    """The n largest US-listed stocks by market cap, from Nasdaq's public stock screener."""
    r = requests.get("https://api.nasdaq.com/api/screener/stocks",
                     params={"tableonly": "true", "limit": "10000", "download": "true"},
                     headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    stocks = []
    for row in rows:
        sym = row["symbol"].strip().replace("/", ".")
        if not sym or "^" in sym or " " in sym:  # skip preferred shares and odd listings
            continue
        try:
            cap = float(row.get("marketCap") or 0)
        except ValueError:
            continue
        if cap > 0:
            stocks.append((cap, sym))
    stocks.sort(reverse=True)
    if len(stocks) < n // 2:
        raise RuntimeError(f"Nasdaq returned only {len(stocks)} stocks")
    return [sym for _, sym in stocks[:n]]


def get_all_closes(symbols, days, key, secret):
    """Daily closes for every symbol: one batch from yfinance, Alpaca's free data for any it misses."""
    closes = {}
    yahoo = {sym.replace(".", "-"): sym for sym in symbols}  # Yahoo writes BRK.B as BRK-B
    try:
        data = yf.download(list(yahoo), period=f"{days}d", interval="1d", auto_adjust=True,
                           group_by="column", threads=True, progress=False)
        if not data.empty:
            col = data["Close"]
            for ysym, sym in yahoo.items():
                if ysym in col:
                    series = col[ysym].dropna()
                    if not series.empty:
                        closes[sym] = series
    except Exception as e:
        print(f"yfinance failed ({e}).")

    missing = [s for s in symbols if s not in closes]
    if missing:
        print(f"Getting {len(missing)} stock(s) from Alpaca price data instead of yfinance.")
        client = StockHistoricalDataClient(key, secret)
        for i in range(0, len(missing), 200):
            chunk = missing[i:i + 200]
            try:
                req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                       start=datetime.now(EASTERN) - timedelta(days=days), feed=DataFeed.IEX)
                bars = client.get_stock_bars(req).df
            except Exception as e:
                print(f"Alpaca price data failed for {len(chunk)} stock(s): {e}")
                continue
            if bars.empty:
                continue
            for sym in chunk:
                if sym in bars.index.get_level_values(0):
                    series = bars.xs(sym)["close"]
                    series.index = series.index.tz_convert(EASTERN)
                    closes[sym] = series
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
    top_n = cfg["top_market_cap"]
    symbols = None
    if top_n:
        try:
            symbols = top_by_market_cap(top_n)
            print(f"Following the {len(symbols)} largest US stocks by market cap.")
        except Exception as e:
            print(f"Could not download the market cap list ({e}). Using the symbols list instead.")
    if not symbols:
        top_n = 0
        symbols = list(dict.fromkeys(sym.strip().upper() for sym in cfg["symbols"]))

    short_n, long_n = cfg["short_period"], cfg["long_period"]
    all_closes = get_all_closes(symbols, cfg["history_days"], key, secret)
    verbose = len(symbols) <= 10
    no_data, signals, failed, too_pricey = [], [], [], []
    for symbol in symbols:
        closes = all_closes.get(symbol)
        if closes is None or len(closes) < long_n + 1:
            no_data.append(symbol)  # no prices, or too new to have a 50-day average
            continue
        signal, d = find_signal(closes, short_n, long_n)
        if verbose or signal:
            print(f"\n=== {symbol} ===")
            print(f"{symbol} close on {d['date']}: ${d['close']:,.2f}")
            print(f"  {d['prev_date']}: {short_n}-day ${d['short_prev']:,.2f} vs {long_n}-day ${d['long_prev']:,.2f}")
            print(f"  {d['date']}: {short_n}-day ${d['short_now']:,.2f} vs {long_n}-day ${d['long_now']:,.2f}")
        if signal is None:
            if verbose:
                trend = "above" if d["short_now"] > d["long_now"] else "below"
                print(f"No crossover. The {short_n}-day average is still {trend} the {long_n}-day. No trade.")
            continue
        signals.append(f"{signal} {symbol}")
        try:
            if act_on_signal(client, symbol, signal, float(closes.iloc[-1]), cfg, args.dry_run) == "too pricey":
                too_pricey.append(symbol)
        except Exception as e:  # one bad order shouldn't stop the others
            print(f"Error placing order for {symbol}: {e}")
            failed.append(symbol)

    checked = len(symbols) - len(no_data)
    summary = f"Checked {checked} stock(s). Crossovers: {', '.join(signals) or 'none'}."
    if no_data:
        summary += f" Skipped {len(no_data)} without enough price history"
        summary += f" ({', '.join(no_data[:20])}{'...' if len(no_data) > 20 else ''})."
    if too_pricey:
        summary += f" Not enough cash to buy: {', '.join(too_pricey)}."
    if failed:
        summary += f" Order errors: {', '.join(failed)}."
    print("\n" + summary)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=Crossover bot::{summary}")  # shows on the run's summary page
    if failed or (no_data and not top_n):  # with your own list, a missing ticker is probably a typo
        raise SystemExit(1)


def buy_quantity(price, cfg):
    """Whole shares worth about position_dollars; at least 1 share even if one share costs more."""
    dollars = cfg.get("position_dollars")
    if not dollars:
        return cfg.get("trade_size", 1)  # older config files bought a fixed number of shares
    return max(1, math.floor(dollars / price))


def act_on_signal(client, symbol, signal, price, cfg, dry_run):
    print(f"Crossover found: {signal} signal.")
    side = OrderSide.BUY if signal == "BUY" else OrderSide.SELL

    if already_traded_today(client, symbol, side):
        print(f"Already placed a {signal} order for {symbol} today. No new order.")
        return

    if signal == "BUY":
        qty = buy_quantity(price, cfg)
        what = f"{qty} {symbol} (about ${qty * price:,.2f} at ${price:,.2f})"
        cash = float(client.get_account().buying_power)
        if qty * price > cash:  # e.g. one BRK.A share costs more than the whole account
            print(f"Not enough buying power (${cash:,.2f}) to buy {what}. Skipping.")
            return "too pricey"
    else:
        try:
            qty = float(client.get_open_position(symbol).qty)
        except APIError:
            qty = 0
        if qty <= 0:
            print(f"You don't hold any {symbol}. Skipping the sell (no short selling).")
            return
        what = f"all {qty:g} {symbol} you hold"

    if dry_run:
        print(f"Dry run: would {signal} {what} at market. No order placed.")
        return

    order = client.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY))
    print(f"Submitted: {order.side.value} {what} (market) status={order.status.value} id={order.id}")

if __name__ == "__main__":
    main()
