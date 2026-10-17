# /// script
# requires-python = ">=3.14"
# dependencies = ["alpaca-py", "requests", "yfinance"]
# ///
"""Replay the crossover bot's rules over past years and compare them with simply buying an S&P 500 fund (SPY).

    uv run --python 3.14 backtest.py              # last 10 years, today's top 1000 stocks (from config.toml)
    uv run --python 3.14 backtest.py --years 5

It never places orders. It uses the same settings as the bot (config.toml) and the same $50,000 starting cash, and
also tries two possible extras: a stop-loss, and only buying while the S&P 500 is above its 200-day average.

Simplifications, so read the numbers as rough:
- Trades happen at each day's closing price (the bot now trades just before the close), with no fees or slippage.
- The stock list is TODAY's largest 1000. Companies that shrank or went bust over the years are missing, which
  flatters the strategy (it can only pick from stocks we already know did well enough to be big today).
- Prices include dividends (Yahoo's adjusted prices), for the stocks and SPY alike.
"""
import argparse
import math
import os

import pandas as pd
import yfinance as yf

import crossover_bot as bot

START_CASH = 50_000


def download(symbols, years):
    yahoo = {s.replace(".", "-"): s for s in symbols}
    data = yf.download(list(yahoo), period=f"{years}y", interval="1d", auto_adjust=True,
                       group_by="column", threads=True, progress=False)["Close"]
    return data.rename(columns=yahoo)


def simulate(closes, spy, cfg, stop_loss=None, market_filter=False):
    short_ma = closes.rolling(cfg["short_period"]).mean()
    long_ma = closes.rolling(cfg["long_period"]).mean()
    diff = short_ma - long_ma
    buy_sig = (diff.shift(1) <= 0) & (diff > 0)
    sell_sig = (diff.shift(1) >= 0) & (diff < 0)
    market_up = (spy > spy.rolling(200).mean()).reindex(closes.index, fill_value=False).to_numpy()
    syms = list(closes.columns)  # largest first, the same order the bot works through them
    px_all, buys, sells = closes.to_numpy(), buy_sig.to_numpy(), sell_sig.to_numpy()

    cash, held, trades, curve = START_CASH, {}, [], []  # held: column -> (shares, entry price, entry date)
    for i in range(cfg["long_period"], len(closes)):
        day, px = closes.index[i], px_all[i]
        for j in list(held):  # sells first, so their cash can fund today's buys
            shares, entry, since = held[j]
            if math.isnan(px[j]):
                continue
            if sells[i, j] or (stop_loss and px[j] < entry * (1 - stop_loss)):
                cash += shares * px[j]
                trades.append((px[j] / entry - 1, (day - since).days))
                del held[j]
        if not market_filter or market_up[i]:
            for j in buys[i].nonzero()[0]:
                if j in held or math.isnan(px[j]):
                    continue
                qty = max(1, math.floor(cfg["position_dollars"] / px[j]))
                if qty * px[j] <= cash:
                    cash -= qty * px[j]
                    held[j] = (qty, px[j], day)
        value = cash + sum(n * (e if math.isnan(px[j]) else px[j]) for j, (n, e, _) in held.items())
        curve.append((day, value))
    return pd.Series(dict(curve)), trades, len(held)


def stats(curve):
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = curve.iloc[-1] / curve.iloc[0]
    drawdown = (curve / curve.cummax() - 1).min()
    return curve.iloc[-1], total ** (1 / years) - 1, drawdown


def main():
    p = argparse.ArgumentParser(description="Backtest the crossover rules against buying SPY")
    p.add_argument("--years", type=int, default=10)
    args = p.parse_args()

    cfg = bot.load_config()
    symbols = bot.top_by_market_cap(cfg["top_market_cap"]) if cfg["top_market_cap"] else cfg["symbols"]
    print(f"Downloading {args.years} years of prices for {len(symbols)} stocks and SPY...")
    data = download(symbols + ["SPY"], args.years)
    spy = data.pop("SPY").dropna()
    closes = data.loc[spy.index].dropna(axis=1, how="all")
    print(f"Got prices for {closes.shape[1]} stocks, {spy.index[0]:%Y-%m-%d} to {spy.index[-1]:%Y-%m-%d}.")

    start = closes.index[cfg["long_period"]]
    spy_curve = START_CASH * spy.loc[start:] / spy.loc[start]
    rows = [("Just buy SPY and hold", *stats(spy_curve), "", "", "")]
    for name, kw in [("Bot as it is now", {}),
                     ("Bot + 10% stop-loss", {"stop_loss": 0.10}),
                     ("Bot + buy only when market is rising", {"market_filter": True}),
                     ("Bot + both", {"stop_loss": 0.10, "market_filter": True})]:
        curve, trades, open_now = simulate(closes, spy, cfg, **kw)
        wins = sum(r > 0 for r, _ in trades)
        win_rate = f"{wins / len(trades):.0%}" if trades else "-"
        avg_days = f"{sum(d for _, d in trades) / len(trades):.0f}" if trades else "-"
        rows.append((name, *stats(curve), f"{len(trades):,}", win_rate, avg_days))

    lines = [f"## Backtest: {start:%b %Y} to {spy.index[-1]:%b %Y}, ${START_CASH:,} start, "
             f"${cfg['position_dollars']:,} per buy, {closes.shape[1]} stocks",
             "", "| Strategy | Ending value | Per year | Worst drop | Trades | Win rate | Avg days held |",
             "|---|---|---|---|---|---|---|"]
    for name, end, cagr, dd, n, wr, days in rows:
        lines.append(f"| {name} | ${end:,.0f} | {cagr:+.1%} | {dd:.0%} | {n} | {wr} | {days} |")
    lines += ["", "Trades at closing prices, no fees. The stock list is today's largest companies, which flatters the bot."]
    report = "\n".join(lines)
    print("\n" + report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
