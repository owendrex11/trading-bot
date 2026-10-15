# Crossover trading bot

On an Alpaca **paper** account, follows the 1000 largest US stocks by market cap (or your own list). When a stock's
20-day moving average crosses above the 50-day it buys about $500 of it (`position_dollars`); when it crosses below,
it sells all the shares held. Settings are in `config.toml`.

GitHub runs it every 15 minutes on weekdays while the market is open
(`.github/workflows/crossover-bot.yml`). Each run's output is under the **Actions** tab.

- To test by hand: Actions > Crossover bot > Run workflow (leave "Check only" ticked).
- To pause it: Actions > Crossover bot > "..." > Disable workflow.
- Keys are stored as repository secrets `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.

## Dashboard website

`dashboard.py` builds a portfolio website (balance, holdings, gain/loss, win rate, chart) from the
Alpaca account, and `.github/workflows/dashboard.yml` publishes it to GitHub Pages after every bot run.
The page itself is `dashboard/index.html`.

Anyone with the link can open a GitHub Pages site, so the account numbers are encrypted with a password
and the page asks for it before showing anything. One-time setup:

1. Settings > Secrets and variables > Actions > New repository secret: name `DASHBOARD_PASSWORD`,
   value = the password you want to type on the website.
2. Settings > Pages > Build and deployment > Source: **GitHub Actions**.
3. Actions > Dashboard > Run workflow. When it finishes, the website address is shown on the run page
   (it looks like `https://owendrex11.github.io/trading-bot/`).

The page also lists the stocks whose 20-day average crossed the 50-day in the last 5 trading days. It runs the
bot's own check with the same `config.toml` settings, but never places orders.

Win rate counts finished trades: each sell is matched with the oldest shares bought before it, and it is
a win if it sold for more than it cost.
