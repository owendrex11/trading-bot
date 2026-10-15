# Crossover trading bot

Buys or sells 1 share of each followed stock (the 1000 largest US stocks by market cap, or your own list; see `config.toml`) on an Alpaca **paper** account when the 20-day moving average
crosses the 50-day. Settings are in `config.toml`.

GitHub runs it every 15 minutes on weekdays while the market is open
(`.github/workflows/crossover-bot.yml`). Each run's output is under the **Actions** tab.

- To test by hand: Actions > Crossover bot > Run workflow (leave "Check only" ticked).
- To pause it: Actions > Crossover bot > "..." > Disable workflow.
- Keys are stored as repository secrets `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
