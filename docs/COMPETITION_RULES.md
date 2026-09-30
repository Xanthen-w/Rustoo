# Competition rules that shape this bot

Summary of the official Roostoo × Susquehanna hackathon rules, limited to what
changes design decisions. The organizers' published text is authoritative.

## Timeline (2026)

| Date | Event |
|---|---|
| Sep 29 | Team registration deadline |
| Oct 1 – Oct 3 | Preparation: build the bot, test deployment on Roostoo |
| **Oct 4 – Oct 17** | **Live trading, 14 full days** (strategies may be iterated and redeployed) |
| before Oct 14 | Submit the open-source repo link (with README) |
| Oct 21 | Top 15 finalists (5 per region) |
| Oct 27 | Finalist deck / markdown explaining the strategy |

## How teams are ranked (in order)

1. **Rule compliance (pass/fail).**
   - *Trade-log integrity:* consistent, autonomous execution matching the declared strategy.
   - *Commit-history transparency:* every strategy change traceable in git; **no traces of
     manually called APIs**. The competition credentials are only ever used by the bot.
2. **Portfolio return:** top 20 per region by `(final - initial) / initial` advance.
   A strategy that sits in cash can't qualify, however good its risk metrics.
3. **Composite score:** `0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar` over the live period.
4. **Code review:** clear strategy logic, clean maintained repo, runs continuously on Roostoo.

## Constraints

- **At least 8 active trading days**, with "enough trades made from strategies each day".
  The bot must trade daily, not only on rare regime switches.
- $100,000 mock portfolio. Spot only, "1x long and short", **no leverage**.
- Fees: **0.1% taker** (market orders), **0.05% maker** (limit orders), matching `config/config.yaml`.
- No high-frequency trading, market-making or arbitrage; excessive requests get failed responses.
- Must run on the provided AWS EC2 instance, autonomously.
- Any data source may be used (Roostoo only covers cloud costs).
- Organizers recommend logging every trade, performance metrics, and the success/failure of every API request.

## Consequences for this repo

- **Research evaluates 14-day windows.** That's the scored horizon. Every candidate is measured on
  return, composite score and trading days per window (`scripts/window_analysis.py`).
- **The bot keeps its own audit trail:** every API request's outcome, every order and fill, and
  periodic equity snapshots, persisted across restarts.
- **Strategy changes go through commits only.** No hand-run scripts that place orders with the
  competition key.
- **Shorting:** the rules text says "1x long and short", but Roostoo can disable shorts per competition
  on the server side (docs/API_NOTES.md open question #1). This stays off
  (`execution.allow_shorting: false`) until the bot itself confirms it works during the prep period.
- **Rate limit:** the official text sets no numeric limit. `execution.min_seconds_between_orders`
  is a client-side throttle, 10 s between orders since 2026-09-30 (was 60 s). A rebalance places
  at most a few orders, so this stays far from high-frequency trading.

## Organizer clarifications (received 2026-09-28)

Questions sent by the team; answers paraphrased from the organizers' reply.

| # | Question | Answer | Consequence for this repo |
|---|---|---|---|
| 1 | What is an "active trading day" / "enough trades"? | A day on which the bot placed **at least 1 trade**. The trade history should be consistent with the stated strategy, not look like manual trading. | Met by the scheduled rebalances (every 6h since 2026-09-28; once a day already qualified). |
| 2 | How to apply code updates during the live round? | Update on the cloud machine; teams may stop, restart and redeploy the bot themselves. | Runbook: commit → `git pull` on EC2 → `sudo systemctl restart rustoo-bot`. The commit history is the audit trail. |
| 3 | Shorts: liquidation? Native stop orders? | Losses are capped at the collateral (a short's value can reach zero, not go negative). **No native stop-loss/take-profit orders**: only the documented endpoints exist, so a bot must monitor prices and send its own market close. The only slippage is the price moving between the bot's check and its order. | Any stop logic lives in the bot. Shorting is possible (see 4). |
| 4 | Long and short in the same pair at once? | **Yes.** | Shorts can be researched as a separate sleeve; `execution.allow_shorting` stays `false` until that research is done and shorts are tested through the bot. |
| 5 | Data source / symbols for bStocks? | Use the symbols exactly as `/v3/exchangeInfo` returns them; Binance (or another source) is fine for history. | Matches what `src/data/binance.py` does. |
| 6 | Exact Sharpe / Sortino / Calmar formulas and sampling? | **Not answered.** | Annualization and return sampling remain unknown; research keeps comparing strategies on the same basis rather than trusting absolute scores. |
| 7 | Idempotency: how to check an order after a timeout? | Verify with `/v3/query_order`. | Already how `LiveBroker` reconciles an order whose outcome is unknown (it never blindly resubmits). |
