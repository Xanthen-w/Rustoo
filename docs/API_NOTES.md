# Roostoo API Notes

Source of truth: the official repo
[`roostoo/Roostoo-API-Documents`](https://github.com/roostoo/Roostoo-API-Documents)
(cloned and read in full on 2026-09-22). Everything below is taken directly from
that README, `python_demo.py`, `PartnerDocument.md`, and `partner_python_demo.py` —
nothing here is guessed. Where something is genuinely undocumented or ambiguous,
it is called out explicitly under "Open questions / assumptions" instead of being
silently assumed.

## Base URL

```
https://mock-api.roostoo.com
```

This is a mock exchange (paper trading against a simulated order book), not a
sandbox for a separate live exchange. There is no separate "live" Roostoo trading
API for this competition — `mock-api.roostoo.com` **is** the competition's
execution venue. Our internal `LIVE_TRADING` flag (see `config/config.yaml`)
governs whether *our bot* is allowed to submit real orders to this mock API, not
whether the API itself is a sandbox.

## Authentication

Two access levels, both apply to increasingly more endpoints:

- `RCL_TSCheck`: requires a `timestamp` parameter (13-digit millisecond epoch).
- `RCL_TopLevelCheck` ("SIGNED"): requires everything `RCL_TSCheck` requires,
  plus two HTTP headers:
  - `RST-API-KEY`: the API key.
  - `MSG-SIGNATURE`: `HMAC-SHA256(secretKey, totalParams)` as a lowercase hex
    digest.

`totalParams` is built by taking every parameter for the request (including
`timestamp`), sorting the keys lexicographically, and joining as
`key1=value1&key2=value2&...`. For a GET request this is exactly the query
string; for a POST request this is exactly the `application/x-www-form-urlencoded`
request body. **Only the documented parameters for an endpoint are used to build
the signature server-side** — sending an extra undocumented parameter and
including it in your own signature will make the signature invalid.

Timestamp tolerance: request is rejected unless
`abs(serverTime - timestamp) <= 60_000` ms. Clock drift on the host (e.g. EC2)
matters, so the client never trusts the local wall clock blindly: it measures
the offset to `/v3/serverTime` before its first timestamped request and again
every 5 minutes, stamps every request with local time + that offset, and logs a
warning if the offset exceeds half the tolerance (`_BaseClient.sync_clock`).

All POST requests must set `Content-Type: application/x-www-form-urlencoded`.

Verified against the doc's worked example (`pair=BNB/USD&quantity=2000&side=BUY&
timestamp=1580774512000&type=MARKET` → signature `20b7fd5550b67b3bf0c1684ed0f04
885261db8fdabd38611e9e6af23c19b7fff`) with a unit test
(`tests/test_signature.py`) so the signing implementation is provably correct,
not just "looks right".

## Rate limits / trading constraints from the competition

- No official numeric rate limit is published in this repo, but hackathon
  materials describe an enforced limit of **at most 1 trade per minute** for
  this specific competition (HFT / market-making / arbitrage are explicitly
  banned by the competition rules given to us, independent of this repo). The
  client throttles every trade-submitting call (`place_order`, `short_open`,
  `short_close`) defensively. Read-only and idempotent calls treat `429`/`5xx`
  and network errors as retryable; **order-creating calls are never retried** —
  a timeout or `5xx` there raises `RoostooOrderStateUnknownError`, because the
  order may have executed server-side, and the caller must reconcile via
  `query_order`/`get_balance` before trying again.
- **No leverage.** Regular orders (`/v3/place_order`) are plain spot buy/sell —
  no margin parameter exists, so leverage is not something the API even
  exposes for that endpoint.

## Endpoints implemented in `src/execution/client.py`

Public (`RCL_NoVerification` or `RCL_TSCheck`, no signature):

| Method | Path | Notes |
|---|---|---|
| GET | `/v3/serverTime` | No params. `{"ServerTime": <13-digit ms>}`. Used for clock-sync. |
| GET | `/v3/exchangeInfo` | No params. Returns `IsRunning`, `InitialWallet`, and `TradePairs` keyed by pair (e.g. `"BTC/USD"`) with `Coin`, `CoinFullName`, `Unit`, `UnitFullName`, `CanTrade`, `PricePrecision` (decimal places for price), `AmountPrecision` (decimal places for quantity), `MiniOrder` (minimum `price*quantity` notional). This is the sole source of the tradable universe — see `src/data/universe.py`. |
| GET | `/v3/ticker` | `timestamp` required, `pair` optional. Returns `MaxBid`, `MinAsk`, `LastPrice`, `Change` (24h fraction), `CoinTradeValue`, `UnitTradeValue` per pair. **This is a live snapshot only — there is no historical OHLCV endpoint anywhere in this API.** If `pair` is omitted, every listed pair's ticker is returned in one call — cheaper than polling per-symbol. |

Private / signed (`RCL_TopLevelCheck`):

| Method | Path | Notes |
|---|---|---|
| GET | `/v3/balance` | `timestamp`. Returns `Wallet` map of `{ASSET: {Free, Lock}}`. |
| GET | `/v3/pending_count` | `timestamp`. Returns `TotalPending` and `OrderPairs` counts. Note: when there are zero pending orders the API still returns HTTP 200 but with `Success: false` and `ErrMsg: "no pending order under this account"` — this is a normal "empty" response, not an error, and must not be treated as a request failure. |
| POST | `/v3/place_order` | `pair, side(BUY/SELL), type(LIMIT/MARKET), quantity, timestamp`, `price` required iff `type=LIMIT`. Response always HTTP 200; check `Success`. Includes `Role` (`TAKER`/`MAKER`), `FilledQuantity`, `FilledAverPrice`, `CommissionCoin`, `CommissionChargeValue`, `CommissionPercent` — **the realized fee and role must be read from this field, never assumed** (maker vs taker is decided by the exchange, not requested by us). |
| POST | `/v3/query_order` | `timestamp` + optional `order_id` OR `pair`(+`offset`,`limit`,`pending_only`). `order_id` excludes all other optional params. Default `limit` is 100 if omitted. Returns `OrderMatched: []`, or `Success: false, ErrMsg: "no order matched"` if nothing matches — again a normal empty case, not a hard error. |
| POST | `/v3/cancel_order` | `timestamp` + optional `order_id` OR `pair` (not both). Neither given cancels **all** pending orders on the account — the client requires an explicit opt-in flag to call it with neither, to prevent an accidental full-account cancel. |
| POST | `/v6/short_open` | `pair, collateral, timestamp`, optional `order_type=LIMIT`+`price` (else market). Sizes by collateral, not quantity; loss is capped at collateral (see below). |
| POST | `/v6/short_close` | `pair, timestamp`, optional `close_qty` or `close_pct` (qty takes precedence; neither closes 100%). Always reduce-only. |
| GET | `/v6/short_positions` | `timestamp`. Live open shorts with `UnrealizedPNL`/`UnrealizedPNLPct`. |

General response quirks documented in the repo (apply everywhere):

- A failed request is still HTTP 200 with `Success: false` and an `ErrMsg` — the
  client must check `Success`, never rely on HTTP status alone.
- A response field that is exactly zero is **omitted** from the JSON rather than
  sent as `0` (e.g. `OpenFee` missing means fee was 0). The client's response
  models must treat missing numeric fields as `0`, not `None`/error.

## Fees (confirmed from the docs, not assumed)

- Spot orders (`/v3/place_order`): fee is `CommissionPercent * FilledQuantity *
  FilledAverPrice`, taken from the actual order response — `CommissionPercent`
  differs by `Role` (maker vs taker) and is not published as a fixed constant in
  this repo. The hackathon brief we were given states **taker 0.10% / maker
  0.05%** — we use those as the backtest cost-model defaults
  (`backtest/costs.py`), but the live client always uses the real
  `CommissionChargeValue` from the order response, never the assumed constant.
- Shorting: flat `0.1%` of position value on open (`OpenFee`) and again on the
  closed notional on close (`CloseFee`), both confirmed in the docs with worked
  numeric examples.

## Precision / minimum order rules

From `/v3/exchangeInfo` per pair: round order `price` to `PricePrecision`
decimals and `quantity` to `AmountPrecision` decimals, and reject/resize any
order where `price * quantity < MiniOrder`. Implemented in
`src/data/universe.py::TradingRule.clamp_order` /
`TradingRule.meets_min_notional`. Truncation is done in decimal arithmetic
(`truncate_to_decimals`), not `floor(x * 10**d) / 10**d`, which is off by one
unit for values like `0.29`. Numbers are sent to the API as plain decimal
strings (`client.format_decimal`), never `str(float)`, which would produce
scientific notation such as `1e-05` for small quantities.

## Open questions / assumptions (do not silently resolve — confirm before relying on live)

1. **Is shorting permitted by this specific competition's rules?** The API
   supports it, loss is capped at collateral (so arguably not "leverage" in the
   sense the competition rules prohibit), but the documented error
   `"this competition does not allow short positions"` shows Roostoo disables
   shorting per-competition on the server side. **Assumption for now: baseline
   strategies are long-only / cash and never call the short endpoints.** The
   short client methods are implemented (so the option exists) but unused by
   any strategy until this is confirmed with organizers.
2. **No historical OHLCV endpoint exists.** Per user decision, the concrete
   historical data source for backtesting is not yet chosen (`src/data/
   market_data.py::HistoricalDataSource` is a pluggable interface with no
   assumption about file layout baked in). The live bot itself only ever reads
   `/v3/ticker`; if we want our own historical bars going forward we must build
   them by polling and storing ticker snapshots over time (see
   `src/data/roostoo_data.py::TickerBarBuilder`), since Roostoo cannot give us
   history retroactively.
3. **Exact API rate limit isn't in this repo.** We rely on hackathon materials
   for "max 1 trade/minute"; if this changes, `config/config.yaml:
   execution.min_seconds_between_orders` is the single place to adjust it.
4. **Not every pair in `exchangeInfo` has ticker data.** Verified live on
   2026-09-22: `exchangeInfo` listed 88 tradable pairs, but the
   no-`pair`-argument call to `/v3/ticker` returned only 86 of them. The
   universe/data layer must not assume every `CanTrade: true` pair has a
   current ticker — `MarketDataFeed.get_ticker` raises `KeyError` in that
   case, and any code iterating the universe to fetch prices should be
   prepared to skip a pair with no ticker rather than treat it as an error.
5. **Whether `MAKER` orders can be requested explicitly.** The docs show
   `Role` as an *outcome* of a `LIMIT` order resting on the book, not an input
   parameter — there is no way to force maker execution. Cost-model
   "optimistic/base/pessimistic" scenarios in `backtest/costs.py` exist
   precisely because real maker/taker mix is not controllable or fully
   predictable ahead of time.
