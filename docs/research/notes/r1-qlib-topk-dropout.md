# R1 note: Qlib `TopkDropoutStrategy` and `SoftTopkStrategy` (turnover control)

## Source

Microsoft Qlib, two files on branch `main`:

- `qlib/contrib/strategy/signal_strategy.py` (`BaseSignalStrategy`, `TopkDropoutStrategy`, `WeightStrategyBase`,
  `EnhancedIndexingStrategy`), 527 lines, sha256 `7ac8dbbbef2e0286131545ccb3fe60deb0a420748dd7477e18b1ab61cf78c1ca`;
- `qlib/contrib/strategy/cost_control.py` (`SoftTopkStrategy`), 117 lines, sha256
  `1f4387c6589a3359256f9d110ad9a0d1f04df5aa223e8bb1cf1f5ff67894bd9b`.

Fetched 2026-10-30 with curl from `https://raw.githubusercontent.com/microsoft/qlib/main/<path>`, each into its own empty
directory under the session scratch area. Commit SHA not recorded (GitHub API refused by the session proxy, HTTP 403);
identified by branch, date and hash. Licence: MIT, both files headed "Copyright (c) Microsoft Corporation. / Licensed under
the MIT License."

## Claim

The plan's question: how does it limit turnover, and does it map onto our keep band (`decision._top` with `keep_rank=30`,
`decision.backtest_weights` with `band=0.03`)?

**`TopkDropoutStrategy.generate_trade_decision`, as read:**

1. The score used at a step is the previous step's: `pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)`.
2. `last` = held names sorted by score. `today` = the best-scoring names not held, `self.n_drop + self.topk - len(last)` of them
   (`method_buy="top"`).
3. `comb` = `last` union `today`, sorted by score. With `method_sell="bottom"`: `sell = last[last.isin(get_last_n(comb, self.n_drop))]`,
   i.e. only held names that fall in the bottom `n_drop` of that combined list are sold. The code comment gives the reason:
   "In case of dropping higher score stock and buying lower score stock."
4. `buy = today[: len(sell) + self.topk - len(last)]`: as many newcomers as there are sales plus empty slots.
5. A sale is skipped if the name has been held fewer than `hold_thresh` steps (`current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh`; default `hold_thresh=1`), or if it is not tradable (limit-up/down rules: `forbid_all_trade_at_limit=True` by default).
6. Sizing: `value = cash * self.risk_degree / len(buy)`, where `cash` is the cash after the sales, and `risk_degree`
   defaults to 0.95. **Held names are never resized**: the order list contains sells of whole positions
   (`sell_amount = current_temp.get_stock_amount(code=code)`) and buys of new names only. Positions drift with prices.

So it limits turnover in two ways: at most `n_drop` names swapped per step, and no rebalancing of the names it keeps. It
has no rank threshold: a held name that falls to any rank stays as long as it is not among the bottom `n_drop` of `comb`.

**Equivalence with our H-D3 `dropout` (my derivation, not in the source).** With a full book (`len(last) == topk`), `today`
has `n_drop` names, and a held name is sold exactly when it is in the bottom `n_drop` of the `topk + n_drop` merged names.
The number sold equals the number of newcomers that land in the top `topk`; newcomer i (i-th best) lands there iff it
beats the i-th worst held name. That is the pairwise rule of `layers_b.dropout` ("swap ... only where the newcomer ranks
above the name it replaces"), so the selection is the same. The differences are in sizing, not selection: ours sets every
chosen name to `1/k` and lets the 3 % band in `backtest_weights` resize drifted holdings, Qlib never resizes and spends 95 %
of the freed cash; ours drops untradable or unscored holdings as forced drops counted against `n_drop`, Qlib skips trading
them; Qlib adds `hold_thresh`.

**`SoftTopkStrategy` (`cost_control.py`), as read.** A target-weight strategy: ideal weight `self.risk_degree / self.topk`
for each of the top `topk`. Names outside the top are sold and overweight names trimmed, each step capped by
`trade_impact_limit` ("Maximum weight change for each stock in one trade"; falls back to `max_sold_weight`, default 1.0,
i.e. no cap). The freed budget plus spare room under `risk_degree` is spread over underweight top names "based on its distance
to target", each buy also capped by the limit. This is partial trading by an absolute per-name cap per step, not by a
fraction `a` of the gap. The constructor stores `buy_method="first_fill"` but nothing in this file uses it. The order
generator it relies on (`OrderGenWInteract`, in `order_generator.py`) was not fetched or read.

**Map onto our keep band.** Partly. Our `_top` is a *rank* hysteresis: a held name stays while it ranks within 30, and any
number of names can be replaced at a rebalance if they fall out of the top 30. TopkDropout is a *count* cap: at most
`n_drop` replaced per step, whatever their ranks. Our 3 % `band` in `backtest_weights` is a *weight* hysteresis that Qlib's
TopkDropout replaces by never resizing at all. `SoftTopkStrategy`'s `trade_impact_limit` corresponds to neither; it is
closest to H-D1's partial step. The two controls are compatible: a count cap can sit on top of a rank band.

## Evidence quality

High for the mechanism: the code was read in full and the rules are explicit. These files contain no measurement of
turnover or of net returns; nothing here says the rule helps after costs. The equivalence argument above is mine and short
enough to check, but it was not tested against Qlib's code (no fetched code was run).

## Transfer to US large caps and to our costs

The rule is market-neutral and transfers as code. At our cost wall it matters more than it does in Qlib's benchmark: K1
found net thresholds of rank IC 0.15-0.30 with fees as the main reason, and "fewer and larger trades" among the few levers
left. Two features read here are relevant beyond what H-D3 already tests: (a) **never resizing kept names** removes every
trim/top-up trade, which at 10,000 SEK each pays the courtage minimum; (b) **`hold_thresh`**, a minimum holding period.
Neither creates edge; they only cut the cost of trading a weak one, so they cannot move the IC needed from 0.30 to the 0.03
we have.

## What we would test

- H-D3, service `h-d3-holding-period` (`layers_b.h_d3_variants`, `dropout(n_drop)` at 5-63 sessions): already the TopkDropout
  selection rule. A reading-driven question for the card: run the dropout variants with `band` large enough that kept
  names are never resized (Qlib's sizing) as a check of how much of the turnover is resizing.
- H-D1, service `h-d1-partial-trading`: `SoftTopkStrategy`'s absolute cap (`trade_impact_limit`) is an alternative form of the
  partial step; it would be a new variant needing its own card, not a change to the registered grid.
- X1, service `x1-qlib-check`: if the Qlib cross-check is extended to the portfolio, the rule as read here (shift=1 scores,
  `risk_degree` 0.95, no resizing) is the reference to match.

## Verdict

**adopt-to-test** (selection rule already in H-D3; the no-resize sizing is the part worth one more test).

## How it was read

`qlib/contrib/strategy/signal_strategy.py` and `qlib/contrib/strategy/cost_control.py`, branch `main`, fetched 2026-10-30 from
raw.githubusercontent.com; **full file read** of both (the `EnhancedIndexingStrategy` part read but not analysed: it needs a
risk-model data feed we do not have). `order_generator.py`, the optimizer, and the benchmark YAML configs (which hold the
`topk` and `n_drop` values actually used) were **not** read. Nothing was executed.
