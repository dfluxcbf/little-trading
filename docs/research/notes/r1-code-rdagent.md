# R1 note: RD-Agent (Microsoft), README and the Qlib scenario's evaluation code

## Source

Repository `microsoft/RD-Agent`, branch `main`, fetched 2026-10-30 from raw.githubusercontent.com: `README.md` (41,910
bytes) and, for the quant scenario, `rdagent/app/qlib_rd_loop/conf.py`, `factor.py`, `quant.py`,
`rdagent/scenarios/qlib/developer/feedback.py`, `developer/factor_runner.py`, `experiment/workspace.py`,
`experiment/utils.py`, `experiment/factor_template/conf_baseline.yaml`, `conf_combined_factors.yaml`, `read_exp_res.py`,
`experiment/model_template/conf_baseline_factors_model.yaml`, `proposal/factor_proposal.py`, `scenarios/qlib/prompts.yaml`.
Paths beyond the README were guessed from the class paths in `conf.py`; every guessed path returned 200. The paper
(arXiv 2505.15155, "R&D-Agent-Quant") was not opened (arXiv is not reachable from this sandbox). Plan question: how
candidates are generated and how leakage is handled.

## Claim

**How candidates are generated** (code):

- A loop of LLM roles, configured in `conf.py`: `hypothesis_gen` (`QlibFactorHypothesisGen` / `QlibModelHypothesisGen`),
  `hypothesis2experiment`, `coder` (`QlibFactorCoSTEER`, which writes factor code), `runner` (Qlib back-test in Docker),
  `summarizer` (`...Experiment2Feedback`). `evolving_n: int = 10`. The joint "quant" loop chooses factor or model per round,
  `action_selection: str = "bandit"` ("'bandit' for bandit-based selection, 'llm' ..., 'random' ...").
- The hypothesis prompt is filled with the whole trace of earlier hypotheses and their feedback
  (`factor_proposal.py`: `hypothesis_and_feedback`, `last_hypothesis_and_feedback`), and asks for "An exact, testable, and
  innovative statement derived from previous experimental trace analysis" (`prompts.yaml` line 66).
- New factors that duplicate the current best set are dropped: `deduplicate_new_factors` keeps a new column only if its
  maximum mean daily IC with any SOTA factor is below 0.99 (`IC_max[IC_max < 0.99]`, `factor_runner.py`).
- README claim (not checked): "at a cost under $10, RD-Agent(Q) achieves approximately 2× higher ARR than benchmark factor
  libraries while using over 70% fewer factors."

**How the result is scored and kept** (code):

- Every candidate is back-tested by Qlib with LightGBM (or the current best model) on CSI 300; default segments
  `train 2008-01-01..2014-12-31`, `valid 2015-01-01..2016-12-31`, `test 2017-01-01..2020-08-01` (`conf.py`, identical for the
  factor, model and quant settings). The back-test (`port_analysis_config`) runs on `test_start..test_end` with
  `TopkDropoutStrategy` `topk: 50`, `n_drop: 5`, costs `open_cost: 0.0005`, `close_cost: 0.0015`, `min_cost: 5`, account
  `100000000`. Label `"Ref($close, -2)/Ref($close, -1) - 1"`.
- `read_exp_res.py` takes the latest Qlib recorder's metrics into `qlib_res.csv`; `workspace.py` returns that as the
  experiment's `result`. `feedback.py` compares current against SOTA on
  `IMPORTANT_METRICS = ["IC", "1day.excess_return_with_cost.annualized_return", "1day.excess_return_with_cost.max_drawdown"]`
  and an LLM returns `"Replace Best Result": "yes or no"`. The system prompt says: "Any small improvement should be
  considered for inclusion as SOTA" and "If the new factor(s) shows an improvement in the annualized return, recommend it to
  replace the current best result" (`prompts.yaml` lines 188-189).
- **Leakage handling found:** fixed chronological segments; the feature normaliser is fitted on the training segment only
  (`RobustZScoreNorm` with `fit_start_time` / `fit_end_time` = train range); labels are dropped where missing
  (`DropnaLabel`). **Not found in the files read:** any purge between segments, any held-out period that the loop never
  sees, any count of trials or multiple-testing correction. The annualised excess return and max drawdown the LLM uses to
  keep or reject a candidate come from the back-test over the **test** segment, and the next hypothesis is written from
  that feedback. So the loop selects on its test window round after round; whether Qlib's `SigAnaRecord` IC is also
  computed on the test segment is Qlib behaviour not verified in this session.

## Evidence quality

Medium for the mechanics (code read directly, the chain `factor_runner -> workspace.execute -> read_exp_res.py ->
qlib_res.csv -> feedback.py` checked file by file). The performance claim ("2x ARR") is the README's; the paper was not
read, and from the code read, the reported test-period numbers are what the agent optimised against, so they are
in-sample for the search.

## Transfer to US large caps and to our costs

- **Generation idea:** an LLM proposing and coding price-volume factors is a cheap way to produce many candidates; it does
  not bring new information (the inputs are the same daily OHLCV), and our H-T1 Alpha158 test covers the same space with
  fixed formulas.
- **Leakage:** the loop's design is the opposite of our protocol (plan section 1, rules 3 and 8): every round reads the test
  window and keeps "any small improvement". Each round would be a trial in our `trials` table; with 10+ rounds per run and
  several runs, the deflated Sharpe and PBO would have to absorb hundreds of trials. An LLM proposer also knows market
  history up to its own cutoff (plan, "Not pursued": "an LLM has seen our backtest years").
- **Costs and K1:** their back-test is a 100M-CNY account holding 50 names with 0.05 % / 0.15 % costs and a 5 CNY minimum;
  our K1 wall (rank IC 0.15 at 50,000 SEK, 0.30 at 10,000 SEK, top-10, net) is set by the courtage minimum at small size.
  A factor library tuned for a 2-day label on CSI 300 says nothing about that wall.
- **One transferable detail:** the near-duplicate filter (drop a new factor whose IC-correlation with a kept factor is
  at least 0.99) is a cheap guard if we ever add generated features; for H-T1 it would cut redundant Alpha158 columns.

## What we would test

No hypothesis from RD-Agent itself (LLM-agent frameworks are "Not pursued" in the plan, section 5). One small, optional
check inside H-T1 (`h-t1-alpha158`): before training, drop features whose mean daily cross-sectional rank correlation with an
earlier feature is above 0.99, and report how many of the ~158 survive; no extra variant unless the count is large.

## Verdict

**read-only.** A useful negative example: candidate generation by LLM with selection on the test segment and no trial
count. Nothing to adopt beyond the duplicate-feature filter.

## How it was read

`curl -sS -f` from `https://raw.githubusercontent.com/microsoft/RD-Agent/main/<path>` into a new directory
`/tmp/claude-0/-home-user-little-trading/366cbf7a-883e-5fa6-b341-e7246410c34d/scratchpad/r1/rdagent/` on 2026-10-30; all 14
files returned 200 on `main` (no 404s). Paths after the README were inferred from the dotted class names in
`rdagent/app/qlib_rd_loop/conf.py` and from `qlib_config_name=` arguments in `factor_runner.py`. Read with `grep` and
`sed`. No fetched code was executed. Not read: `rdagent/components/` (CoSTEER coder internals), the model runner, the paper.
