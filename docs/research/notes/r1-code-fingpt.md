# R1 note: FinGPT (AI4Finance), README and FinGPT-Forecaster data and evaluation code

## Source

Repository `AI4Finance-Foundation/FinGPT`, branch `master` (404 on `main`), fetched 2026-10-30 from
raw.githubusercontent.com: `README.md` (32,497 bytes) and, under `fingpt/FinGPT_Forecaster/`: `README.md`,
`RLSP_GUIDE.md`, `data.py`, `basics.py`, `prompt.py`, `data_pipeline.py`, `indices.py`, `utils.py`, `train_lora.py`,
`app.py`. `fingpt/FinGPT_Benchmark/README.md` returned 404 on `master` and `main` (not read). Paths were guessed from the
README's links and from `import` lines (`from basics import select_published_basic`). Hugging Face (models, datasets,
demo) and arXiv were not reachable. Plan question: how candidates are generated and how leakage is handled.

## Claim

FinGPT is a family (sentiment models, multi-task instruction models, RAG, a benchmark); the part that forecasts prices is
**FinGPT-Forecaster**, which is what this note reads.

**What it predicts and how training targets are made** (code):

- Input: a company profile, 1 to 3 past weeks of price moves and up to 5 sampled Finnhub news items per week, optionally
  the latest quarterly basic financials; output: "positive developments", "potential concerns" and "a **prediction** of stock
  price movement for the coming week". Base model Llama-2-7b-chat, LoRA (`r=8`, `lora_alpha=16`).
- The label is the realised weekly return of the adjusted close (`resample('W')`), binned as up/down by whole percent
  ("5+" above 5 %) (`data.py` `get_returns`, `bin_mapping`).
- The training *answer* is written by GPT-4, which is told the realised outcome: the prompt ends "Then let's assume your
  prediction for next week ({start_date} to {end_date}) is {prediction}. Provide a summary analysis to support your
  prediction." with `prediction = map_bin_label(row['Bin Label'])` (`prompt.py`). The model therefore learns rationales
  written with knowledge of the answer.
- Released model: "trained on DOW30 market data from 2022-12-30 to 2023-9-1" (Forecaster README).

**Evaluation** (code): per symbol, a chronological split, first `train_ratio=0.8` of weeks to train, the rest to test
(`data.py` `create_dataset`), no gap between them. Metrics (`utils.py` `calc_metrics`): binary up/down accuracy, MSE of the
binned move, ROUGE of the generated text against GPT-4's text. During training a callback generates on 50 shuffled test
examples (`original_dataset['test'].shuffle(seed=42).select(range(50))`, `train_lora.py`), and the `Trainer` gets
`eval_dataset=dataset['test']`; no `load_best_model_at_end` appears in the file, so the test set is watched during
training, not visibly used to pick a checkpoint.

**Leakage handling found:**

- Point-in-time financials, added after a reported bug: `basics.py` ("selecting by `period` alone puts figures into a
  prompt before the market had them (see issue #280)") treats a quarter as public `DEFAULT_PUBLICATION_LAG_DAYS = 45`
  after its period end and selects "the latest quarter that was public strictly before `decision_date`", the start of the
  predicted week.
- News for a week is fetched with `company_news(symbol, _from=start_date, to=end_date)` for that past week and goes into the
  prompt of later weeks only (`prompt.py` keeps `prev_rows`).
- **Not handled, from the code read:** the company introduction is the profile fetched at data-preparation time ("As of
  today, {name} has a market capitalization of ..."), pasted into every historical prompt; the base LLM's own pre-training
  knowledge of 2022-2023 events is not addressed; and the GPT-4-written rationales carry the outcome.
- `RLSP_GUIDE.md` is a design note, not code: "The environment must apply the action after the observation timestamp.
  News, financial statements, and market data released later must not appear in the state for that step", and "Evaluate
  once on the held-out test period". It says "no validated RLSP trainer is included".

## Evidence quality

Medium for the mechanics (code read directly). Low for any forecasting claim: the README gives no accuracy figure for the
Forecaster in the files read; the training window is about 35 weeks of 30 stocks, and its test split is the last fifth of
that window; the sentiment benchmark table (weighted F1 on FPB, FiQA-SA, TFNS, NWGI) measures sentiment labelling, not
returns.

## Transfer to US large caps and to our costs

- **Target mismatch:** a weekly up/down call per stock, evaluated by accuracy, is not a cross-sectional rank and carries no
  cost model; K1 says our top-10 layer needs rank IC 0.15 (50,000 SEK) to 0.30 (10,000 SEK) net, and accuracy on 30 Dow
  names in 2023 says nothing about that.
- **Leakage:** the LLM base and GPT-4-written targets both know the period; the plan already excludes LLM-agent forecasting
  for this reason ("an LLM has seen our backtest years").
- **What transfers:** the point-in-time rule for financials (public 45 days after quarter end when no filing date is known)
  is a sensible fallback, but our H-I2 design is stricter: as-filed XBRL facts keyed by the actual filing date. Sentiment
  models (FinGPT v3, weighted F1 0.882 on FPB per the README) are an alternative to FinBERT for H-I1 text features, but they
  need a GPU ("1 × RTX 3090" in the README table), and our lab runs on CPU.

## What we would test

Nothing new. For H-I1 (`h-i1-edgar-earnings`) and H-I2 (`h-i2-fundamentals`) the existing designs already key text and
facts by filing time; if a filing date is ever missing, use a fixed lag of at least 45 days after period end, as
`basics.py` does, and record that as a fallback in the service's log.

## Verdict

**not useful** for scoring (weekly direction calls, outcome-aware training targets, tiny window, no costs). Read-only value:
the 45-day publication lag as a fallback rule.

## How it was read

`curl -sS -f` from `https://raw.githubusercontent.com/AI4Finance-Foundation/FinGPT/{master,main}/<path>` into a new directory
`/tmp/claude-0/-home-user-little-trading/366cbf7a-883e-5fa6-b341-e7246410c34d/scratchpad/r1/fingpt/` on 2026-10-30. 404s:
`README.md` on `main` (200 on `master`), `fingpt/FinGPT_Benchmark/README.md` on both branches. Read with `grep` and `sed`:
`get_returns`, `get_news`, `get_basics`, `create_dataset`, `basics.py` in full, `get_all_prompts` and `PROMPT_END`,
`calc_metrics`, the training callback and `Trainer` set-up. No fetched code was executed. Not read: `prepare_data.ipynb`,
`demo.ipynb`, the Hugging Face model and dataset cards, the papers.
