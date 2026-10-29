# Improvement research: everything found (2026-10-29)

Question: after the bake-off (`bakeoff-summary.md`: no alternative beats the controls, costs are the second wall), what
should be improved in **decision making** and **model training**? This file is the inventory of what the research
found. The plan that uses it is `docs/plans/improvements-lab.md`. Nothing here changes the running system.

How to read it: each entry says what the source is, what it claims, how well we know that (read in full, read as an
abstract, or only seen in a listing), and whether it transfers to our case (US large caps, daily bars, a 10,000 SEK
account, long-only, costs of 15-50 % of the account over three years).

## 1. Method and its limits

- **Code projects: only projects with at least 1,000 stars** (the user's rule). Star counts were read from the GitHub
  API on 2026-10-29, not from memory. Licence and last push date are recorded because they decide whether code may be
  copied, adapted or only read.
- **Papers: little-library.** `library_search` failed on every query on 2026-10-29 (`Error executing tool`), so the 302
  stored references could not be searched; `library_get` and `library_read` still work. New references were found
  with `library_research` (online search, asked per call). The library keeps **abstracts, not full texts**, for the
  new references, so every claim about a paper below is a claim about its abstract until the paper itself is read
  (plan step R1).
- **What was not verified:** results quoted from READMEs and abstracts were not reproduced. The Qlib benchmark numbers
  are for China's CSI300 index, a retail-heavy market where rank IC is 0.04-0.05; ours on US large caps was 0.00-0.03.
- Research queries run: 6. The broad query stored nothing (9 candidates offered at relevance 0.40-0.43, all off
  topic). "Empirical asset pricing via machine learning" was sent to the wrong sources (standards and CSS specifications)
  and stored nothing. A title query for Gu, Kelly and Xiu failed with an empty-PDF error, so that paper is **not** in
  the library yet. The three focused queries (costs, ranking, earnings text) stored 16 new references: 7, 7 and 2.

## 2. Code projects (at least 1,000 stars)

| Project | Stars | Licence | Last push | What it offers us | Use |
|---|---|---|---|---|---|
| `microsoft/qlib` | 49,792 | MIT | 2026-10-22 | The most complete open quant-ML pipeline. `Alpha158` (158 OHLCV features) and `Alpha360`; a benchmark of 20+ models; `DoubleEnsemble`; `TopkDropoutStrategy` (swap only the worst holdings, which is turnover control); cross-sectional z-scored labels | Read, port formulas with attribution, run as an **independent cross-check** of our harness (plan step A2) |
| `stefan-jansen/machine-learning-for-trading` | 21,623 | MIT | 2026-10-23 | The book's code: 27 chapters, nine case studies (US equities, ETFs, S&P 500 with options, FX, futures), chapters on transaction costs and risk management | Read for methodology and as a second opinion on feature and label choices |
| `skfolio/skfolio` | 2,557 | BSD-3-Clause | 2026-10-26 | Portfolio optimisation in the scikit-learn style: turnover and transaction-cost terms, hierarchical risk parity, `CombinatorialPurgedCV`, `WalkForward`, entropy pooling | Read; compare its combinatorial purged CV with our PBO; optional dependency of the lab only |
| `cvxgrp/cvxportfolio` | 1,316 | **GPL-3.0** | 2026-04-27 | Multi-period optimisation with an explicit cost model (`StocksTransactionCost`, `MultiPeriodOptimization`, `simulator`, forecast-error risk) from the Boyd group | Read only; **reimplement** from the papers, do not copy (licence) |
| `cvxpy/cvxpy` | 6,379 | Apache-2.0 | 2026-10-28 | The solver front end the optimisers above need | Optional lab dependency if a convex cost-aware layer is built |
| `shiyu-coder/Kronos` | 44,031 | MIT | 2026-04-13 | Foundation model for K-lines (OHLCV tokens), pre-trained on 45 exchanges; accepted at AAAI 2026 (`arXiv 2508.02739`); fine-tuning demo on Qlib A-share data with its own disclaimer that it is not a production system | Frozen feature extractor in **shadow mode only** (its pre-training data overlaps our test years) |
| `google-research/timesfm` | 34,817 | Apache-2.0 | 2026-09-29 | General time-series foundation model | Same as Kronos; lower prior (not trained for finance) |
| `amazon-science/chronos-forecasting` | 6,204 | Apache-2.0 | 2026-10-05 | General time-series foundation model | Same |
| `microsoft/RD-Agent` | 15,125 | MIT | 2026-10-21 | An agent loop that proposes and tests factors and models on Qlib | Read for method; its search over many candidates is exactly the multiple-testing risk our trial ledger exists for |
| `AI4Finance-Foundation/FinGPT` | 21,482 | MIT | 2026-10-25 | Open LLMs fine-tuned for financial sentiment and forecasting | Read how its sentiment benchmarks avoid leakage before any use (our N2/N3) |
| `AI4Finance-Foundation/FinRL` | 16,851 | MIT | 2026-10-25 | Deep-RL trading environments; the repository calls itself an educational and research framework | Already covered by our D4/D6 (policy gradient through the known costs); nothing new |
| `TauricResearch/TradingAgents` | 118,006 | Apache-2.0 | 2026-10-27 | Multi-agent LLM trading framework; v0.5 added point-in-time integrity and SEC EDGAR fundamentals "as filed"; states it is for research purposes | **Not a candidate.** Only its EDGAR data handling is of interest |
| `virattt/ai-hedge-fund` | 64,333 | MIT | 2026-10-27 | LLM-agent "hedge fund" proof of concept; states it is educational and not for real trading | **Not a candidate** |
| `PyPortfolio/PyPortfolioOpt` | 6,131 | MIT | 2026-07-07 | Classical mean-variance, Black-Litterman, HRP | Reference only; skfolio covers it |
| `polakowo/vectorbt` | 9,580 | NOASSERTION | 2026-10-28 | Fast vectorised backtesting | Reference only (licence unclear) |
| `mementum/backtrader` | 23,611 | GPL-3.0 | 2024-08-19 | Event-driven backtester | Not used (GPL, stale) |
| `kernc/backtesting.py` | 9,157 | AGPL-3.0 | 2026-08-05 | Small backtester | Not used (AGPL) |
| `hudson-and-thames/mlfinlab` | 4,958 | NOASSERTION | 2023-10-02 | López de Prado's methods (meta-labeling, purged CV) | Not used: stale, commercial licence. The ideas are taken from the book and papers |
| `quantopian/alphalens` | 4,484 | Apache-2.0 | 2024-02-12 | Factor analysis | Not used: stale; our `metrics.py` already computes rank IC and quantile spreads |

Excluded by the star rule: `stefan-jansen/alphalens-reloaded` (681 stars).

Extra tools I would add to the **lab environment only** (never to the product's `dependencies = []`), all above the
1,000-star line: `microsoft/LightGBM` (18,938, MIT), `dmlc/xgboost` (28,921, Apache-2.0), `catboost/catboost` (9,181,
Apache-2.0), `optuna/optuna` (14,987, MIT). The lab environment today has numpy 2.5.4, torch 2.14.1 (CPU only),
scikit-learn 1.9.1 and transformers 5.21.0; it has no lightgbm, no GPU, 24 threads and 13 GB of memory.

### What the Qlib benchmark shows (read from `examples/benchmarks/README.md`)

Mean of 20 seeds, CSI300, Alpha158 features, rank IC: DoubleEnsemble 0.0502, XGBoost 0.0505, LightGBM 0.0469, TRA 0.0540,
Linear 0.0472, MLP 0.0429, Transformer 0.0407, GRU 0.0428 (20 selected features). On the 360-feature raw-window set
the recurrent models do better than the trees (ALSTM 0.0599, GRU 0.0584, LightGBM 0.0499). So: (a) trees on rich
hand-built features are as good as any network; (b) an ensemble is the best entry; (c) the feature set matters more
than the architecture; (d) none of this was measured on US large caps after costs. Its label is the next-day return,
z-scored across stocks each day (`Ref($close,-2)/Ref($close,-1)-1`, `CSZScoreNorm`), and holdings are turned over by
`TopkDropout`.

## 3. Papers (little-library)

### 3.1 Stored on 2026-10-29 for this question (16, abstracts only)

**Trading under costs (decision making)**

| id | Reference | What the abstract says | Why it matters here |
|---|---|---|---|
| `t2mf` | Dynamic Trading with Predictable Returns and Transaction Costs (2013) | Closed-form optimal policy: "aim in front of the target" and "trade partially toward the aim"; slower-decaying signals get more weight | The theory behind partial adjustment and signal-decay-aware holding periods |
| `6kf1` | Multiperiod portfolio optimization with multiple risky assets and general transaction costs (2016) | For proportional costs the optimal policy has a no-trade region (a parallelogram); computable with one quadratic program; ignoring costs and being myopic costs a lot | A principled replacement for our fixed 3 % band |
| `q868` | Mean-variance portfolio optimization subject to transaction costs (1996) | Ignoring costs gives inefficient portfolios; with costs, the way expected returns are estimated made no significant difference | Supports "costs first, estimator second" |
| `gf0t` | Stochastic programming for multi-period portfolio optimisation with transaction costs (2012) | Scenario trees with a predictable-returns model | Background only |
| `we9a` | Dynamic portfolio rebalancing with explicit transaction costs (2026) | Cost-aware rebalancing beats periodic and threshold rules; cost level sets the best frequency and size of adjustment | Matches our planted-edge finding; a sweep design |
| `cayl` | Multi-period VaR-constrained portfolio optimisation via DC programming (2026, preprint) | Costs plus VaR constraints, with equal-weight and buy-and-hold benchmarks | Background; heavy method |
| `2f59` | Memetic algorithm for cardinality-constrained portfolio optimisation with transaction costs (2015) | No abstract stored | Cardinality limit (10 names) is our case; read the paper |

Offered but not stored: `7xfa` Parameter uncertainty in multiperiod portfolio optimization with transaction costs
(relevance 0.58).

**Ranking and learning to rank (training)**

| id | Reference | What the abstract says | Why it matters here |
|---|---|---|---|
| `eiak` | On Evaluating Loss Functions for Stock Ranking: transformer on S&P 500 (2025) | Systematic benchmark of pointwise, pairwise and listwise losses for daily ranking | Direct template for our loss experiment. The stored abstract gives no results |
| `crt7` | Long-short portfolio with a new listwise learn-to-rank loss (2021) | A listwise loss emphasising top and bottom of the list; China A-shares, 68 factors, 38 % annual return, Sharpe 2 (in-sample method, one market) | Candidate loss; treat the numbers as unverified |
| `rchl` | Rank-based approaches for portfolio stock selection (2022) | Ranking models beat the Saudi index over four years | Weak evidence (one small market) |
| `6wr0` | STHAN-SR, hypergraph attention, learning to rank (2021) | Ranking formulation beats forecasting networks on three markets | Needs relations between stocks (not available); the ranking formulation is the point |
| `72xt` | Return Prediction Weighted Scoring (2025) | Maps the ranking to weight tilts and then optimises with a covariance matrix; US, A-share and crypto | Candidate bridge between scores and weights |
| `zpqu` | Stock return prediction in the Tokyo Stock Exchange, Ridge, MLP, LightGBM (2026) | LightGBM best, Sharpe 0.039 daily spread, 0.028 after a forward-bias test; MLP and Ridge weak | Same ordering as our development window |
| `xn75` | Explainable ML with fundamental ratios, Borsa Istanbul (2026) | Trees on 13 ratios, quarterly top-k rebalancing beats equal weight; SHAP says profitability and valuation drive it | Evidence for fundamentals as new information; 20 firms, one emerging market |

Offered but not stored: `dtgl` Spatiotemporal hypergraph learning for stock return sequences (0.62).

**Earnings text (new information)**

| id | Reference | What the abstract says | Why it matters here |
|---|---|---|---|
| `rmih` | PEAD.txt: post-earnings drift using text (2022) | A text-based surprise measure (SUE.txt) gives a drift larger than the classic one, which is near zero in recent years | The strongest argument that text is worth testing, on earnings events only |
| `ty80` | Predictability of PEAD with textual and contextual factors of earnings calls (2023) | S&P 500 point-in-time constituents 2010-2022: contextual features add 53-354 bp per trade over earnings, fundamentals and technicals | Matches our universe; point-in-time membership is a method to copy |

Duplicates already in the library: `89et`, `nztb`, `3cfq`. Offered but not stored: `8ral` Linguistic complexity and PEAD
(0.61), `m6i6` Manager sentiment and stock returns (0.55), `r3u0` Linguistic passiveness in earnings calls (0.54), `b7kn`
Media abnormal tone and earnings announcements (0.53).

### 3.2 Already in the library from the earlier research (about 90 ids in `docs/plans/neural-decision-system.md`)

Reinforcement learning, sequence encoders, FinBERT and LLM text, memory, regimes, pre-training, backtest overfitting and
conformal prediction. Only three abstracts (`c3je`, `bz97`, `we9r`) had been read before; the rest are listed by title.
Most relevant to this plan: `pfvt`, `xjxs`, `whj9`, `bw40` (probability of backtest overfitting), `u8of` (cross-sectional
equity signals under overfitting diagnostics), `c3je` (reliable stock prediction), `dkfw` (a warning that pre-training must
beat a random start), `j2ie` and `8f2b` (cross-sectional neural return prediction), `247s` and `pgnk` (conformal
prediction).

## 4. What the whole body of research says, in short

1. **Costs and turnover are first-class** in every credible source: partial trading, no-trade regions and cost-aware
   optimisation (`t2mf`, `6kf1`, `we9a`, cvxportfolio, `TopkDropout`).
2. **Features and ensembles beat architectures** on daily cross-sections (Qlib benchmark, `zpqu`).
3. **Ranking losses and rank targets** are the standard for stock selection (`eiak`, `crt7`, Qlib's z-scored label); we
   use a rank target already but a pointwise loss.
4. **New information is the best-supported route to a real edge**: earnings text and fundamentals (`rmih`, `ty80`, `xn75`),
   consistent with our Stage 2 finding that headline sentiment alone adds nothing.
5. **Foundation models and LLM agents have no clean historical test**, because their training data overlaps our test
   years; they can only be judged forward in time.
6. **Most reported gains are from other markets** (China, Japan, Turkey, Saudi Arabia, crypto), on one path and often
   before costs. They are hypotheses for our data, not results.
