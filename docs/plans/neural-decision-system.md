# Plan: a neural decision system

Goal: replace the six hand-built models and the skill-gated vote with a small family of neural networks that read the
market and the news, remember what they learned, and output **value points** (an expected, risk-adjusted payoff per
company) from which trades follow, maximising profit after the real costs of a Swedish SEK account.

## Where we are, and the honest constraint

Ten years of daily bars over ~190 large caps, tested walk-forward on 37-42k samples: no model, and no single price
feature, beats the base rate (every cross-sectional IC |t| < 2). Large caps at a 5-day horizon from daily bars and
headlines are close to efficient. A neural network is a better *function approximator*, not a source of new
information, so the plan spends its first phase on **measuring honestly** and its later phases on **new information
and better decisions**, not just a bigger model. Every stage has a go/no-go gate; if the network stack cannot beat
the current baseline net of costs out of sample, it does not trade.

Reading that shaped the plan (little-library, abstracts only so far): *Reliable Stock Prediction: Data, Models,
Testing* (`c3je`; credible results depend on point-in-time data, simple strong baselines, temporal cross-validation,
backtest-overfitting tests and explicit costs, more than on novel architectures), *Reinforcement Learning for
Portfolio Management* (`bz97`; model-free agents shared across assets, helped by pre-training and data augmentation
because daily data is scarce, with transaction costs in the reward), *FinBERT* (`gxey`) and *Financial Sentiment
Analysis ... LLMs and FinBERT* (`we9r`; few-shot LLMs match a fine-tuned FinBERT on financial text). FinRL
(`h0j6`, offered, not stored) is the reference RL stock-trading environment.

## Architecture

```
 prices (60-250d window, per company) --> price encoder ---+
 market context (SPY, VIX, sector, rates) -> context enc. -+--> fusion --> company state (embedding)
 news (headlines, recency, source) -> text encoder -> pooling+                 |
                                                                               v
                   long-term memory: recurrent state + episodic retrieval + replay buffer
                                                                               |
                         +-----------------------------------------------------+
                         v                                v
                 value heads (supervised)         policy (RL, on frozen states)
          return quantiles 1/5/20d, volatility,      target weights per company,
          P(beat market), drawdown risk              net of cost and risk limits
                         \\                              /
                          +-------> gate + sizing <----+   (hard risk limits live outside the networks)
```

**1. Perception: networks that digest everything.**
- *Price encoder*: a small temporal convolution or Transformer over normalised multi-feature windows (returns,
  volatility-scaled, volume, gaps, position in the 52-week range) with the market series alongside; shared across all
  companies with a company/sector embedding, so 190 companies teach one model.
- *News encoder*: frozen FinBERT (or a small distilled finance encoder) embeds every headline once, stored in SQLite;
  a light attention layer pools a company's headlines with recency decay and source features. Frozen text embeddings
  keep this cheap and avoid overfitting on the ~months of news we have; the lexicon and naive Bayes remain as
  baselines.
- *Context*: SPY, VIX, rates, sector ETFs, earnings calendar (new data, see Phase 2).
- *Pre-training (self-supervised)* on all 10 years x all companies: predict several horizons at once (multi-task
  heads), masked/contrastive windows. This is where long history pays off, since the labels are free.

**2. Value points.** The heads output a *distribution* (quantiles of the 1/5/20-day excess return, volatility,
P(beat market)), not one number. Value points = expected net return after cost divided by risk. Uncertainty comes
from an ensemble of seeds plus quantile spread, and widens when the input looks unlike the training data.

**3. Long-term memory** (three layers, different time scales):
- *Slow weights*: continual learning. Retrain on a rolling plus replay-buffer mix of old regimes (2018 vol spike, 2020,
  2022 rates) with a weight-anchoring penalty, the generalisation of today's logistic prior.
- *Episodic memory*: retrieval over the embeddings of past situations and their outcomes (the learned successor of
  `pattern-knn`): "the 25 most similar past states and how they ended", fed back into the value heads.
- *Recurrent state per company*: a GRU/state-space state persisted in the database between rounds, so the model
  carries a company's recent story instead of recomputing from a window.
- Live outcomes (every decision and what followed) go in the replay buffer; lessons already stored become labels.

**4. Decision layer, in two stages.**
- *Stage A, supervised policy*: size positions from value points with a fractional-Kelly rule on the quantiles,
  subject to cost, minimum courtage, whole shares, position and cash limits. Simple, auditable, a strong baseline.
- *Stage B, reinforcement learning* (PPO or SAC, offline-first): the agent acts on the **frozen** company states, not
  raw data (RL on raw noisy prices overfits). Trained in a simulator that charges Avanza/Nordnet courtage and
  conversion fees, slippage, tax; reward = risk-adjusted excess return minus turnover and drawdown penalties;
  randomised start dates, block-bootstrap and synthetic series for augmentation; a seed ensemble, and the agent
  must **abstain** when its ensemble disagrees (the current skill gate, generalised).
- *Mixture*: several specialists (trend, mean reversion, news-event, risk-off) behind a gating network, with the
  classical models kept as ensemble members and fallback.

**5. Not neural, on purpose.** Restricted issuers, stop loss, max hold, max position, cash buffer, the trade pacing
and the dry-run-only mode stay as code outside the networks.

## Phases and gates

| Phase | Work | Gate to continue |
|---|---|---|
| 0 Harness | Optional `nn` extra (numpy + PyTorch CPU, a separate training process; the stdlib-only rule stays for the server core). Point-in-time panel dataset. Purged walk-forward CV with embargo, cost-aware backtester with the real broker costs, deflated Sharpe / probability of backtest overfitting, baselines (equal weight, SPY, current models, a gradient-boosted tree). | Harness reproduces today's "no skill" verdict and passes a planted-signal test (a synthetic edge it must find, a pure-noise series it must reject). |
| 1 Price network | Encoder + multi-horizon heads, pre-trained, cross-sectional rank targets (market drift removed) and 20-day horizon alongside 5-day. | Out-of-sample IC or Brier skill > 0 with t > 2 after purging, net of cost; otherwise stop and go to Phase 2 for information, not capacity. |
| 2 New information | Earnings dates and surprises, VIX/rates/sector context, more universe (mid caps, ETFs), intraday bars where free, FinBERT embeddings stored for all headlines. | Each source shows incremental out-of-sample gain over Phase 1 (ablation), or it is dropped. |
| 3 Memory | Replay buffer, episodic retrieval, persisted recurrent state, continual retraining schedule. | Better walk-forward score than retraining from scratch, and no collapse on old regimes. |
| 4 Decision | Stage A sizing, then Stage B RL in the cost-accurate simulator. | Beats Stage A and equal weight on a final untouched period, deflated Sharpe > 0, drawdown inside limit. |
| 5 Shadow, then live paper | Run beside the current trader in the dry run for 8+ weeks; compare matured decisions net of cost; drift monitors; model registry with rollback. | Promoted only if it beats the incumbent live, not just in backtest. |

## Efficiency

Small models (under a few million parameters), embeddings cached, so a 15-minute round is inference on cached states
(milliseconds on CPU). Training runs as a nightly job on the GPU only when `workshop_generation_gpu` shows it idle.
Parameter sharing across companies and self-supervised pre-training are the answer to scarce data: ~475k daily
samples are heavily autocorrelated, so the effective sample size is far smaller than it looks.

## Risks

- *Nothing to find*: if efficient-market limits hold at this horizon and universe, the right result is the gate
  saying so; profit then needs different information (Phase 2) or a different horizon, not more layers.
- *Backtest overfitting*: the main hazard of RL; hence the untouched final period, deflated Sharpe, and shadow mode.
- *Dependencies and memory*: PyTorch is heavy; keep it in a separate worker so the server stays small.
- *Employer restriction*: unchanged; the restricted list filters every stage, including embeddings and universe.

## Decisions for the owner

1. Allow numpy/PyTorch as an optional extra (recommended), or stay stdlib-only and hand-write small networks (slow,
   limited to tiny MLPs/GRUs).
2. Use the GPU nightly when idle (recommended) or CPU only.
3. Start with Phase 0 and 1 now (recommended), since they decide whether the rest is worth building.

## References (little-library ids)

Found by research requests on 2026-10-28; stored in little-library. Only the abstracts of `c3je`, `bz97` and `we9r`
have been read so far; the rest are listed by title and relevance and still to be read before a design depends on them.

**Reinforcement-learning agents for trading and portfolios (decision layer, Phase 4)**
`xl1r` Deep RL for automated stock trading · `uf8i` Application of deep RL on automated stock trading · `sgf0` Deep
RL for automated stock trading: an ensemble strategy · `ycm8` FinRL library · `vu6e` Applying deep RL in automated
stock trading · `5l67` Deep-RL decision support system for automated trading · `vx1z` Synchronous deep RL for
multi-stock trading · `wmmn` Empirical analysis of automated stock trading using deep RL · `bz97` RL for portfolio
management · `xydv` General framework on enhancing portfolio management with RL · `jm39` Risk-aware deep RL in
portfolio optimisation · `vwgr` LLMs with deep RL for portfolio optimisation · `wsjp` Offline RL for automated stock
trading · `b0mx` Policy gradient methods in deep RL · `bh1h`, `hbhg` Deep RL portfolio management with transaction
costs and efficiency bounds (cryptocurrency)

**Generalisation and overfitting of RL agents (augmentation, Phase 4)**
`nc8o` Generalisation in RL trading with a generative adversarial market model · `uizw` Boosting deep RL agents with
generative data augmentation · `dblw` Policy-aware adversarial data augmentation · `7ylc` Selective noise injection
and information bottleneck

**Sequence encoders for prices (price network, Phase 1)**
`6zi1` LSTM vs transformer for stock forecasting · `0079` BiLSTM-Transformer hybrid · `1414` Hybrid TCN-Transformer
with multi-head attention · `fmno` Transformers for long-term, LSTM for short-term forecasting · `4kx0` Temporal
Fusion Transformers (multi-horizon, interpretable) · `rtu3` Temporal Fusion Transformer for Bitcoin · `z4wd` Mamba and
ensemble learning for stock movement · `n6h2` Attention-Mamba for stock prediction · `tsxr` Graph-Mamba for stock
prediction · `rn99` Stock price prediction using Mamba · `wa9b` T-Mamba hybrid framework

**Cross-sectional return prediction (rank targets, Phase 1)**
`j2ie` Neural networks to improve cross-sectional momentum · `te7d` Spatio-temporal momentum (time-series and
cross-sectional jointly) · `8f2b` Machine learning for cross-sectional return predictability, global markets · `iljd`
Supply-chain propagation of textual signals, LLM embeddings and cross-sectional predictability

**News, sentiment and text (news network, Phases 1-2)**
`gxey` FinBERT · `d6vi` FinBERT sentiment dynamics · `3mq7` FinBERT in market prediction · `we9r` LLMs and FinBERT for
financial sentiment · `x7tc` Sentiment analysis and transformers in price forecasting · `7dxy` LSTM with news
sentiment · `h2c4` PCA, EMD and LSTM with sentiment · `ndzk` FinBERT-LSTM · `8a1i` LSTM with news sentiments · `f6z2`
Stock prediction from news text · `bu2p` Returns from news sentiment with NLP and deep learning · `fvbe` LSTM with
sentiment analysis · `s9l8` Fine-tuning LLMs for return prediction from newsflow · `efwd` Quantitative factors plus
LLM newsflow representations · `7wn2` Can ChatGPT forecast stock price movements? · `77j1` Stock features and global
information via LLMs · `qln4` Faithful LLM narratives of return predictions

**Earnings text (new information, Phase 2)**
`89et` Predictive power of earnings conference calls · `3cfq` Earnings call and stock price movement · `nztb`
Corporate sentiment from earnings call transcripts

**Memory (Phase 3)**
`ad43` RAMTime, retrieval-augmented memory for forecasting · `22y9`, `sgjk` Retrieval-augmented time series
forecasting · `g0i1` Retrieval-augmented diffusion forecasting · `v2hs` Spectral retrieval-augmented forecasting ·
`wxnt` Stationarity-aware retrieval-augmented forecasting · `ha0o` Continual learning methods on financial and
industrial time series · `15ob` Dynamic neuroplastic networks against catastrophic forgetting in financial decisions ·
`bie2` Gradient-based mitigation of catastrophic forgetting, review

**Regimes (gating and mixture, Phase 4)**
`yjwv` CRAT-Net, contrastive regime-aware multi-asset regime detection · `4356` Regime detection with k-means and
hidden Markov models · `p7ge` Adaptive HMM with anomaly states

**Pre-training (Phase 1)**
`dkfw` Random initialisation as a baseline for time-series foundation models (a warning: check that pre-training
beats a random start) · `45u4` UTICA multi-objective self-distillation pre-training

**Honest evaluation (Phase 0)**
`c3je` Reliable stock prediction: data, models, testing · `pfvt` The probability of backtest overfitting · `bw40`
pbo package · `whj9` Mathematical appendices to the probability of backtest overfitting · `xjxs` Bayesian approach to
measuring backtest overfitting · `u8of` Cross-sectional equity signals under backtest-overfitting diagnostics

**Uncertainty and abstaining (gate and sizing, Phases 4-5)**
`247s` Conformal prediction for financial time series under asset-pricing models · `pgnk` Adaptive conformal
predictions for time series · `ut2w` Uncertainty-aware financial forecasting with conformal prediction · `v43u`
Conformal prediction sets for time series · `8ois` Adaptive multi-scale forecasting with gate-localised conformal
prediction

Also stored by the same searches but not relevant to this plan (satellite, earthquake, GNSS, well-performance, GDP,
medical and game-agent papers: `ypvm`, `wrcn`, `gc8n`, `ygs4`, `hqv7`, `vfi6`, `efld`, `up1v`, `ph68`, `meup`).
