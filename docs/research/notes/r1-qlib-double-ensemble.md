# R1 note: Qlib `DEnsembleModel` (DoubleEnsemble: sample re-weighting and feature selection)

## Source

Microsoft Qlib, `qlib/contrib/model/double_ensemble.py` (class `DEnsembleModel`), branch `main`, 277 lines, sha256
`699cfd4bf09bae95fc9d95994551d0660d62ebcebde88c7358857aca95641669`. Fetched 2026-10-30 with curl from
`https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/model/double_ensemble.py` into its own empty directory
under the session scratch area. Commit SHA not recorded (GitHub API refused by the session proxy, HTTP 403). Licence: MIT,
headed "Copyright (c) Microsoft Corporation. / Licensed under the MIT License." The paper behind it (Zhang et al.,
DoubleEnsemble) was not read for this note.

## Claim

The plan's question: what do the sample re-weighting and feature selection do, and can they be ported?

**Training loop (`fit`).** `num_models` (default 6) LightGBM sub-models are trained one after another on the training
segment. After sub-model k (except the last): its per-sample loss curve over boosting iterations is computed
(`retrieve_loss_curve`, by predicting one tree at a time, `start_iteration=i_tree, num_iteration=1`), the ensemble so far
predicts the training set, and its per-sample loss is `loss_values` (`get_loss` supports only `"mse"`: `(label - pred) ** 2`).
Then SR sets the next sub-model's sample weights and FS its feature subset. Prediction is the `sub_weights`-weighted mean of
the sub-models, each on its own feature subset (`predict`).

**Sample re-weighting (`sample_reweight`).**

- `loss_curve_norm = loss_curve.rank(axis=0, pct=True)`: per iteration, each sample's loss as a percentile across samples.
- `l_start`, `l_end`: means of that over the first and last 10 % of iterations (`part = np.maximum(int(T * 0.1), 1)`).
- `h1 = (-loss_values).rank(pct=True)`: high for samples the current ensemble fits well.
- `h2 = (l_end / l_start).rank(pct=True)`: high for samples whose relative loss rose during training.
- `h = alpha1 * h1 + alpha2 * h2` (both default 1.0), cut into `bins_sr` (10) equal-width bins; every sample in bin b gets
  `1.0 / (self.decay**k_th * h_avg[b] + 0.1)`.

So samples the ensemble fits badly (low h1) and samples whose loss fell relative to the others (low h2) get the largest
weights, by up to about a factor of (max h + 0.1) / (min h + 0.1) across bins; `decay ** k_th` shrinks the spread as more
sub-models are added. Note: `decay` defaults to `None` in the constructor, and `None**k_th` raises a TypeError on the first
SR step, so any working configuration must set it (the value used in Qlib's benchmark configs was not read here).

**Feature selection (`feature_selection`).** For each feature: permute that column of the *training* data
(`np.random.permutation`), predict with all sub-models so far (equally weighted, `/ M`), and compute
`g = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)`: a t-like score of how much the loss
rises without the feature. g-values are cut into `bins_fs` (5) equal-width bins; from the best bin down,
`ceil(sample_ratios[i] * size)` features are drawn at random with `sample_ratios` default `[0.8, 0.7, 0.6, 0.5, 0.4]`. The
result is `pd.Index(set(res_feat))`, so the column order is set order (the same index is reused at predict time, so this is
consistent).

**Points that matter for a port (read from the code):**

- Everything is computed on the training segment; the validation set is only passed to LightGBM for logging and optional
  early stopping. Permutation importance on training data rewards features the trees fit, including overfit ones.
- Ranks in SR are pooled over all rows, not per date, so dates with large cross-sectional dispersion dominate the losses.
- FS uses `np.random` without a seed of its own; reproducibility depends on the global seed.
- FS's permuted prediction weights sub-models equally (`/ M`) while `loss_values` uses `sub_weights`; identical only with
  the default equal weights.
- Only `base_model="gbm"` is implemented (`raise ValueError("not implemented yet")` otherwise).

**Can it be ported?** Yes: the two steps are about 80 lines of pandas over a loss matrix, need only per-iteration
predictions (LightGBM per-tree predictions, or scikit-learn `staged_predict`), and are MIT-licensed. Our repository already
has a numpy port (`src/little_trading/lab/ensemble_t4.py`) whose docstring states the same rules, sets `decay=0.5`, adds a
mean-1 weight normalisation, and leaves FS off in the registered H-T4 variant. Cost on our data: the loss curve is N rows x
T trees of predictions per sub-model, and FS is F x M full predictions per round (158 features x up to 5 sub-models);
cheap for trees on about 200 names, slow if the base model were a network.

## Evidence quality

High for what the code does (full file read). Nothing in this file shows the method helps; the performance claim is in the
benchmark README (CSI300 Alpha158 rank IC 0.0502 against LightGBM's 0.0469 and XGBoost's 0.0505; see
`r1-qlib-benchmarks-readme.md`), on Chinese A-shares, gross of our costs, and with a suspicious CSI500 row. The "why" (h1
targets hard samples, h2 targets learnable ones) is the paper's framing as reflected in the code, not verified here.

## Transfer to US large caps and to our costs

SR and FS change how a model is trained, not what information it sees. On CSI300 the README's DoubleEnsemble rank IC is
within 0.004 of the plain trees; even if a similar gain transferred, it would move our best real rank IC (0.034) by a few
thousandths, against K1's net bar of 0.15 (50,000 SEK) to 0.30 (10,000 SEK). Up-weighting high-loss samples on noisy
daily labels may also up-weight noise; on our US large caps, where our measured rank IC (0.00-0.03) is below the
benchmark's 0.04-0.05, the signal-to-noise ratio is lower and that risk is higher. Its realistic value for us is variance reduction (several sub-models), which is
the plan's stated reason for H-T4.

## What we would test

H-T4, service `h-t4-ensemble`, variant "double-ensemble-sr" (already registered: SR on, FS off), paired daily rank IC
against single trees on development. Reading-driven checks for that service, not new variants: (1) set `decay` explicitly
and record it; (2) confirm the port reproduces Qlib's SR weights on a small fixed loss matrix (an X1-style unit check;
no Qlib code is run, the expected numbers come from the formulas above); (3) report whether SR beats the "trees-bagged"
variant, because a gain from SR that bagging also gives is variance reduction, not re-weighting. FS would be a fourth
variant needing its own card.

## Verdict

**adopt-to-test** (SR, inside H-T4 as registered); FS **read-only** until a card registers it.

## How it was read

`qlib/contrib/model/double_ensemble.py`, branch `main`, fetched 2026-10-30 from raw.githubusercontent.com; **full file
read**. The DoubleEnsemble paper and the benchmark YAML configs (which set `decay` and the other hyper-parameters) were not
read. Nothing was executed.
