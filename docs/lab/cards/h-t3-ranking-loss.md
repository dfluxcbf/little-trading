# Card: H-T3, ranking losses in place of the pointwise loss

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-t3-ranking-loss` (`src/little_trading/lab/services/h_t3_ranking_loss.py`),
losses and the scorer in `src/little_trading/lab/losses_t3.py`. Sources eiak, crt7, rchl (the losses are the standard
ones, written from their definitions; no code copied). Check it with
`little-trading lab card check docs/lab/cards/h-t3-ranking-loss.md`.

Hypothesis: A loss that looks at the order of the scores inside a session (pairwise RankNet, listwise ListNet, or the
  per-day IC itself) gives the same MLP a higher daily rank IC than our pointwise squared error on the rank label, because
  the trades only use that order and the pointwise loss spends capacity on the level and scale of each name's target.
Change: The training loss only, computed per session (the names tradable on one day) and averaged over sessions: RankNet
  (logistic loss on the score difference of every pair with a higher target), ListNet (cross-entropy between softmax of the
  per-session z-scored targets and softmax of the scores, tau 1) or the negative Pearson correlation of scores and targets
  per session. Same MLP (the architecture of alternatives.MLP: two GELU hidden layers of 64, dropout 0.1, AdamW lr 2e-3,
  weight decay 1e-3, 8 epochs, early stopping on the last 15 % of training sessions purged by 5 sessions), same 18 ranked
  features, same 5-session rank label, every third session up to the purged train_end with whole sessions kept together,
  same purged walk-forward folds (refit every 126 sessions).
Family: mlp
Baseline: mlp-pointwise, the same MLP trained with the per-session mean squared error on the 5-session rank label (our
  current loss), seeds averaged the same way.
Metric: Paired daily rank IC difference (variant minus baseline) against the 5-session relative return, t-statistic with
  the overlap correction of 5 sessions, reported over the window and on the two development spans.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 3
Window: development 2019-2023
Leak test: tests/unit/test_lab_t_loss.py::test_walk_forward_scores_of_every_loss_never_depend_on_later_bars (future
  perturbation of the raw OHLCV arrays, cuts inside the sample, every loss), plus
  tests/unit/test_lab_t_loss.py::test_numpy_loss_gradients_match_finite_differences (the losses themselves)
Notes: Three variants: mlp-ranknet, mlp-listnet, mlp-ic (5 seeds, 0 to 4, are one variant each); the pointwise baseline is
  not counted. The Transformer named in the plan is not wired: alternatives.SeqModel trains its own multi-head pointwise
  loss with pre-training, and swapping its loss is a separate change; if the MLP shows a loss effect, the Transformer is
  the next card. The torch MLP path is not run by the unit tests (no torch in the test environment): the forward losses it
  trains on are the same code the numpy tests check (an array-operations shim), but its training loop is exercised only in
  the nn environment. A linear numpy scorer (model "linear") runs the same trial for the tests. RankNet uses every pair in a
  session (about 18,000 for 190 names), no sampling. The IC loss is scale-free; a small L2 (linear) or the weight decay
  (MLP) keeps the scale bounded. Prior: low to medium. K1 says an IC gain must be large before a top-10 layer pays at
  10,000 SEK, so even a pass is unlikely to change trading on its own.
