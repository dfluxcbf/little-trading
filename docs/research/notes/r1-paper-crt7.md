# R1 note: Long-short portfolio with a new listwise learn-to-rank loss (2021) (`crt7`)

## Source

Xin Zhang, Lan Wu and Zhixue Chen, "Constructing long-short stock portfolio with a new listwise learn-to-rank algorithm",
arXiv 2104.12484v1 (q-fin.PM), submitted 26 April 2021; the title page is dated 7 December 2020. Library id `crt7`
(little-library). Plan question (`docs/plans/improvements-lab.md` section 6): Which loss won, by how much, and was the test
clean? Code named in the paper's section 4: `https://github.com/TCtobychen/ListFold`.

## How it was read

**Full text read on 2026-11-01**: all 20 PDF pages, including the appendix. The proofs were read but not checked; nothing here
depends on them.

- Text used: `https://arxiv.org/pdf/2104.12484v1`, fetched with curl into a new scratch directory (sha256
  `ef07225b3fbc53225d39e146a8983cb8bdd2de807d6ca88e9b34e3591fb87dcb`), text extracted with pypdf. All paper numbers below come from
  this copy.
- Pages 12 to 15 were also rendered as images and read by eye, to check the columns of Tables 2, 3 and 4 and to see Figures 3,
  4 and 5. **Figure 5's numbers were transcribed from an image. Values from Figures 3 and 4 are readings off a chart and are
  approximate.** Figures 1 and 2 were not viewed.
- Not used: the library copy of `crt7`, because `library_read` returned only the abstract.
- Code, as a check on the paper: `process_rolling.py`, `train.py`, `torch_model.py`, `back_test.py` and `first_process.py` read in
  full, branch `main`, commit SHA not recorded (sha256 of the first four: `cb2c8b72...`, `e800c8c2...`, `92e68a4f...`, `74260940...`).
  `train_mlp.py` was read in part. `torch_model_mlp.py` and `back_test_mlp.py` were only searched. The README was seen only through
  a summarising fetch tool. **Nothing was run and the data files were not downloaded.** The code may not be exactly what produced
  the tables: the README names a training script, `train_20200111_rolling_torch.py`, that is not in the repository.
- Peer-review status was not checked. The arXiv page, also seen through that fetch tool, showed no journal reference.
- Checks marked "my arithmetic" were run on the paper's numbers. No model or backtest was run.

## Claim

**Which loss won.** ListFold-exp, the authors' own loss with an exponential transform. It builds the ranking as a sequence of
long-short pairs, so the top and the bottom of the list count equally (equations 3 and 4). All numbers below are for long 8, short
the bottom 8 or the average of 80, equal weight, rebalanced weekly.

| Loss | Long-short: return over rf | Sharpe | Max drawdown | Turnover | Rank IC | Short the average: return over rf | Sharpe |
|---|---|---|---|---|---|---|---|
| **ListFold-exp** | **0.38** | **2.01** | 0.14 | 0.48 | **0.079** | **0.08** | **0.71** |
| ListFold-sgm | 0.26 | 1.27 | 0.25 | 0.45 | 0.077 | 0.06 | 0.50 |
| List2MLE (two ListMLE fits) | 0.26 | 1.29 | 0.21 | 0.46 | not reported | not run | not run |
| ListMLE | 0.20 | 0.91 | 0.23 | 0.45 | 0.077 | -0.06 | -0.53 |
| MLP (squared error on returns) | 0.16 | 0.72 | 0.28 | 0.39 | 0.055 | -0.19 | -1.79 |

Table 2 gives everything except the IC, which is Table 3. ListMLE-rvs, with reversed labels, has IC 0.057 (Table 3).

**By how much.**

- On return over rf, 0.12 above the runner-up and 0.22 above MLP. On Sharpe, 0.72 above the runner-up (List2MLE) and 1.29 above MLP.
- On IC, 0.002 above ListMLE and ListFold-sgm, and 0.024 above MLP (my arithmetic).
- The only significance test in the paper compares ListFold-exp with List2MLE on the weekly difference: **t = 1.45** "under normal
  assumption" (section 4.3.1). That is short of the usual 2, and the surrounding text is about the no-fee results. The conclusion
  still says the losses "have significant advantage over the others".

**Was the test clean?** No. The time split is sound, but the universe was chosen with sight of the test period, and the settings have
no validation set behind them. The paper's batch-size check was run on the test period; it does not say how 32 was first chosen. See
the next section.

The catalog's "in-sample method" (`docs/research/improvements-catalog.md`) is not the right description. The test is a real
walk-forward. The problems are the ones below.

## Evidence quality

Low as evidence that this loss is better. Good as a definition of the loss. The design:

- **Market and period:** China A-shares. 631 weeks, 3,712 stocks, 68 factors, 2006-12-29 to 2019-04-19, from the Wind database
  (section 4.1). **Only 80 stocks are used.**
- **Out of sample:** yes, rolling. Train on 300 weeks, test on the next 16, for 320 test weeks from 2012-11-30 to 2019-02-01.
- **Model:** the same 4-layer ReLU network for every loss, 1,000 mini-batches, batch size 32 (sections 4.1 and 4.2). Footnote 2:
  the MLP drops the last ReLU.
- **Gross or net:** Figures 3 and 5 are stated to be without fees. Table 4 does not say, but its batch-32 row matches Figure 5's
  k = 8 row, so I read it as gross. Table 2 is introduced with a cost, so I read it as net, although the paper does not use the
  word. The Table 3 ICs are not stated to be test-period figures, though the section implies it.
- **Cost assumption:** "the transaction cost per trade 30 bps in total (such as tax, spread crossing and getting short)", and rf 3 %.
  Turnover is the share of names that differ between two consecutive weeks.
- **Variants tried:** 5 losses and 2 short legs, 9 portfolios in Table 2 (List2MLE has no short-the-average form). Five batch
  sizes across 5 loss columns, 25 cells in Table 4. Forty values of k in Figure 5 (the text says 1 to 80; rows stop at 40). No
  multiple-testing correction. The code shows a further choice the paper does not report (item 3 below).
- **Leakage control:** the paper says only "we use no future information", and explains it by treating each week as an independent
  sample. That is not a control. No purge, no leak test. The words validation, seed, survivorship, delisting and suspension do
  not appear.

**What holds up (from the code; the paper does not say these).**

- Min-max scaling is fitted per stock on the training window. In the test window it only widens with data seen so far, so it
  uses no future values.
- A column list named `no_cheating_feature` drops `period_return` and three others from the 72 columns.
- The label is the next week's log return and the windows abut. One-week labels do not overlap, so no purge is needed.
- k = 8 is the natural 10 % of 80 and not ListFold-exp's peak in Figure 5, which is k = 5 at 112. (The highest cell in the figure is
  List2MLE at k = 2, 121.)

**What fails, each of which the question "was it clean?" turns on.**

1. **The universe was chosen once, using the whole sample.** The paper keeps stocks with under 0.1 % missing values. If that share is per
   stock over the whole span, it is under one week's worth of data out of 631 (my arithmetic). Described that way, it cannot be
   known before 2019. Every stock must have traded
   nearly unbroken through the test period, so names that delisted or had long halts are gone. The code confirms a fixed list used
   in all windows: 85 hard-coded indices, of which positions 50 to 54 are dropped with no reason given. The paper describes no such
   step. The effect on a long-short spread cannot be signed from this paper. Our own universe has a survivorship caveat too (K1 Limits).
2. **The batch size that makes the win was seen on the test period.** There is no validation set in the paper or the code: training
   stops at batch 1,000 and that model is used. The reported batch size, 32, is the best of the five for ListFold-exp. Table 4,
   gross, average weekly return of long-short 8, in bp:

   | Batch size | ListFold-exp | ListFold-sgm | ListMLE | ListMLE-rvs | List2MLE |
   |---|---|---|---|---|---|
   | 8 | 57.1 | 55.9 | **85.6** | 78.3 | 78.4 |
   | 16 | 72.6 | 79.9 | **82.3** | 71.0 | 73.5 |
   | 32 (reported) | **108.1** | 88.7 | 74.5 | 90.8 | 85.9 |
   | 64 | **99.3** | 76.2 | 88.1 | 70.3 | 78.3 |
   | 128 | **84.0** | 82.0 | 76.3 | 77.6 | 77.4 |
   | Mean (my arithmetic) | 84.2 | 76.5 | 81.4 | 77.6 | 78.7 |

   - ListFold-exp wins three rows of five. At batch size 8 it is fourth of five, 28.5 bp behind ListMLE.
   - At the reported setting it leads ListMLE by 33.6 bp a week. Averaged over the five settings the lead is 2.9 bp (my arithmetic).
   - The spread across settings for the winner alone is 51.0 bp, more than twice its 22.2 bp lead over List2MLE at batch size 32.
   - Batch size 32 is ListMLE's lowest value of the five, so the headline comparison is also its weakest setting.
   - Batch size 32 is also the best of the five for ListFold-sgm, ListMLE-rvs and List2MLE, so it does not suit the winner alone. ListMLE
     is the widest case: averaged over settings, the lead over the other three is still 5.5 to 7.7 bp (my arithmetic).
   - The MLP is not in this table, so the MLP comparison has no such check.
3. **The winner has its own learning rate in the code.** `torch_model.py` sets 5e-5 for ListFold-exp and 1e-4 for ListFold-sgm and ListMLE;
   `train_mlp.py` uses 1e-4. The paper says "same neural network, same data and same training procedure", and it does not report
   how the rate was chosen. With no validation set, this may be another look at the test. The paper gives no way to tell.
4. **One run per cell, as far as can be seen.** Neither paper nor code reports repeated runs, and the code sets no seed. Our own
   neural runs vary a lot between seeds (plan, O4), and Table 4 shows the same kind of swing.
5. **The winner changes with k (Figure 5, gross).** ListFold-exp is best at k = 1 and k = 5 to 13. List2MLE beats it at k = 2, 3 and 4
   (121, 115 and 105 bp against 80, 96 and 101). ListFold-sgm beats it from k = 14 to 40 (60 bp against 48 at k = 40). At k = 10, nearest our
   top-10: ListFold-exp 101, ListFold-sgm 86, List2MLE 82, ListMLE 78, MLP 68. The authors do say ListFold-sgm suits wide books, which
   is what its theorem predicts.
6. **The cost charged looks low, and it is not in the code.** No published script has a cost step. Gross return from Table 4 and Figure 5,
   less a flat 30 bp a week and rf, gives 0.376, 0.275, 0.201, 0.261 and 0.157 against Table 2's 0.38, 0.26, 0.20, 0.26 and 0.16
   (ListFold-exp, ListFold-sgm, ListMLE, List2MLE, MLP; my arithmetic). A charge of 30 bp times turnover times two legs fits worse, up
   to 0.032 off. So Table 2 seems to charge about 15.6 points a year.
   The code holds $1 in each leg, so turnover of 0.48 trades about 1.92 of notional a week. That is about 16 bp per unit traded, and
   the paper says this also pays for tax and borrow. Taken as 30 bp on every buy and sell in both legs, the cost would be 57.6 bp a week
   and return over rf would be about 0.23 (my arithmetic).
7. **Execution is not modelled.** As far as the code shows, the return runs from the price on the signal's bar to the next bar
   (`Y_cl`), with no lag stated. The data were not opened, so this alignment is not confirmed. In the code a missing
   return becomes 0, and `trade_status` is dropped as a feature and not used to block a trade. The paper has no rule for halts or limit
   moves. Shorting single names is assumed; the paper itself offers the short-the-average panel in case one must short an index future.
8. **Return is on one leg of capital.** With $1 long and $1 short, 0.38 on $1 would be 0.19 on gross exposure (my arithmetic). Sharpe is
   unchanged. It is also a sum of mean log returns on a fixed $1 and is not compounded.
9. **Chance.** The iid standard error on a Sharpe of 2.01 over 6.15 years is about 0.70 (my arithmetic). The short-the-average
   Sharpe of 0.71 has about 0.45, so a ratio near 1.6. The plain t on the mean, Sharpe times the square root of years, is about 1.8
   (my arithmetic). None of these is in the paper.
10. **Smaller mismatches.**
    - The conclusion says 39 % and Sharpe 2.07. Table 2 and the abstract say 38 % and 2.
    - The abstract calls 38 % an "annual return", while Table 2 labels 0.38 as return over rf.
    - The text gives the network as 68, 136, 272, 34, 1. The code has 68, 272, 136, 34, 1.
    - The paper says each algorithm views 1,000 mini-batches. `train_mlp.py` defaults to 4,000, while `back_test_mlp.py` loads the model
      saved at batch 1,000, so those two defaults do not fit together and the code does not settle what the MLP run used.
    - In the code ListFold-sgm pairs rank i with rank i + 40. Equation 8 pairs rank i with rank 81 - i.
    - The code applies a ReLU to the final score, so names can tie at exactly 0, which would leave their order to the array position.
      I did not measure whether this happens.
    - The paper says IC "coincides with the pnl performance fairly well", yet ListFold-sgm and ListMLE share an IC of 0.077 with Sharpe
      1.27 and 0.91, and 0.002 more IC goes with Sharpe 2.01.
    - The theory is partly open: Theorem 3.2 proves consistency with the 0-1 loss only when the top and bottom halves are already
      split correctly. Appendix A leaves the general case to future study.

## Transfer to US large caps and to our costs

Very little of the result transfers. The market, the universe (80 against our 208), the features (68 with fundamentals against 18
ranked price features) and the book (long-short against long only) all differ. The horizon, one week, is close to our 5 sessions.

**Against K1 (`docs/research/k1-edge-needed.md`).** Net of costs a top-10 layer needs a rank IC of 0.15 at 50,000 SEK and 0.30 at
10,000 SEK. Our best real IC is 0.034.

- **The levels.** Even at face value, the winner's IC of 0.079 is about half of 0.15 and about a quarter of 0.30. It comes from another
  market and a flattered universe. Its MLP baseline of 0.055 is already 1.6 times our best (my arithmetic), so most of that level is
  not the loss.
- **The effect of the loss is what could transfer, and it is small.** Against MLP the gain is 0.024, which is 21 % of the gap from 0.034 to
  0.15 and 9 % of the gap to 0.30. Adding it to our best gives 0.058, still under K1's 0.08 for the realistic fee cuts at 50,000 SEK.
  Even that overstates it: the paper's MLP is squared error on raw returns, while ours is squared error on ranks (H-T3 card). The nearer
  comparison is ListMLE, another rank loss, and that gain is 0.002, which is 2 % of the gap to 0.15 and under 1 % of the gap to 0.30
  (all my arithmetic).
- **The 38 points need a short book.** Our layers are long only. The nearest panel to K1's equal-weight bar is short the average: 0.08
  over rf and Sharpe 0.71. Gross, the short leg supplies a little over half of the 108 bp a week (about 58 bp; Figure 4 gives about 50 bp
  for the long leg, a chart reading). The same flat charge then falls on the smaller panel, leaving only 0.08. There is one point in favour of
  long-only use: the gap to MLP is wider here (0.27 against 0.22), and the authors say almost all the gain over MLP comes from the long leg.
  The gap to ListMLE is narrower here (0.14 against 0.18), so this is not a clean signal. Also, ListFold-exp's NDCG@8 (0.229) is below ListFold-sgm's (0.234) and ListMLE-rvs's (0.244) (Table 3).
- **Our costs would take it.** The long leg over the average earns about 26 points a year gross (Figure 4 ends near 2.6 on $1 a week;
  a chart reading, and 0.08 + 0.03 + 0.156 gives the same). At turnover 0.48 the book is replaced about 25 times a year. In
  `broker.py` terms a round trip costs about 1.1 % at 50,000 SEK and about 2.6 % at 10,000 SEK with ten names (the 1 USD floor is about 1 %
  of 1,000 SEK). That is about 27.5 points a year at 50,000 SEK and about 65 at 10,000 SEK (my arithmetic). So at the paper's own gross edge and
  turnover, costs take all of it at 50,000 SEK and two and a half times it at 10,000 SEK. This agrees with K1. The paper also expects
  turnover to rise "significantly" with a larger stock pool, and ours is larger.

## What we would test

**H-T3 as registered does not test this paper's loss.** Its variants are RankNet, ListNet and the IC loss
(`docs/lab/cards/h-t3-ranking-loss.md`). The plan's wording "listwise (ListNet, the `crt7` loss)" can be read as two losses or as one. As one it would be wrong: the word
ListNet appears nowhere in this paper, which generalises ListMLE. As two, the card dropped the `crt7` loss. Either way, a result from
H-T3 says nothing about ListFold. H-T3 should run as it stands.

If the loss is wanted, it needs a new card, not a change to the registered grid, and it should come after H-I1 and H-I2, as K1 advises.
The card should say:

- Variants `mlp-listfold-exp` and `mlp-listmle`, baseline `mlp-pointwise`, in the card's existing purged walk-forward with 5 seeds, a paired
  daily-IC test and Benjamini-Hochberg. ListMLE is included because it is the top-focused form and we are long only.
- Written from equation 4 with no code copied. The fetch tool showed no licence on the repository.
- One learning rate and one batch size for every loss, fixed before the run. This is the direct answer to items 2 and 3 above.
- A rule for sessions with an odd number of names, since the loss assumes an even count. Drop the median name, for example.
- No ReLU on the final score.
- The long-only top-10 net result through the same layer, reported beside the IC. Here and in `eiak`, portfolio results move while IC barely
  does. An IC-only test would miss an effect that sits only at the top of the list.

Prior: low. Even a pass moves the IC by an amount that does not reach K1's wall, so it would not change trading on its own.

## Verdict

**read-only for the numbers; adopt-to-test narrowly for the formula, on a new card, after H-I1 and H-I2.** ListFold-exp won on paper, by 0.12
of return over rf and 0.72 of Sharpe over the runner-up, and 0.002 of IC over ListMLE. The test was walk-forward with sound scaling, but it
was not clean: 80 names picked with the whole sample, a batch size that is the winner's best on the test period (averaging over settings
cuts the lead on ListMLE from 33.6 to 2.9 bp a week, and on the other three to 5.5 to 7.7), a separate learning rate for the winner in the code, no seeds, and a single t of 1.45. Even taken at face value, IC 0.079 is
about half of what K1 needs at 50,000 SEK and a quarter of what it needs at 10,000 SEK. This paper does not raise H-T3's prior and does not move
the wall.
