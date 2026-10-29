"""Stage 1 alternatives (need the `nn` extra: torch, scikit-learn): P1 trees, P2 MLP, P3 Transformer, P4 GRU.

All predict the cross-sectional rank of the next 5 sessions' return (a target without the market's drift) from the
same ranked features, are fitted on rows purged before each fold, and take their hyperparameters from a random
search of the same size on data before the first development test (`tune`), so the comparison is fair and the
development window is never used to choose them.
"""
from __future__ import annotations

import math
import random

import numpy as np

from . import metrics as M
from .panel import FEATURES, Panel
from .predictors import Predictor
from .splits import FIRST_TEST

TUNE_TRIALS = 8
STRIDE = 3  # training rows: every third session, the newest included (neighbours share 4 of 5 label days)
MAX_ROWS = 120_000
SEQ_LEN = 60
HEADS = (1, 5, 20)
HEAD_W = (0.2, 1.0, 0.3)


def _rows(panel: Panel, lo: int, hi: int, heads=(5,), stride: int = STRIDE, cap: int = MAX_ROWS, seed: int = 0,
          need_seq: bool = False):
    """Training rows in sessions [lo, hi]: (t, s, Y) with Y (n, len(heads)) rank targets (NaN where unknown);
    only rows with a valid 5-session target and complete features."""
    cube = panel.ranked()
    ts = np.arange(lo, hi + 1)
    ts = ts[(hi - ts) % stride == 0]
    if need_seq:
        ts = ts[ts >= SEQ_LEN]
    lab = {h: panel.label(h, "rank") for h in heads}
    ok = panel.tradable[ts] & ~np.isnan(cube[ts]).any(axis=-1) & ~np.isnan(lab[5][ts])
    ti, si = np.nonzero(ok)
    t, s = ts[ti], si
    if len(t) > cap:
        sel = np.sort(np.random.default_rng(seed).choice(len(t), cap, replace=False))
        t, s = t[sel], s[sel]
    Y = np.stack([lab[h][t, s] for h in heads], axis=-1).astype(np.float32)
    return t, s, Y


def tune(make, space: dict, panel: Panel, trials: int = TUNE_TRIALS, seed: int = 0):
    """Random search on 2016..2018: fit before 2018-01-01 (purged), score rank IC on 2018. Returns (params, table)."""
    val_lo, val_hi = panel.index_of(__import__("datetime").date(2018, 1, 1)), panel.index_of(FIRST_TEST) - 1
    rng, table = random.Random(seed), []
    fwd = panel.label(5, "relative")
    for _ in range(trials):
        params = {k: rng.choice(v) for k, v in space.items()}
        pred = make(params)
        pred.fit(panel, val_lo - 1 - pred.horizon)
        idx = np.arange(val_lo, val_hi + 1)
        sc = pred.predict(panel, idx)
        ic = M.rank_ic(sc, fwd[idx], panel.tradable[idx], overlap=5)["ic"]
        table.append((ic, params))
    table.sort(key=lambda x: -x[0])
    return table[0][1], table


# --- P1: gradient-boosted trees --------------------------------------------------------------------------------------
class Trees(Predictor):
    name, horizon = "P1-trees", 5
    SPACE = {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1], "max_iter": [100, 200, 400],
             "l2_regularization": [0.0, 1.0, 10.0], "min_samples_leaf": [200, 1000, 5000]}

    def __init__(self, seed=0, **params):
        super().__init__(seed, **params)
        self.params = params

    def fit(self, panel, train_end):
        from sklearn.ensemble import HistGradientBoostingRegressor
        t, s, Y = _rows(panel, 0, train_end, seed=self.seed)
        X = panel.ranked()[t, s]
        self.m = HistGradientBoostingRegressor(random_state=self.seed, early_stopping=False, **self.params).fit(X, Y[:, 0])

    def predict(self, panel, idx):
        cube = panel.ranked()[idx]
        flat = np.nan_to_num(cube.reshape(-1, cube.shape[-1]))
        out = self.m.predict(flat).reshape(len(idx), -1)
        return np.where(np.isnan(cube).any(axis=-1), np.nan, out)


# --- torch models ----------------------------------------------------------------------------------------------------
def _torch():
    import torch
    torch.set_num_threads(12)
    return torch


class _Net:
    """Builds torch modules lazily so importing this module without torch fails only when used."""

    @staticmethod
    def mlp(nin, hidden, dropout, nout):
        t = _torch()
        return t.nn.Sequential(t.nn.Linear(nin, hidden), t.nn.GELU(), t.nn.Dropout(dropout), t.nn.Linear(hidden, hidden),
                               t.nn.GELU(), t.nn.Dropout(dropout), t.nn.Linear(hidden, nout))

    @staticmethod
    def seq_model(kind, d, layers, dropout, nstatic, nout):
        t = _torch()
        nn = t.nn

        class Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(len(Panel.SEQ_CHANNELS) + 1, d)  # + a mask flag for pre-training
                if kind == "transformer":
                    self.pos = nn.Parameter(t.zeros(SEQ_LEN, d))
                    layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, activation="gelu")
                    self.enc = nn.TransformerEncoder(layer, layers)
                else:
                    self.enc = nn.GRU(d, d, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
                self.recon = nn.Linear(d, 1)
                self.static = nn.Linear(nstatic, d)
                self.head = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(3 * d, d), nn.GELU(),
                                          nn.Linear(d, nout))
                self.kind = kind

            def encode(self, x):
                h = self.inp(x)
                if self.kind == "transformer":
                    h = self.enc(h + self.pos)
                else:
                    h, _ = self.enc(h)
                return h

            def forward(self, x, static):
                h = self.encode(x)
                return self.head(t.cat([h[:, -1], h.mean(1), self.static(static)], dim=-1))

        return Seq()


class _TorchPredictor(Predictor):
    horizon = 20  # the 20-session head's labels must be known too
    epochs = 8

    def _seed(self):
        t = _torch()
        t.manual_seed(self.seed)
        np.random.seed(self.seed)
        return t

    def _batches(self, n, bs, rng):
        order = rng.permutation(n)
        for i in range(0, n, bs):
            yield order[i:i + bs]

    @staticmethod
    def _loss(t, pred, Y):
        w = t.tensor(HEAD_W)
        m = ~t.isnan(Y)
        err = t.where(m, (pred - t.nan_to_num(Y)) ** 2, t.zeros_like(pred))
        return ((err * w).sum(1) / (m.float() * w).sum(1).clamp(min=1e-6)).mean()

    def _split(self, t_idx):
        """Last 15% of the training sessions are held out for early stopping, purged by the longest horizon."""
        cut = np.quantile(t_idx, 0.85)
        return t_idx <= cut - max(HEADS), t_idx > cut


class MLP(_TorchPredictor):
    name = "P2-mlp"
    SPACE = {"hidden": [32, 64, 128], "dropout": [0.0, 0.1, 0.3], "lr": [1e-3, 3e-3], "wd": [1e-4, 1e-2],
             "epochs": [6, 12]}

    def __init__(self, seed=0, hidden=64, dropout=0.1, lr=2e-3, wd=1e-3, epochs=8):
        super().__init__(seed, hidden=hidden, dropout=dropout, lr=lr, wd=wd, epochs=epochs)
        self.p = dict(hidden=hidden, dropout=dropout, lr=lr, wd=wd, epochs=epochs)

    def _pick(self, panel, train_end):
        """The training rows (t, s, Y); memory variants (memory.py) choose which past to keep."""
        return _rows(panel, 0, train_end, HEADS, seed=self.seed)

    def _init_net(self, nin):
        return _Net.mlp(nin, self.p["hidden"], self.p["dropout"], len(HEADS))

    def fit(self, panel, train_end):
        t = self._seed()
        ti, si, Y = self._pick(panel, train_end)
        X = np.nan_to_num(panel.ranked()[ti, si])
        tr, va = self._split(ti)
        net = self._init_net(X.shape[1])
        opt = t.optim.AdamW(net.parameters(), lr=self.p["lr"], weight_decay=self.p["wd"])
        Xt, Yt = t.tensor(X), t.tensor(Y)
        rng, best, best_state = np.random.default_rng(self.seed), math.inf, None
        itr = np.nonzero(tr)[0]
        for _ in range(self.p["epochs"]):
            net.train()
            for b in self._batches(len(itr), 1024, rng):
                j = itr[b]
                opt.zero_grad()
                self._loss(t, net(Xt[j]), Yt[j]).backward()
                opt.step()
            net.eval()
            with t.no_grad():
                v = float(self._loss(t, net(Xt[va]), Yt[va])) if va.any() else 0.0
            if v < best:
                best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
        net.load_state_dict(best_state)
        self.net = net.eval()

    def predict(self, panel, idx):
        t = _torch()
        cube = panel.ranked()[idx]
        flat = t.tensor(np.nan_to_num(cube.reshape(-1, cube.shape[-1])))
        with t.no_grad():
            out = self.net(flat)[:, HEADS.index(5)].numpy().reshape(len(idx), -1)
        return np.where(np.isnan(cube).any(axis=-1), np.nan, out)


class SeqModel(_TorchPredictor):
    """P3 (Transformer) and P4 (GRU): a shared model over the last 60 sessions' daily channels plus the ranked
    features. With `pretrain` > 0 epochs the encoder first learns to reconstruct masked days' returns on every
    training session (self-supervised: no labels, so it may use the most recent sessions too)."""
    kind = "transformer"
    SPACE = {"d": [16, 32], "layers": [1, 2], "dropout": [0.1, 0.3], "lr": [1e-3, 3e-3], "epochs": [4, 8],
             "pretrain": [3]}

    def __init__(self, seed=0, d=32, layers=2, dropout=0.1, lr=2e-3, epochs=6, pretrain=3):
        super().__init__(seed, d=d, layers=layers, dropout=dropout, lr=lr, epochs=epochs, pretrain=pretrain,
                         kind=self.kind)
        self.p = dict(d=d, layers=layers, dropout=dropout, lr=lr, epochs=epochs, pretrain=pretrain)

    def _x(self, panel, t_idx, s_idx):
        seq = panel.sequences()
        win = t_idx[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        x = seq[win, s_idx[:, None]]
        return np.concatenate([x, np.zeros(x.shape[:2] + (1,), np.float32)], axis=-1)

    def fit(self, panel, train_end):
        t = self._seed()
        ti, si, Y = _rows(panel, 0, train_end, HEADS, seed=self.seed, need_seq=True, cap=80_000)
        S = np.nan_to_num(panel.ranked()[ti, si])
        net = _Net.seq_model(self.kind, self.p["d"], self.p["layers"], self.p["dropout"], S.shape[1], len(HEADS))
        rng = np.random.default_rng(self.seed)
        if self.p["pretrain"]:
            self._pretrain(t, net, panel, train_end, rng)
        tr, va = self._split(ti)
        opt = t.optim.AdamW(net.parameters(), lr=self.p["lr"], weight_decay=1e-3)
        St, Yt = t.tensor(S), t.tensor(Y)
        itr, best, best_state = np.nonzero(tr)[0], math.inf, None
        for _ in range(self.p["epochs"]):
            net.train()
            for b in self._batches(len(itr), 512, rng):
                j = itr[b]
                opt.zero_grad()
                self._loss(t, net(t.tensor(self._x(panel, ti[j], si[j])), St[j]), Yt[j]).backward()
                opt.step()
            net.eval()
            with t.no_grad():
                vs = np.nonzero(va)[0][:6000]
                v = float(self._loss(t, net(t.tensor(self._x(panel, ti[vs], si[vs])), St[vs]), Yt[vs])) if len(vs) else 0.0
            if v < best:
                best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
        net.load_state_dict(best_state)
        self.net = net.eval()

    def _pretrain(self, t, net, panel, train_end, rng):
        """Masked-day reconstruction on sessions up to train_end: hide 15% of days, predict their return channel."""
        ti, si, _ = _rows(panel, SEQ_LEN, train_end, (5,), seed=self.seed + 1, need_seq=True, cap=60_000)
        params = list(net.inp.parameters()) + list(net.enc.parameters()) + list(net.recon.parameters())
        if hasattr(net, "pos"):
            params.append(net.pos)
        opt = t.optim.AdamW(params, lr=self.p["lr"], weight_decay=1e-3)
        for _ in range(self.p["pretrain"]):
            net.train()
            for b in self._batches(len(ti), 512, rng):
                x = t.tensor(self._x(panel, ti[b], si[b]))
                mask = t.rand(x.shape[:2]) < 0.15
                target = x[..., 0].clone()
                x = x.clone()
                x[mask] = 0.0
                x[..., -1] = mask.float()
                opt.zero_grad()
                rec = net.recon(net.encode(x)).squeeze(-1)
                ((rec - target) ** 2)[mask].mean().backward()
                opt.step()

    def predict(self, panel, idx):
        t = _torch()
        cube = panel.ranked()[idx]
        out = np.full(cube.shape[:2], np.nan, np.float32)
        ti, si = np.nonzero(~np.isnan(cube).any(axis=-1) & panel.tradable[idx])
        ti_abs = idx[ti]
        keep = ti_abs >= SEQ_LEN
        ti, si, ti_abs = ti[keep], si[keep], ti_abs[keep]
        with t.no_grad():
            for i in range(0, len(ti), 4096):
                sl = slice(i, i + 4096)
                x = t.tensor(self._x(panel, ti_abs[sl], si[sl]))
                st = t.tensor(np.nan_to_num(cube[ti[sl], si[sl]]))
                out[ti[sl], si[sl]] = self.net(x, st)[:, HEADS.index(5)].numpy()
        return out


class Transformer(SeqModel):
    name, kind = "P3-transformer", "transformer"


class TransformerScratch(Transformer):
    name = "P3-transformer-scratch"

    def __init__(self, seed=0, **kw):
        kw["pretrain"] = 0
        super().__init__(seed, **kw)


class GRU(SeqModel):
    name, kind = "P4-gru", "gru"


ALTERNATIVES = {
    "P1-trees": (lambda seed=0, **p: Trees(seed, **p), Trees.SPACE),
    "P2-mlp": (lambda seed=0, **p: MLP(seed, **p), MLP.SPACE),
    "P3-transformer": (lambda seed=0, **p: Transformer(seed, **p), SeqModel.SPACE),
    "P3-transformer-scratch": (lambda seed=0, **p: TransformerScratch(seed, **{k: v for k, v in p.items() if k != "pretrain"}),
                               {k: v for k, v in SeqModel.SPACE.items() if k != "pretrain"}),
    "P4-gru": (lambda seed=0, **p: GRU(seed, **p), SeqModel.SPACE),
}
