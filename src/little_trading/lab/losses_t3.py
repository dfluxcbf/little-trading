"""H-T3: ranking losses for the training family (docs/plans/improvements-lab.md section 5; sources eiak, crt7, rchl, whose
losses are the standard ones, defined here from first principles). Our models learn the cross-sectional rank of the next
5 sessions' return with a pointwise squared error; the trades only use the order of the scores inside a session, so a loss
that looks at that order directly might rank better. Every loss here is computed per session (one trading day, the names
tradable that day) and averaged over the sessions that have at least two names. For a session with scores s_i and
targets y_i, i = 1..n:

- pointwise (our current loss): L = (1/n) sum_i (s_i - y_i)^2 / 2. dL/ds_i = (s_i - y_i) / n.
- pairwise RankNet: over the P ordered pairs with y_i > y_j, L = (1/P) sum log(1 + exp(-(s_i - s_j))), the logistic loss of
  "i ranks above j". With G_ij = -sigmoid(-(s_i - s_j)) / P on those pairs, dL/ds_k = sum_j G_kj - sum_i G_ik.
- listwise ListNet (top-one probabilities): the target distribution p = softmax(z(y) / tau), z the per-session z-score of
  the targets, the score distribution q = softmax(s); L = -sum_i p_i log q_i, the cross-entropy. dL/ds_i = q_i - p_i.
- per-day IC: L = -corr(s, y), the negative Pearson correlation of the scores and the targets inside the session (on rank
  targets the Spearman IC the lab is judged on, up to the ranking of the scores). With centred sc, yc, a = |sc|^2,
  b = |yc|^2, c = sc.yc and D = sqrt(a b + eps): dL/ds = -(yc / D - c b sc / D^3); the centring passes through because
  that vector already sums to 0.

Sessions are padded to one (G, n) array with a mask M (the panel's names are a natural padding: a row per session, a
column per company), so every loss and gradient is vectorised over sessions. The forward losses are written once against a
tiny array-operations shim (`NumpyOps`, `TorchOps`): the numpy tests check them against the analytic gradients by finite
differences, and the torch MLP trains on the very same forward through autograd (torch is imported only inside
`TorchOps` and `_fit_mlp`, never at module level).

`RankingModel` is the `Predictor` the service walks forward: the 18 ranked features, the 5-session rank label, every
`stride`-th session up to the fold's purged `train_end`, whole sessions kept together. model "linear" (numpy: s = X w + b,
Adam on the analytic gradient, minibatches of sessions; what the tests run) or "mlp" (torch, the architecture of
`alternatives.MLP`: two GELU hidden layers with dropout, AdamW, early stopping on the last 15 % of the training sessions
purged by the label horizon). A small L2 penalty on w keeps the scale-free losses (IC; RankNet on separable data) bounded.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .hypothesis import ranked_cube
from .panel import Panel
from .predictors import Predictor

LOSSES = ("pointwise", "ranknet", "listnet", "ic")
BASELINE = "pointwise"
EPS = 1e-12
NEG = -1e9  # a masked score in a softmax


# --- the forward losses, written once for numpy and torch -------------------------------------------------------------------
class NumpyOps:
    float = staticmethod(lambda m: np.asarray(m, float))
    where = staticmethod(np.where)
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    softplus = staticmethod(lambda x: np.logaddexp(0.0, x))
    clamp_min = staticmethod(np.maximum)

    @staticmethod
    def sum(x, axis, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def logsumexp(x, axis):
        m = np.max(x, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m


class TorchOps:
    def __init__(self):
        import torch
        self.t = torch
        self.float = lambda m: m.to(torch.float32)
        self.where, self.exp, self.log, self.sqrt = torch.where, torch.exp, torch.log, torch.sqrt
        self.softplus = torch.nn.functional.softplus
        self.clamp_min = lambda x, v: torch.clamp(x, min=v)

    def sum(self, x, axis, keepdims=False):
        return x.sum(dim=axis, keepdim=keepdims)

    def logsumexp(self, x, axis):
        return self.t.logsumexp(x, dim=axis, keepdim=True)


def _sessions(M, ops):
    """Per session: the number of names, and 1 for a session with at least two (the ones a loss averages over)."""
    n = ops.sum(ops.float(M), 1)
    v = ops.float(n >= 2)
    return n, v, ops.clamp_min(ops.sum(v, 0), 1.0)


def forward(name: str, S, Y, M, ops=NumpyOps, tau: float = 1.0):
    """The scalar loss `name` of scores S against targets Y, both (G, n), on the names where the mask M is true."""
    Mf = ops.float(M)
    n, v, V = _sessions(M, ops)
    S0, Y0 = ops.where(M, S, 0.0 * S), ops.where(M, Y, 0.0 * Y)
    if name == "pointwise":
        per = ops.sum(0.5 * (S0 - Y0) ** 2 * Mf, 1) / ops.clamp_min(n, 1.0)
    elif name == "ranknet":
        P = ops.float((Y0[:, :, None] > Y0[:, None, :]) & M[:, :, None] & M[:, None, :])
        D = S0[:, :, None] - S0[:, None, :]
        per = ops.sum(ops.sum(P * ops.softplus(-D), 2), 1) / ops.clamp_min(ops.sum(ops.sum(P, 2), 1), 1.0)
    elif name == "listnet":
        z = _zscore(Y0, Mf, n, ops)
        p = _masked_softmax(z / tau, M, ops)
        Sm = ops.where(M, S, NEG + 0.0 * S)
        logq = Sm - ops.logsumexp(Sm, 1)
        per = -ops.sum(ops.where(M, p * logq, 0.0 * S), 1)
    elif name == "ic":
        nn = ops.clamp_min(n, 1.0)[:, None]
        sc = (S0 - ops.sum(S0, 1, True) / nn) * Mf
        yc = (Y0 - ops.sum(Y0, 1, True) / nn) * Mf
        per = -ops.sum(sc * yc, 1) / ops.sqrt(ops.sum(sc * sc, 1) * ops.sum(yc * yc, 1) + EPS)
    else:
        raise ValueError(f"unknown loss {name!r}: one of {LOSSES}")
    return ops.sum(per * v, 0) / V


def _zscore(Y0, Mf, n, ops):
    nn = ops.clamp_min(n, 1.0)[:, None]
    yc = (Y0 - ops.sum(Y0, 1, True) / nn) * Mf
    sd = ops.sqrt(ops.sum(yc * yc, 1, True) / nn + EPS)
    return yc / sd


def _masked_softmax(X, M, ops):
    Xm = ops.where(M, X, NEG + 0.0 * X)
    e = ops.where(M, ops.exp(Xm - ops.logsumexp(Xm, 1)), 0.0 * X)
    return e


# --- analytic gradients (numpy) ---------------------------------------------------------------------------------------------
def loss_and_grad(name: str, S: np.ndarray, Y: np.ndarray, M: np.ndarray, tau: float = 1.0) -> tuple[float, np.ndarray]:
    """(loss, dLoss/dS) with dS zero on the masked cells; the loss equals `forward(name, ...)`."""
    S, Y, M = np.asarray(S, float), np.asarray(Y, float), np.asarray(M, bool)
    Mf = M.astype(float)
    n = Mf.sum(1)
    v = (n >= 2).astype(float)
    V = max(1.0, v.sum())
    S0, Y0 = np.where(M, S, 0.0), np.where(M, Y, 0.0)
    w = (v / V)[:, None]
    if name == "pointwise":
        loss = forward(name, S, Y, M)
        return float(loss), (S0 - Y0) * Mf / np.maximum(n, 1.0)[:, None] * w
    if name == "ranknet":
        P = ((Y0[:, :, None] > Y0[:, None, :]) & M[:, :, None] & M[:, None, :]).astype(float)
        D = S0[:, :, None] - S0[:, None, :]
        npairs = np.maximum(P.sum((1, 2)), 1.0)
        per = (P * np.logaddexp(0.0, -D)).sum((1, 2)) / npairs
        G = -P * _sigmoid(-D) / npairs[:, None, None]
        dS = (G.sum(2) - G.sum(1)) * Mf
        return float((per * v).sum() / V), dS * w
    if name == "listnet":
        p = _masked_softmax(_zscore(Y0, Mf, n, NumpyOps) / tau, M, NumpyOps)
        Sm = np.where(M, S, NEG)
        lse = NumpyOps.logsumexp(Sm, 1)
        q = np.where(M, np.exp(Sm - lse), 0.0)
        per = -np.where(M, p * (Sm - lse), 0.0).sum(1)
        return float((per * v).sum() / V), (q - p) * Mf * w
    if name == "ic":
        nn = np.maximum(n, 1.0)[:, None]
        sc = (S0 - S0.sum(1, keepdims=True) / nn) * Mf
        yc = (Y0 - Y0.sum(1, keepdims=True) / nn) * Mf
        a, b, c = (sc * sc).sum(1, keepdims=True), (yc * yc).sum(1, keepdims=True), (sc * yc).sum(1, keepdims=True)
        D = np.sqrt(a * b + EPS)
        per = -(c / D)[:, 0]
        dS = -(yc / D - c * b * sc / D ** 3) * Mf
        return float((per * v).sum() / V), dS * w
    raise ValueError(f"unknown loss {name!r}: one of {LOSSES}")


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# --- the linear scorer (numpy) ----------------------------------------------------------------------------------------------
def linear_scores(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """s = X w + b for X of shape (..., F) and theta = (w, b)."""
    return X @ theta[:-1] + theta[-1]


def objective(theta: np.ndarray, X: np.ndarray, Y: np.ndarray, M: np.ndarray, name: str, l2: float = 0.0,
              tau: float = 1.0) -> tuple[float, np.ndarray]:
    """Loss of the linear scorer on padded sessions X (G, n, F), Y and M (G, n), plus l2/2 |w|^2, and its gradient in theta."""
    loss, dS = loss_and_grad(name, linear_scores(theta, X), Y, M, tau)
    g = np.concatenate([np.einsum("gnf,gn->f", X, dS), [dS.sum()]])
    g[:-1] += l2 * theta[:-1]
    return loss + 0.5 * l2 * float(theta[:-1] @ theta[:-1]), g


def fit_linear(X: np.ndarray, Y: np.ndarray, M: np.ndarray, name: str, l2: float = 1e-4, steps: int = 300,
               lr: float = 0.02, batch: int = 32, seed: int = 0, tau: float = 1.0) -> np.ndarray:
    """theta = (w, b) trained by Adam on minibatches of `batch` sessions (all sessions when there are fewer)."""
    rng = np.random.default_rng(seed)
    G, F = X.shape[0], X.shape[-1]
    theta = np.concatenate([rng.normal(0.0, 0.01, F), [0.0]])
    m1, m2 = np.zeros_like(theta), np.zeros_like(theta)
    for k in range(1, steps + 1):
        idx = rng.choice(G, batch, replace=False) if G > batch else slice(None)
        _, g = objective(theta, X[idx], Y[idx], M[idx], name, l2, tau)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        theta = theta - lr * (m1 / (1 - 0.9 ** k)) / (np.sqrt(m2 / (1 - 0.999 ** k)) + 1e-8)
    return theta


def pad(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows (X (n, F), y (n,)) with a session id each, to padded (G, nmax, F), (G, nmax) and the mask."""
    keys, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    G, nmax = len(keys), int(counts.max())
    pos = np.zeros(len(y), int)
    seen = np.zeros(G, int)
    for i, g in enumerate(inv):
        pos[i], seen[g] = seen[g], seen[g] + 1
    Xp, Yp, Mp = np.zeros((G, nmax, X.shape[1])), np.zeros((G, nmax)), np.zeros((G, nmax), bool)
    Xp[inv, pos], Yp[inv, pos], Mp[inv, pos] = X, y, True
    return Xp, Yp, Mp


def session_ic(S: np.ndarray, Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(G,) Pearson correlation of scores and targets per session (the negative of the per-session IC loss)."""
    return np.array([-forward("ic", S[g:g + 1], Y[g:g + 1], M[g:g + 1]) for g in range(len(S))])


# --- the MLP (torch, lazy) -------------------------------------------------------------------------------------------------
MLP_DEFAULTS = {"hidden": 64, "dropout": 0.1, "lr": 2e-3, "wd": 1e-3, "epochs": 8, "batch": 16, "threads": 4}
LINEAR_DEFAULTS = {"l2": 1e-4, "steps": 300, "lr": 0.02, "batch": 32}


def _fit_mlp(X, Y, M, name: str, seed: int, horizon: int, stride: int, params: dict, tau: float = 1.0):
    """A torch MLP trained on the loss `name` (same forward as the numpy tests, through `TorchOps`); returns a numpy
    scoring function (N, F) -> (N,)."""
    import torch
    p = {**MLP_DEFAULTS, **params}
    torch.set_num_threads(int(p["threads"]))
    torch.manual_seed(seed)
    ops = TorchOps()
    nin = X.shape[-1]
    net = torch.nn.Sequential(torch.nn.Linear(nin, p["hidden"]), torch.nn.GELU(), torch.nn.Dropout(p["dropout"]),
                              torch.nn.Linear(p["hidden"], p["hidden"]), torch.nn.GELU(), torch.nn.Dropout(p["dropout"]),
                              torch.nn.Linear(p["hidden"], 1))
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
    Xt, Yt, Mt = (torch.tensor(np.asarray(X, np.float32)), torch.tensor(np.asarray(Y, np.float32)),
                  torch.tensor(np.asarray(M, bool)))
    G = len(Xt)
    # the last 15 % of the sessions validate, purged by the label horizon (sessions are `stride` apart)
    cut = int(G * 0.85)
    gap = int(math.ceil(horizon / max(1, stride)))
    tr, va = np.arange(max(1, cut - gap)), np.arange(cut, G)
    rng, best, best_state = np.random.default_rng(seed), math.inf, None
    for _ in range(int(p["epochs"])):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), int(p["batch"])):
            j = torch.tensor(order[i:i + int(p["batch"])])
            opt.zero_grad()
            forward(name, net(Xt[j]).squeeze(-1), Yt[j], Mt[j], ops, tau).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            jv = torch.tensor(va)
            v = float(forward(name, net(Xt[jv]).squeeze(-1), Yt[jv], Mt[jv], ops, tau)) if len(va) else 0.0
        if v < best or best_state is None:
            best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()

    def score(Xn: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.tensor(np.asarray(Xn, np.float32))).squeeze(-1).numpy().astype(float)
    return score


# --- the predictor ---------------------------------------------------------------------------------------------------------
class RankingModel(Predictor):
    """A scorer trained with one of `LOSSES` on whole sessions: features (T, N, F) (the 18 ranked ones by default) against
    the 5-session rank label, every `stride`-th session up to `train_end`, at most `max_sessions` of them (the newest kept)."""

    def __init__(self, seed: int = 0, loss: str = BASELINE, model: str = "linear", params: dict | None = None,
                 features: Callable[[Panel], np.ndarray] = ranked_cube, label: Callable[[Panel], np.ndarray] | None = None,
                 horizon: int = 5, stride: int = 3, max_sessions: int = 1500, tau: float = 1.0, name: str | None = None,
                 fitter: Callable | None = None):
        if loss not in LOSSES:
            raise ValueError(f"unknown loss {loss!r}: one of {LOSSES}")
        if model not in ("linear", "mlp"):
            raise ValueError(f"unknown model {model!r}: linear or mlp")
        super().__init__(seed, loss=loss, model=model, horizon=horizon, **(params or {}))
        self.loss, self.model, self.params, self.features, self.horizon = loss, model, dict(params or {}), features, horizon
        self.label = label or (lambda p: p.label(5, "rank"))
        self.stride, self.max_sessions, self.tau, self.name = stride, max_sessions, tau, name or f"{model}-{loss}"
        self.fitter = fitter  # replaces the trainer: (X, Y, M, loss, seed) -> scoring function (N, F) -> (N,)
        self.score = None

    def sessions(self, panel: Panel, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(ts, X (G, N, F), Y (G, N), M (G, N)): the training sessions, padded by the panel's own columns."""
        cube, y_all = np.asarray(self.features(panel), float), self.label(panel)
        ts = np.arange(0, train_end + 1)
        ts = ts[(train_end - ts) % self.stride == 0][-self.max_sessions:]
        M = panel.tradable[ts] & ~np.isnan(cube[ts]).any(axis=-1) & ~np.isnan(y_all[ts])
        keep = M.sum(1) >= 2
        ts, M = ts[keep], M[keep]
        return ts, np.where(M[..., None], np.nan_to_num(cube[ts]), 0.0), np.where(M, np.nan_to_num(y_all[ts]), 0.0), M

    def fit(self, panel: Panel, train_end: int) -> None:
        self.score = None
        if train_end < 1:
            return
        ts, X, Y, M = self.sessions(panel, train_end)
        if len(ts) < 10:
            return
        if self.fitter is not None:
            self.score = self.fitter(X, Y, M, self.loss, self.seed)
        elif self.model == "linear":
            p = {**LINEAR_DEFAULTS, **self.params}
            theta = fit_linear(X, Y, M, self.loss, float(p["l2"]), int(p["steps"]), float(p["lr"]), int(p["batch"]),
                               self.seed, self.tau)
            self.theta = theta
            self.score = lambda Xn, th=theta: linear_scores(th, Xn)
        else:
            self.score = _fit_mlp(X, Y, M, self.loss, self.seed, self.horizon, self.stride, self.params, self.tau)

    def predict(self, panel: Panel, idx: np.ndarray) -> np.ndarray:
        cube = np.asarray(self.features(panel), float)[idx]
        out = np.full(cube.shape[:2], np.nan)
        if self.score is None:
            return out
        flat = cube.reshape(-1, cube.shape[-1])
        ok = ~np.isnan(flat).any(axis=-1)
        if ok.any():
            out.reshape(-1)[ok] = self.score(flat[ok])
        return out


def makers(losses=LOSSES, model: str = "linear", params: dict | None = None, tau: float = 1.0,
           fitter: Callable | None = None) -> dict[str, Callable[[int], RankingModel]]:
    """One maker (seed -> RankingModel) per loss, named "<model>-<loss>"."""
    return {f"{model}-{loss}": (lambda s, loss=loss: RankingModel(s, loss=loss, model=model, params=params, tau=tau,
                                                                   fitter=fitter))
            for loss in losses}


__all__ = ["LOSSES", "BASELINE", "NumpyOps", "TorchOps", "forward", "loss_and_grad", "objective", "fit_linear", "pad",
           "session_ic", "linear_scores", "RankingModel", "makers"]
