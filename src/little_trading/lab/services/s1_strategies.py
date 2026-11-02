"""Service `s1-strategies`: six strategies from market value and earnings news, alone or together, through one fixed layer.

The method is in `little_trading.lab.strategies`; the plan is docs/plans/strategies.md and the card docs/lab/cards/s1-strategies.md.
It reads the window `Ctx.window("strategies")` hands out and cuts the panel at the end of its spans, so no label, price or
filing past them is read. Its EDGAR data comes through the lab's client into its own `edgar-cache`, at the client's rate
limit: companyfacts, every page of each company's submissions, and the press release of each earnings 8-K. The EDGAR contact
comes only from the lab's local config. The transport, clock and sleep come from the module-level hooks, which tests replace.

Restricted issuers leave the panel before any request, whether by ticker, by the user's list or by the company's name in SEC's
ticker map, so no model is ever fitted on them and no cell holds them.

Checks run before any score. The run reports and stops, with no strategy scored, no variant counted and no selection look
taken, if a fetch failed for a reason that could pass (a rerun asks again, since a failure is never cached), if too few
companies have earnings events in any year of the window, or if too few of a year's events have a press-release tone. A
missing document (`edgar.NotFound`) is a fact about the company, not a failure. The look is taken just before the first score;
until then the run reports only filings and event counts from the window, no outcome. It writes only its own service.db.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from little_trading import restricted
from little_trading.lab import earnings_i1 as E
from little_trading.lab import edgar, hypothesis, label_horizon_o1 as O1, layers_b
from little_trading.lab import fundamentals_i2 as F
from little_trading.lab import strategies as S
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip
from little_trading.lab.services.k1_edge_needed import make_cost

SERVICE_ID = "s1-strategies"
FAMILY = S.FAMILY
VARIANTS = S.VARIANTS
TRANSPORT = None  # url -> bytes; tests replace it, None is the network
CLOCK = None
SLEEP = None
COST_CASES = {"net": {"broker": "avanza-mini"}, "fees x0.5": {"broker": "avanza-mini", "fee_scale": 0.5},
              "fees x2": {"broker": "avanza-mini", "fee_scale": 2.0}, "gross": {"free": True}}
DEFAULTS = {**layers_b.DEFAULTS, "accounts": [10_000, 50_000, 250_000], "planted_ics": [0.05, 0.10, 0.30], "noise_runs": 200,
            "planted_runs": 5, "rate": 5.0, "submissions_max_age_hours": 24 * 7, "facts_max_age_hours": 24 * 7,
            "acceptance_tz": "America/New_York", "costs": COST_CASES}

MAX_FAILS_IN_A_ROW = 20  # SEC is down or has blocked this client: stop asking and let the run report it

SCHEMA = """
CREATE TABLE IF NOT EXISTS s1_fetch (run_id TEXT, symbol TEXT, cik INTEGER, facts INTEGER, events INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS s1_coverage (run_id TEXT, year INTEGER, eligible INTEGER, covered INTEGER, share REAL, ok INTEGER);
CREATE TABLE IF NOT EXISTS s1_report (run_id TEXT, section TEXT, key TEXT, detail TEXT);
"""


def dropped_from_panel(panel: Panel, client: edgar.Client) -> list[str]:
    """The panel's names that SEC's ticker map or the restricted list refuses, by ticker or by company name."""
    client.tickers()
    return [s for s in panel.symbols
            if client.restrictions.reason(s) or s.upper() in client.dropped or s.upper().replace(".", "-") in client.dropped]


def without(panel: Panel, names: list[str]) -> Panel:
    """The panel with those columns gone, so nothing later can train on or trade them."""
    if not names:
        return panel
    keep = [j for j, s in enumerate(panel.symbols) if s not in set(names)]
    return Panel.from_arrays(panel.dates, [panel.symbols[j] for j in keep], panel.open[:, keep], panel.high[:, keep],
                             panel.low[:, keep], panel.close[:, keep], panel.volume[:, keep], panel.market_close,
                             panel.market_open, usdsek=panel.usdsek)


def collect(ctx, cfg: dict, panel: Panel, client: edgar.Client, last: date) -> tuple[dict, list[dict], list[tuple], set]:
    """Per allowed company: its facts as filed by `last`, and its earnings events between the panel's start and `last` with
    each press release's lexicon tone. No exhibit filed after `last` is fetched. Facts and filings are fetched apart, so one
    failing does not lose the other. The fifth item of a log row says what went wrong; `failed` holds the companies with a
    fetch failure that could pass, which stops the run. Asking stops after `MAX_FAILS_IN_A_ROW` such failures in a row."""
    first = panel.dates[0]
    facts_by, events, log, ciks, failed = {}, [], [], set(), set()
    subs_age = float(cfg["submissions_max_age_hours"]) * 3600
    facts_age = float(cfg["facts_max_age_hours"]) * 3600
    in_a_row = 0

    def fetch(sym: str, what: str, call, notes: list[str]):
        """`call()`, or None with a note. A document that does not exist or does not parse is not a failure of the fetch."""
        nonlocal in_a_row
        try:
            out = call()
        except restricted.RestrictedSymbolError:
            raise
        except edgar.NotFound:
            notes.append(f"no {what}")
        except edgar.TRANSIENT as exc:
            in_a_row += 1
            failed.add(sym)
            notes.append(f"{what} failed: {type(exc).__name__}")
        except (ValueError, KeyError) as exc:
            notes.append(f"{what} unreadable: {type(exc).__name__}")
        else:
            in_a_row = 0
            return out
        return None

    for sym in panel.symbols:
        if in_a_row >= MAX_FAILS_IN_A_ROW:
            ctx.log(f"{SERVICE_ID}: {in_a_row} failed fetches in a row, not asking for {sym} or the names after it")
            failed.add(sym)
            log.append((sym, client.cik_for(sym), 0, 0, "not asked: too many failed fetches in a row"))
            continue
        cik = client.cik_for(sym)
        if cik is None:
            log.append((sym, None, 0, 0, "no CIK in the ticker map"))
            continue
        ciks.add(sym)
        notes: list[str] = []
        facts = fetch(sym, "facts", lambda: S.filed_by(F.parse_facts(client.companyfacts(cik, facts_age), S.FIELDS), last), notes)
        if facts is not None:
            facts_by[sym] = facts
        filings = fetch(sym, "filings", lambda: E.item_202(client.all_filings(cik, subs_age, since=first.isoformat()),
                                                           cfg["acceptance_tz"]), notes) or []
        mine = [f for f in filings if first <= f["acceptance"].date() <= last]
        lost: dict[str, int] = {}
        for f in mine:
            ev = {"symbol": sym, "cik": cik, **f}
            if in_a_row < MAX_FAILS_IN_A_ROW:
                said: list[str] = []
                name = fetch(sym, "press release index",
                             lambda: edgar.press_release_from_index(client.filing_index(cik, f["accession"])), said)
                if name:
                    tone = fetch(sym, "press release",
                                 lambda: E.lexicon_tone(edgar.html_text(client.document(cik, f["accession"], name))), said)
                    if tone is not None:
                        ev["tone"] = tone
                for note in said:
                    lost[note] = lost.get(note, 0) + 1
            else:
                failed.add(sym)
                lost["press release not asked"] = lost.get("press release not asked", 0) + 1
            events.append(ev)
        notes += [f"{note} x{n}" for note, n in lost.items()]
        log.append((sym, cik, sum(len(v) for v in (facts or {}).values()), len(mine), "; ".join(notes) or "ok"))
    return facts_by, events, log, ciks, failed


def stop_reason(failed: set, cover: list[dict], tones: list[dict]) -> tuple[str, str] | None:
    """(key, sentence) when the data cannot be scored, in the order a reader should fix it: a failed fetch makes the other two
    numbers meaningless. None when it can."""
    if failed:
        names = sorted(failed)
        shown = ", ".join(names[:10]) + (", ..." if len(names) > 10 else "")
        return "fetch errors", f"fetching failed for {len(names)} companies ({shown}); run it again, a failure is not cached"
    short = [c for c in cover if not c["ok"]]
    if short:
        return "coverage", (f"fewer than {S.COVERAGE_SHARE:.0%} of companies have {S.COVERAGE_EVENTS} events in "
                            + ", ".join(str(c["year"]) for c in short))
    toneless = [c for c in tones if not c["ok"]]
    if toneless:
        return "press-release tone", (f"fewer than {S.TONE_SHARE:.0%} of the events have a press-release tone in "
                                      + ", ".join(str(c["year"]) for c in toneless))
    return None


def run(ctx) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    win = ctx.window_dates(FAMILY)  # refuses what `window` would; the look itself is taken just before the first score
    default_spans = S.SPANS if cfg.get("window", "development") == "development" else hypothesis.spans_for(win)
    spans = hypothesis.spans_in_window(cfg.get("spans") or default_spans, win)
    chosen = list(cfg.get("strategies") or S.STRATEGIES)
    if unknown := set(chosen) - set(S.STRATEGIES):
        raise ValueError(f"not on the card: {sorted(unknown)}")
    start = date.fromisoformat(min(a for a, _ in spans.values()))
    end = date.fromisoformat(max(b for _, b in spans.values()))
    panel = Panel.from_db(str(ctx.snapshot_path), restrictions=restricted.load())
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    panel = O1.until(panel, end)
    client = edgar.client_for(ctx, TRANSPORT, CLOCK, SLEEP, rate=float(cfg["rate"]))
    gone = dropped_from_panel(panel, client)
    panel = without(panel, gone)

    facts, events, log, ciks, failed = collect(ctx, cfg, panel, client, end)
    placed = E.place_events(S.tone_changes(events), panel.dates, panel.symbols, panel.close, panel.market_close)
    cover = S.coverage(panel, ciks, placed, start, end)
    tones_by_year = S.tone_coverage(panel, placed, start, end)
    db = ctx.db()
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO s1_fetch VALUES (?,?,?,?,?,?)", [(ctx.run_id, *row) for row in log])
    db.executemany("INSERT INTO s1_coverage VALUES (?,?,?,?,?,?)",
                   [(ctx.run_id, c["year"], c["eligible"], c["covered"], c["share"], int(c["ok"])) for c in cover])
    db.commit()
    tones = sum(1 for ev in placed if ev.get("tone") is not None)
    data = {"companies": len(panel.symbols), "with_cik": len(ciks), "with_facts": len(facts), "events": len(placed),
            "events_with_tone": tones, "restricted_dropped": sorted(gone), "requests": len(client.fetched), "coverage": cover,
            "tone_coverage": tones_by_year, "fetch_failed": sorted(failed)}
    stop = stop_reason(failed, cover, tones_by_year)
    if stop:
        reason, why = stop
        msg = f"{SERVICE_ID}: stopped before scoring: {why}"
        ctx.log(msg)
        ctx.result(metrics=layers_b._jsonable({"service": SERVICE_ID, "stopped": reason, **data}), variants=0,
                   window=str(cfg.get("window", "development")), data_hash=ctx.manifest().get("hash"), message=msg)
        return {"stopped": True, "coverage": cover}

    ctx.window(FAMILY)  # records the one look at the selection window; nothing reported so far holds an outcome from it
    metrics, report = evaluate(ctx, cfg, panel, facts, placed, spans, start, end, chosen)
    db.executemany("INSERT INTO s1_report VALUES (?,?,?,?)",
                   [(ctx.run_id, section, key, json.dumps(layers_b._jsonable(detail))) for section, key, detail in report])
    db.commit()
    metrics = layers_b._jsonable({**metrics, **data})
    json.dumps(metrics)  # fails here, not in the lab, if anything is not plain JSON
    new = VARIANTS if cfg.get("window", "development") == "development" else 0  # a confirming look tries nothing new
    ctx.result(metrics=metrics, variants=new, window=str(cfg.get("window", "development")), data_hash=ctx.manifest().get("hash"),
               message=f"{SERVICE_ID}: {metrics['summary']} at {int(S.DESIGN_ACCOUNT)} SEK; "
                       f"eligible for a selection look: {', '.join(metrics['eligible']) or 'none'}")
    return metrics


def evaluate(ctx, cfg: dict, panel: Panel, facts: dict, placed: list[dict], spans: dict, start: date, end: date, chosen: list[str]):
    seeds, refit = tuple(cfg["seeds"]), int(cfg["refit_every"])
    design = S.DESIGN_ACCOUNT
    accounts = tuple(float(a) for a in cfg["accounts"])
    if design not in accounts:
        raise ValueError(f"the pass rule is read at {int(design)} SEK, and it is not among the accounts {accounts}")
    value = S.panel_value(panel.dates, panel.symbols, panel.close, facts)
    true_price = value[..., S.VALUE_COLUMNS.index("true_price")]
    ctx.log(f"{SERVICE_ID}: T-20 and the Stage-1 scores ({cfg['learner']}) to {end}")
    t20 = S.t20_scores(panel, S.t20_maker(cfg), end, seeds, refit, cache=str(ctx.run_dir / "t20.npz"))
    stage1 = hypothesis.real_scores(panel, layers_b.stage1_maker(cfg), start, end, seeds, refit,
                                    cache=str(ctx.run_dir / "stage1.npz"))
    m = S.models(panel, value, placed, t20, start, end, refit)
    variants = S.strategy_variants(m, true_price, chosen)
    costs = {name: make_cost(spec) for name, spec in cfg["costs"].items()}
    if "net" not in costs:
        raise ValueError("the pass rule is read on the net cost case, and it is not among the costs")
    champion = hypothesis.BASELINE_LAYER

    cells, through_champion = [], []
    for case, (broker, slip) in costs.items():
        label = "real" if case == "net" else f"real {case}"
        ctx.log(f"{SERVICE_ID}: {case}, {len(variants)} strategies x {len(accounts)} accounts")
        cells += S.cells_of(panel, champion, label, stage1, S.champion_spec(), spans, accounts, broker, slip)
        for sid, (score, spec) in variants.items():
            cells += S.cells_of(panel, sid, label, score, spec, spans, accounts, broker, slip)
        if case == "net":
            for sid, (score, _) in variants.items():
                through_champion += S.cells_of(panel, f"{sid} through D1", "real, champion layer", score, S.champion_spec(),
                                               spans, accounts, broker, slip)
    verdicts = hypothesis.decision_verdicts(cells, judge_on="real")
    hypothesis.store_decision(ctx, FAMILY, cells + through_champion, verdicts)

    ew = S.ew_daily(panel, spans)
    vs_ew = S.against_ew([c for c in cells if c["variant"] != champion], ew)
    net_broker, net_slip = costs["net"]
    bar = {c["span"]: c["daily"] for c in cells if c["variant"] == champion and c["scores"] == "real" and c["account"] == design}
    noise = S.noise_arms(panel, int(cfg["noise_runs"]), spans, true_price, bar, ew, design, net_broker, net_slip)
    planted = S.planted_arms(panel, tuple(cfg["planted_ics"]), int(cfg["planted_runs"]), spans, true_price, bar, ew, design,
                             net_broker, net_slip)
    credit = S.together_credit(cells)
    t1 = {v["variant"]: v for v in verdicts if v["judged"] and v["account"] == design}
    eligible = {sid: S.eligibility(t1[sid]["passes"], vs_ew[(sid, "real", design)], noise) for sid in variants}
    ic = {sid: S.rank_ic(panel, score, spans) for sid, (score, _) in variants.items()}
    same_score = {}
    mine = {(c["variant"], c["span"], c["account"]): c for c in cells if c["scores"] == "real"}
    for c in through_champion:
        sid = c["variant"].removesuffix(" through D1")
        l5 = mine[(sid, c["span"], c["account"])]
        same_score.setdefault(f"{sid} | {int(c['account'])}", {})[c["span"]] = S.paired_interval(l5["daily"], c["daily"])
    design_cells = [c for c in cells if c["scores"] == "real" and c["account"] == design]
    report = [("eligibility", k, v) for k, v in eligible.items()]
    report += [("rank_ic_63", k, v) for k, v in ic.items()]
    report += [("noise", k, v) for k, v in noise.items()]
    report += [("planted", k, v) for k, v in planted.items()]
    report += [("together", k, v) for k, v in credit.items()]
    report += [("same_score_vs_champion_layer", k, v) for k, v in same_score.items()]
    report += [("against_equal_weight", f"{v} | {s} | {int(a)}", row) for (v, s, a), row in vs_ew.items()]
    report += [("spy", "bought once", S.spy_returns(panel, spans, design, net_broker))]
    report += [("fees_per_year", f"{c['variant']} | {c['span']}", S.fees_per_year(c)) for c in design_cells]
    judged = [t1[sid] for sid in variants]
    metrics = {
        "service": SERVICE_ID, "design_account": design, "variants": VARIANTS, "bar": f"{champion} on Stage-1 scores",
        "summary": hypothesis.summary_line(judged), "passing_t1": [v["variant"] for v in judged if v["passes"]],
        "eligible": [k for k, v in eligible.items() if v["eligible"]], "eligibility": eligible, "rank_ic_63": ic,
        "t1": {v["variant"]: v["spans"] for v in judged},
        "equal_weight": {sid: vs_ew[(sid, "real", design)] for sid in variants}, "noise": noise, "planted": planted,
        "together": credit, "spy": S.spy_returns(panel, spans, design, net_broker),
        "fees_per_year": {f"{c['variant']} | {c['span']}": S.fees_per_year(c) for c in design_cells},
        "scored_share": {sid: float(np.isfinite(np.where(panel.tradable, score, np.nan)[panel.index_of(start):]).mean())
                         for sid, (score, _) in variants.items()},
        "cells": len(cells) + len(through_champion),
        "cannot_be_bought": [panel.symbols[j] for j in range(len(panel.symbols)) if panel.tradable[:, j].any()
                             and not np.isfinite(true_price[panel.tradable[:, j], j]).any()],
    }
    return metrics, report
