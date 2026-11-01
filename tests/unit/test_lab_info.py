"""Phase I, new information (REQ-000000031): the EDGAR client (no contact, no fetch; never above 10 requests a second;
cached; restricted issuers dropped before any request; the right addresses), 8-K item 2.02 events and the after-close rule,
the event study on a planted drift and on noise, point-in-time fundamentals with a restatement, future perturbation of the
event and fundamental features, a planted fundamental edge found by the training trial, the survivorship measures, the
three services end to end on a fabricated EDGAR (tests/unit/fixtures/edgar, served by a fake transport), their cards,
their isolation, and that no file of this phase holds an address. Numpy only, no network."""
from __future__ import annotations

import bisect
import json
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from little_trading import restricted
from little_trading.lab import cards, edgar, leakcheck
from little_trading.lab import earnings_i1 as E
from little_trading.lab import fundamentals_i2 as F
from little_trading.lab import survivorship_i3 as S
from little_trading.lab.hypothesis import training_trial
from little_trading.lab.info_cube import augmented_cube
from little_trading.lab.panel import Panel
from little_trading.lab.service import isolation
from little_trading.lab.service.context import LookRefused
from little_trading.lab.services import h_i1_edgar_earnings as I1
from little_trading.lab.services import h_i2_fundamentals as I2
from little_trading.lab.services import h_i3_survivorship as I3

from test_lab import business_days
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000031"
ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent / "fixtures" / "edgar"
CONTACT = "test-contact-token"
NY = E.NY


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        assert s > 0
        self.now += s


class FakeSEC:
    """Serves the fabricated fixtures by URL, as SEC's hosts would; anything else is a 404 (an OSError)."""

    def __init__(self, clock=None):
        self.calls, self.times, self.clock = [], [], clock

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if self.clock:
            self.times.append(self.clock())
        if url == edgar.TICKERS_URL:
            return (FIX / "company_tickers.json").read_bytes()
        m = re.fullmatch(r"https://data\.sec\.gov/(submissions|api/xbrl/companyfacts)/(CIK\d{10})\.json", url)
        if m:
            return (FIX / ("submissions" if m.group(1) == "submissions" else "companyfacts") / f"{m.group(2)}.json").read_bytes()
        m = re.fullmatch(r"https://www\.sec\.gov/Archives/edgar/data/(\d+)/(\d{18})/([\w.-]+)", url)
        if m:
            cik, acc, name = m.groups()
            if name.endswith("-index.htm"):
                return (FIX / "filing-index.htm").read_text().replace("CIKNUM", cik).replace("ACCDIR", acc).encode()
            if name == "ex991.htm":
                return (FIX / ("ex991-positive.htm" if int(acc) // 2 % 2 == 0 else "ex991-negative.htm")).read_bytes()
        raise OSError(f"404 {url}")


def client(tmp_path, transport=None, clock=None, **kw) -> edgar.Client:
    clock = clock or FakeClock()
    return edgar.Client(CONTACT, tmp_path / "cache", transport=transport or FakeSEC(), clock=clock, sleep=clock.sleep, **kw)


def cik_of(i: int) -> int:
    return 1000000000 + i


# --- the client ------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_client_refuses_to_exist_without_a_contact_and_says_where_to_set_it(tmp_path):
    sec = FakeSEC()
    for missing in (None, "", "   ", "a\nb"):
        with pytest.raises(edgar.ContactMissing) as err:
            edgar.Client(missing, tmp_path, transport=sec)
        assert "edgar_contact" in str(err.value) and "lab/service/config.json" in str(err.value)
    ctx = ctx_for(tmp_path / "lab", "h-i1-edgar-earnings", {}, "r0")
    with pytest.raises(edgar.ContactMissing):
        edgar.client_for(ctx, transport=sec)  # no lab config at all
    (tmp_path / "lab" / "config.json").write_text(json.dumps({"edgar_contact": ""}))
    with pytest.raises(edgar.ContactMissing):
        edgar.client_for(ctx, transport=sec)
    assert sec.calls == []
    (tmp_path / "lab" / "config.json").write_text(json.dumps({"edgar_contact": CONTACT}))
    c = edgar.client_for(ctx, transport=sec)
    assert c.user_agent == f"little-trading-lab {CONTACT}" and c.cache_dir == ctx.service_dir / "edgar-cache"
    # the contact is never read from the service's own (registered, committed) config
    ctx2 = ctx_for(tmp_path / "lab2", "h-i1-edgar-earnings", {"edgar_contact": CONTACT}, "r0")
    with pytest.raises(edgar.ContactMissing):
        edgar.client_for(ctx2, transport=sec)


@pytest.mark.requirement(REQ)
def test_a_burst_never_exceeds_ten_requests_in_any_second(tmp_path):
    for asked, cap in ((50.0, 10), (10.0, 10), (5.0, 5)):
        clock = FakeClock()
        sec = FakeSEC(clock)
        c = edgar.Client(CONTACT, tmp_path / f"c{asked}", transport=lambda u, s=sec: s(edgar.TICKERS_URL), rate=asked,
                         clock=clock, sleep=clock.sleep)
        assert c.limiter.rate == cap
        for i in range(120):
            c.get(edgar.submissions_url(i))
        t = np.array(sec.times)
        assert len(t) == 120 and np.all(np.diff(t) > 0)
        assert np.all(t[cap:] - t[:-cap] >= 1.0)  # any cap + 1 consecutive requests span at least one second
        per_window = max(int(((t >= x) & (t < x + 1.0)).sum()) for x in t)
        assert per_window <= cap
        assert t[-1] - t[0] >= (120 - cap) / cap  # and it is not slower than it has to be by much
        assert t[-1] - t[0] <= 120 / cap + 1
    assert edgar.RateLimiter(1000).rate == edgar.MAX_RATE and edgar.RateLimiter().rate == 5.0


@pytest.mark.requirement(REQ)
def test_a_document_is_fetched_once_and_then_read_from_the_cache(tmp_path):
    sec = FakeSEC()
    c = client(tmp_path, sec)
    a = c.get_json(edgar.submissions_url(cik_of(1)))
    b = c.get_json(edgar.submissions_url(cik_of(1)))
    assert a == b and sec.calls == [edgar.submissions_url(cik_of(1))] and c.fetched == sec.calls
    assert c.cache_path(edgar.submissions_url(cik_of(1))).is_file()
    c2 = client(tmp_path, sec)  # a new client on the same directory (the next run) reads the cache too
    c2.submissions(cik_of(1))
    assert len(sec.calls) == 1
    c2.submissions(cik_of(1), max_age_s=-1)  # a maximum age refreshes the growing JSON
    assert len(sec.calls) == 2
    with pytest.raises(OSError):
        c.get("https://www.sec.gov/Archives/edgar/data/1/000000000119000001/missing.htm")
    assert not c.cache_path("https://www.sec.gov/Archives/edgar/data/1/000000000119000001/missing.htm").exists()


@pytest.mark.requirement(REQ)
def test_restricted_issuers_are_dropped_before_any_request(tmp_path):
    sec = FakeSEC()
    rules = restricted.Restrictions({"S3": "test restriction"}, {r"\bblocked example\b": "test restriction by name"})
    c = client(tmp_path, sec, restrictions=rules)
    assert c.allowed(["S0", "S3", "S3.XX", "S5"]) == ["S0", "S5"] and sec.calls == []
    assert set(c.dropped) == {"S3", "S3.XX"}
    assert c.cik_for("S3") is None and sec.calls == []  # not even the ticker map for a restricted ticker
    assert c.cik_for("S0") == cik_of(0) and sec.calls == [edgar.TICKERS_URL]
    assert c.cik_for("S29") is None and "S29" in c.dropped  # dropped by its company name in the ticker map
    with pytest.raises(restricted.RestrictedSymbolError):
        c.submissions(1000000099)  # its CIK is refused too
    assert sec.calls == [edgar.TICKERS_URL]
    # a second listing of a restricted company (same CIK, another ticker root) is refused with it
    two = {"0": {"cik_str": 7, "ticker": "S3", "title": "Synthetic Seven (fabricated)"},
           "1": {"cik_str": 7, "ticker": "SVN", "title": "Synthetic Seven (fabricated)"},
           "2": {"cik_str": 8, "ticker": "S8", "title": "Synthetic Eight (fabricated)"}}
    calls = []
    c2 = edgar.Client(CONTACT, tmp_path / "c2", transport=lambda u: calls.append(u) or json.dumps(two).encode(),
                      clock=FakeClock(), sleep=lambda s: None, restrictions=rules)
    assert c2.cik_for("SVN") is None and "SVN" in c2.dropped and c2.cik_for("S8") == 8
    with pytest.raises(restricted.RestrictedSymbolError):
        c2.submissions(7)
    assert calls == [edgar.TICKERS_URL]


@pytest.mark.requirement(REQ)
def test_the_client_builds_sec_addresses_and_declares_the_contact():
    assert edgar.TICKERS_URL == "https://www.sec.gov/files/company_tickers.json"
    assert edgar.submissions_url(1234) == "https://data.sec.gov/submissions/CIK0000001234.json"
    assert edgar.submissions_url("CIK0000001234") == edgar.submissions_url(1234)
    assert edgar.companyfacts_url(1234) == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000001234.json"
    assert (edgar.archive_url(1234, "0001234567-19-000012", "ex991.htm")
            == "https://www.sec.gov/Archives/edgar/data/1234/000123456719000012/ex991.htm")
    for bad in (("19-000012", "a.htm"), ("0001234567-19-000012", "../x"), ("0001234567-19-000012", "a/b.htm")):
        with pytest.raises(ValueError):
            edgar.archive_url(1234, *bad)
    req = edgar.build_request(edgar.submissions_url(1234), f"little-trading-lab {CONTACT}")
    assert req.get_header("User-agent") == f"little-trading-lab {CONTACT}" and req.get_header("Host") == "data.sec.gov"
    with pytest.raises(ValueError):
        edgar.build_request("https://example.org/x.json", "ua")
    assert edgar.exhibit_from_index((FIX / "filing-index.htm").read_text()) == "ex991.htm"
    assert edgar.exhibit_from_index("<table><tr><td>EX-99.2</td><td><a href='x'>x</a></td></tr></table>") is None


# --- events and their timing -----------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_only_8k_item_202_filings_are_events_with_their_acceptance_time():
    sub = json.loads((FIX / "submissions" / f"CIK{cik_of(0):010d}.json").read_text())
    every = edgar.recent_filings(sub)
    ev = E.earnings_filings(sub)
    assert len(ev) == 15 and len(every) == 32  # 15 releases among 10-Qs, a 5.02 8-K and an amended 8-K
    assert all(a["acceptance"] < b["acceptance"] for a, b in zip(ev, ev[1:]))
    raw = {f["accessionNumber"]: f for f in every}
    for e in ev:
        f = raw[e["accession"]]
        assert f["form"] == "8-K" and "2.02" in f["items"].split(",")
        assert e["acceptance"].tzinfo is not None and e["acceptance"].astimezone(NY).strftime("%H:%M") in ("16:05", "08:00", "12:30")
    stamp = "2019-01-24T16:05:12.000Z"
    assert E.parse_acceptance(stamp).astimezone(NY).hour == 16  # SEC's clock is New York time despite the Z
    assert E.parse_acceptance(stamp, "UTC").astimezone(NY).hour == 11


@pytest.mark.requirement(REQ)
def test_an_event_after_the_close_or_on_a_non_session_counts_from_the_next_session():
    sessions = business_days(date(2019, 1, 21), 15)  # Monday 2019-01-21 on, weekends skipped
    row = {d: i for i, d in enumerate(sessions)}
    at = lambda d, h, m=0, s=0: datetime(d.year, d.month, d.day, h, m, s, tzinfo=NY)  # noqa: E731
    wed, fri = date(2019, 1, 23), date(2019, 1, 25)
    assert E.event_session(at(wed, 8), sessions) == row[wed]  # before the open: that session's close reflects it
    assert E.event_session(at(wed, 12, 30), sessions) == row[wed]
    assert E.event_session(at(wed, 15, 59, 59), sessions) == row[wed]
    assert E.event_session(at(wed, 16), sessions) == row[wed] + 1  # at or after 16:00: the next session
    assert E.event_session(at(wed, 16, 5), sessions) == row[wed] + 1
    assert E.event_session(at(fri, 17), sessions) == row[date(2019, 1, 28)]  # Friday evening -> Monday
    assert E.event_session(at(date(2019, 1, 26), 10), sessions) == row[date(2019, 1, 28)]  # Saturday -> Monday
    assert E.event_session(datetime(2019, 1, 23, 21, 30, tzinfo=timezone.utc), sessions) == row[wed] + 1  # 16:30 in New York
    assert E.event_session(at(sessions[-1], 17), sessions) is None  # past the data
    # the EAR is known at the close of e+1 and spans closes e-1 .. e+1
    close = np.array([[100.0], [100.0], [110.0], [121.0], [121.0]])
    mkt = np.array([10.0, 10.0, 10.0, 11.0, 11.0])
    x, end = E.ear(close, mkt, 0, 2)
    assert end == 3 and x == pytest.approx(121 / 100 - 11 / 10)
    assert np.isnan(E.ear(close, mkt, 0, 4)[0])  # its window runs past the data


@pytest.mark.requirement(REQ)
def test_the_tone_of_a_press_release_follows_its_words():
    pos = E.lexicon_tone(edgar.html_text((FIX / "ex991-positive.htm").read_bytes()))
    neg = E.lexicon_tone(edgar.html_text((FIX / "ex991-negative.htm").read_bytes()))
    assert pos > 0.5 and neg < -0.5 and E.lexicon_tone("the quarter ended") == 0.0
    assert "SYNTHETIC" not in edgar.html_text((FIX / "ex991-positive.htm").read_bytes())  # comments are not text


# --- the event study -------------------------------------------------------------------------------------------------------
def drift_world(drift: float, seed: int = 3, T: int = 1300, N: int = 60, every: int = 63):
    """Prices with an earnings event per company every `every` sessions; the surprise moves the price on the event
    session, and in the top fifth of surprises the price then drifts up by `drift` a session for 60 sessions."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((T, N)) * 0.015
    dates = business_days(date(2016, 1, 4), T)
    events = []
    for j in range(N):
        for e in range(70 + (j * 7) % every, T - 70, every):
            z = rng.standard_normal()
            r[e, j] += 0.03 * z
            if z > 0.8416:  # top quintile of a standard normal
                r[e + 2: e + 62, j] += drift
            hh = (8, 12, 17)[(j + e) % 3]
            when = datetime(dates[e].year, dates[e].month, dates[e].day, hh, tzinfo=NY)
            if hh == 17:  # filed after the close of the session before
                p = dates[e - 1]
                when = datetime(p.year, p.month, p.day, hh, tzinfo=NY)
            events.append({"symbol": f"S{j}", "acceptance": when, "tone": float(z + rng.normal(0, 0.5))})
    close = 100 * np.exp(np.cumsum(r, axis=0))
    mkt = 100 * np.exp(np.cumsum(r.mean(axis=1)))
    return dates, [f"S{j}" for j in range(N)], close, mkt, events


@pytest.mark.requirement(REQ)
def test_the_event_study_finds_a_planted_drift_and_nothing_in_noise():
    out = {}
    for drift in (0.0015, 0.0):
        dates, syms, close, mkt, events = drift_world(drift)
        placed = E.place_events(events, dates, syms, close, mkt)
        assert len(placed) == len(events) and all(ev["end"] == ev["session"] + 1 for ev in placed)
        out[drift] = {p: E.drift_study(placed, dates, close, mkt, p) for p in ("ear", "tone")}
    for p in ("ear", "tone"):
        hit = out[0.0015][p]["spread"][-1]
        assert hit["horizon"] == 60 and hit["diff"] > 0.04 and hit["t"] > 3, (p, hit)
        top = [b for b in out[0.0015][p]["by_quintile"] if b["horizon"] == 60]
        assert top[-1]["mean"] > max(b["mean"] for b in top[:-1])  # the top quintile drifts, the others do not
        null = out[0.0][p]["spread"][-1]
        assert abs(null["t"]) < 2, (p, null)
    hit = out[0.0015]["ear"]["spread"]
    assert hit[59]["diff"] > hit[19]["diff"] > hit[4]["diff"] > 0  # the drift builds over the 60 sessions


@pytest.mark.requirement(REQ)
def test_standard_errors_are_clustered_by_date():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    one = E.clustered_mean(y, np.arange(200))
    dup = E.clustered_mean(np.repeat(y, 4), np.repeat(np.arange(200), 4))  # four copies of each event on its date
    assert dup["mean"] == pytest.approx(one["mean"]) and dup["clusters"] == 200
    assert dup["se"] == pytest.approx(one["se"], rel=1e-9)  # copies on the same date add no information
    assert one["se"] == pytest.approx(y.std(ddof=1) / np.sqrt(200), rel=0.01)
    d = E.clustered_diff(np.r_[y, y + 1.0], np.r_[np.zeros(200, bool), np.ones(200, bool)],
                         np.r_[np.ones(200, bool), np.zeros(200, bool)], np.r_[np.arange(200), np.arange(200, 400)])
    assert d["diff"] == pytest.approx(1.0) and d["clusters"] == 400
    with pytest.raises(ValueError):
        E.cluster_keys([], [], "week")


# --- point-in-time fundamentals -----------------------------------------------------------------------------------------
def fy(y, val, filed, start=None):
    return {"start": start or f"{y}-01-01", "end": f"{y}-12-31", "val": val, "filed": filed}


@pytest.mark.requirement(REQ)
def test_a_fact_counts_from_the_session_after_its_filing_and_a_restatement_only_from_its_own_filing():
    doc = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        fy(2015, 80.0, "2016-02-12"), fy(2016, 100.0, "2017-02-10"),  # Friday 2017-02-10
        fy(2016, 50.0, "2017-06-15"),  # restated in a 10-K/A
        fy(2016, 999.0, "2016-12-01"),  # "filed" before its period ended: not a reported fact
        {"start": "2016-10-01", "end": "2016-12-31", "val": 7.0, "filed": "2017-02-10"}]}}}}}  # a quarter: not annual
    facts = F.parse_facts(doc)
    assert [r[2] for r in facts["net_income"]] == [80.0, 100.0, 50.0]
    sessions = business_days(date(2016, 2, 1), 400)
    at = {d: i for i, d in enumerate(sessions)}
    latest, prior = F.pit_values(facts["net_income"], sessions)
    assert np.isnan(latest[at[date(2016, 2, 12)]]) and latest[at[date(2016, 2, 15)]] == 80.0
    assert latest[at[date(2017, 2, 10)]] == 80.0  # the filing day itself still sees the old year
    assert latest[at[date(2017, 2, 13)]] == 100.0 and prior[at[date(2017, 2, 13)]] == 80.0
    assert latest[at[date(2017, 6, 15)]] == 100.0  # what was known before the restatement's filing
    assert latest[at[date(2017, 6, 16)]] == 50.0  # the restated value, from the session after
    assert np.isnan(F.pit_values(facts["net_income"], sessions, max_age_days=30)[0][at[date(2017, 6, 16)]])  # stale


@pytest.mark.requirement(REQ)
def test_the_fundamental_features_use_the_close_of_the_day_and_the_facts_as_filed():
    doc = json.loads((FIX / "companyfacts" / f"CIK{cik_of(0):010d}.json").read_text())
    facts = F.parse_facts(doc)
    sessions = business_days(date(2016, 1, 4), 900)
    close = np.full(len(sessions), 50.0)
    f = F.company_features(facts, sessions, close)
    t = sessions.index(date(2017, 3, 1))
    us = doc["facts"]["us-gaap"]
    v = lambda c, y, filed=None: [x["val"] for x in us[c]["units"]["USD"] if x["end"].startswith(str(y))  # noqa: E731
                                  and (filed is None or x["filed"] == filed)][0]
    shares = doc["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"][2]["val"]
    assert f["roa"][t] == pytest.approx(v("NetIncomeLoss", 2016) / v("Assets", 2016))
    assert f["gpa"][t] == pytest.approx(v("GrossProfit", 2016) / v("Assets", 2016))
    assert f["log_mcap"][t] == pytest.approx(np.log(50.0 * shares))
    assert f["bm"][t] == pytest.approx(v("StockholdersEquity", 2016) / (50.0 * shares))
    assert f["ey"][t] == pytest.approx(v("NetIncomeLoss", 2016) / (50.0 * shares))
    assert f["rev_growth"][t] == pytest.approx(v("Revenues", 2016) / v("Revenues", 2015) - 1)
    after = sessions.index(date(2017, 6, 16))
    assert f["roa"][after] == pytest.approx(v("NetIncomeLoss", 2016, "2017-06-15") / v("Assets", 2016))  # the restatement


FACT_SYMS = [f"S{i}" for i in range(10)]
FACT_DATES = business_days(date(2016, 1, 4), 900)
FACT_ORDS = [d.toordinal() for d in FACT_DATES]


def fixture_facts() -> dict:
    return {f"S{i}": F.parse_facts(json.loads((FIX / "companyfacts" / f"CIK{cik_of(i):010d}.json").read_text()))
            for i in range(10)}


def fundamentals_of(arrays: dict, facts: dict) -> np.ndarray:
    """The fundamental features on perturbable inputs: a fact filed on (or just before) session row r is scaled by
    fact_scale[r] (dropped when NaN) and dropped when r is past the data, so the check perturbs later filings too."""
    close = arrays["close"]
    T = len(close)
    scale = arrays["fact_scale"]
    out = {}
    for sym, fields in facts.items():
        out[sym] = {}
        for name, rows in fields.items():
            kept = []
            for end, start, val, filed in rows:
                r = bisect.bisect_left(FACT_ORDS, filed)
                if r < T and np.isfinite(scale[r]):
                    kept.append((end, start, val * scale[r], filed))
            out[sym][name] = kept
    return F.panel_features(FACT_DATES[:T], FACT_SYMS, close, out)


@pytest.mark.requirement(REQ)
def test_fundamental_features_never_change_when_later_prices_or_filings_change():
    rng = np.random.default_rng(5)
    T = len(FACT_DATES)
    arrays = {"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, 10)), axis=0)), "fact_scale": np.ones(T)}
    facts = fixture_facts()
    compute = lambda a: fundamentals_of(a, facts)  # noqa: E731
    cuts = (300, 450, 700)
    full = compute(arrays)
    for cut in cuts:
        assert np.isfinite(full[cut]).sum() >= 50  # the check looks at real values at every cut
    leakcheck.assert_no_lookahead(compute, arrays, cuts)
    # filings dated after the cut: a new year and a restatement of an old one change nothing up to the cut
    cut = 450
    later = FACT_DATES[cut + 1].toordinal()
    more = {s: {k: list(v) for k, v in f.items()} for s, f in facts.items()}
    old_end = date(2016, 12, 31).toordinal()
    for s in more:
        more[s]["net_income"].append((old_end, date(2016, 1, 1).toordinal(), -1e12, later))
        more[s]["assets"].append((FACT_DATES[cut].toordinal(), None, 1.0, FACT_DATES[cut + 3].toordinal()))
        more[s]["net_income"].sort(key=lambda r: (r[3], r[0]))
    moved = fundamentals_of(arrays, more)
    np.testing.assert_array_equal(np.isnan(moved[: cut + 1]), np.isnan(full[: cut + 1]))
    np.testing.assert_allclose(moved[: cut + 1][np.isfinite(full[: cut + 1])], full[: cut + 1][np.isfinite(full[: cut + 1])])
    assert not np.allclose(np.nan_to_num(moved[cut + 2:]), np.nan_to_num(full[cut + 2:]))  # and they do count afterwards
    # a filing on the cut's own date is not usable at the cut
    same = {s: {k: list(v) for k, v in f.items()} for s, f in facts.items()}
    for s in same:
        same[s]["assets"] = sorted(same[s]["assets"] + [(FACT_DATES[cut].toordinal() - 1, None, 1.0, FACT_DATES[cut].toordinal())],
                                   key=lambda r: (r[3], r[0]))
    got = fundamentals_of(arrays, same)
    np.testing.assert_allclose(got[: cut + 1], full[: cut + 1], equal_nan=True)
    assert not np.allclose(got[cut + 1], full[cut + 1], equal_nan=True)


def event_arrays(T=700, N=20, seed=4) -> tuple[dict, list[date], list[str], list[dict]]:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, N)), axis=0))
    a = {"open": close * np.exp(rng.normal(0, 0.003, (T, N))), "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": np.full((T, N), 1e7), "market_close": close.mean(axis=1)}
    dates = business_days(date(2016, 1, 4), T)
    events = []
    for j in range(N):
        for e in range(80 + 3 * j, T - 5, 40):
            hh = (8, 16, 12)[e % 3]
            events.append({"symbol": f"S{j}", "acceptance": datetime(dates[e].year, dates[e].month, dates[e].day, hh, 30, tzinfo=NY),
                           "tone": float(rng.normal())})
    return a, dates, [f"S{j}" for j in range(N)], events


def event_cube(arrays: dict, dates, syms, events) -> np.ndarray:
    """The trees' input of H-I1 (ranked price features + ranked event features) from perturbable price arrays."""
    T = len(arrays["close"])
    p = Panel.from_arrays(dates[:T], syms, arrays["open"], arrays["high"], arrays["low"], arrays["close"], arrays["volume"],
                          arrays["market_close"])
    placed = E.place_events(events, p.dates, p.symbols, p.close, p.market_close)
    feats = E.event_features(placed, T, len(syms))
    return np.concatenate([feats, augmented_cube(p, feats)], axis=-1)


@pytest.mark.requirement(REQ)
def test_event_features_never_change_when_later_prices_or_filings_change():
    a, dates, syms, events = event_arrays()
    compute = lambda x: event_cube(x, dates, syms, events)  # noqa: E731
    cuts = (250, 400, 550)
    full = compute(a)
    for cut in cuts:
        assert np.isfinite(full[cut, :, 0]).sum() >= 10 and np.isfinite(full[cut, :, 1]).sum() >= 10
    leakcheck.assert_no_lookahead(compute, a, cuts)
    cut = 400
    d = dates[cut]
    late = [{"symbol": s, "acceptance": datetime(d.year, d.month, d.day, 16, 1, tzinfo=NY), "tone": 9.0} for s in syms]
    late += [{"symbol": s, "acceptance": datetime(d.year, d.month, d.day, 9, 0, tzinfo=NY) + timedelta(days=3), "tone": -9.0}
             for s in syms]
    moved = event_cube(a, dates, syms, events + late)
    np.testing.assert_allclose(moved[: cut + 1], full[: cut + 1], equal_nan=True)
    # filed after the close of the cut: its event session is cut + 1, so its window (closes cut .. cut + 2) ends at cut + 2
    np.testing.assert_allclose(moved[cut + 1], full[cut + 1], equal_nan=True)
    assert not np.allclose(moved[cut + 2:, :, 1], full[cut + 2:, :, 1], equal_nan=True)
    # an event in the session's own hours is known once its window ends: the close after the event session
    pre = [{"symbol": "S0", "acceptance": datetime(d.year, d.month, d.day, 10, tzinfo=NY), "tone": 5.0}]
    got = event_cube(a, dates, syms, events + pre)
    assert got[cut, 0, 1] == full[cut, 0, 1] and got[cut + 1, 0, 1] == 5.0


# --- a planted fundamental edge through the training trial ---------------------------------------------------------------
def fundamental_world(edge: float, seed: int = 2, T: int = 1350, N: int = 40):
    """Each year every company files its net income (assets 1e9) on 15 January; from the session after the filing its
    daily return drifts by `edge` x its profitability rank for a year. Price-only features cannot see it."""
    rng = np.random.default_rng(seed)
    dates = business_days(date(2016, 10, 27), T)
    ords = np.array([d.toordinal() for d in dates])
    r = rng.normal(0, 0.015, (T, N))
    facts = {}
    for j in range(N):
        facts[f"S{j}"] = {"net_income": [], "assets": []}
    for y in range(2016, 2023):
        filed = date(y, 1, 15).toordinal()
        x = rng.uniform(-0.5, 0.5, N)
        rows = (ords > filed) & (ords <= date(y + 1, 1, 15).toordinal())
        r[rows] += edge * x
        for j in range(N):
            end = date(y - 1, 12, 31).toordinal()
            facts[f"S{j}"]["net_income"].append((end, date(y - 1, 1, 1).toordinal(), 1e8 * (1 + x[j]), filed))
            facts[f"S{j}"]["assets"].append((end, None, 1e9, filed))
    close = 100 * np.exp(np.cumsum(r, axis=0))
    syms = [f"S{j}" for j in range(N)]
    p = Panel.from_arrays(dates, syms, np.vstack([close[:1], close[:-1]]), close * 1.01, close * 0.99, close,
                          np.full((T, N), 1e7), close.mean(axis=1))
    return p, F.panel_features(dates, syms, p.close, facts)


@pytest.mark.requirement(REQ)
def test_the_fundamentals_variant_finds_a_planted_fundamental_edge_and_nothing_in_noise():
    cfg = {**I2.DEFAULTS, "learner": "ridge", "params": {"alpha": 1.0}}
    out = {}
    for edge in (0.002, 0.0):
        p, fund = fundamental_world(edge)
        assert np.isfinite(fund[p.index_of(date(2019, 6, 3)), :, 0]).all()
        res = training_trial(p, I2.makers(cfg, fund), date(2019, 1, 1), date(2021, 6, 30), I2.BASELINE, seeds=(0,))
        out[edge] = res["variants"][I2.VARIANT]
    assert out[0.002]["passes"] and out[0.002]["diff"]["mean"] > 0.03, out[0.002]["diff"]
    assert not out[0.0]["passes"] and abs(out[0.0]["diff"]["t"]) < 2, out[0.0]["diff"]


# --- survivorship -------------------------------------------------------------------------------------------------------
def late_world(premium: float, seed: int = 7, T: int = 900, N: int = 40, late: int = 12):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.015, (T, N))
    first = np.zeros(N, int)
    first[:late] = rng.integers(200, 500, late)
    for j in range(late):
        r[first[j]:, j] += premium
    close = 100 * np.exp(np.cumsum(r, axis=0))
    for j in range(late):
        close[: first[j], j] = np.nan
    dates = business_days(date(2018, 6, 1), T)
    market = 100 * np.exp(np.cumsum(r[:, late:].mean(axis=1)))
    return Panel.from_arrays(dates, [f"S{j}" for j in range(N)], close, close * 1.01, close * 0.99, close,
                             np.full((T, N), 1e7), market), first


@pytest.mark.requirement(REQ)
def test_survivorship_measures_find_planted_late_winners_and_bound_with_stated_assumptions():
    p, first = late_world(0.002)
    lo, hi = p.index_of(date(2019, 1, 1)), len(p.dates) - 1
    cov = S.coverage(p, lo, hi)
    assert cov[0]["share"] < 1.0 and cov[-1]["share"] == 1.0 and [c["date"][:4] for c in cov] == ["2019", "2020", "2021"]
    assert all(np.isfinite(c["share"]) for c in cov)
    assert list(S.first_rows(p)) == list(first)
    ent = S.entrants(p, lo, hi)
    assert ent["late"] >= 8 and ent["present"] == 28 and ent["mean_gap"] > 0.2 and ent["mean_gap"] > 2 * ent["se_gap"]
    null = S.entrants(late_world(0.0)[0], lo, hi)
    assert abs(null["mean_gap"]) < 2 * null["se_gap"]
    bound = S.exit_bound(5.0)
    assert len(bound) == 12 and max(b["per_year"] for b in bound) == pytest.approx(0.035)
    assert min(b["per_year"] for b in bound) == 0.0 and bound[-1]["cumulative"] == pytest.approx(1.035 ** 5 - 1)
    ew = S.ew_minus_market(p, lo, hi)
    assert [r["year"] for r in ew] == [2019, 2020, 2021] and all(np.isfinite(r["diff"]) for r in ew)
    note = S.report((date(2019, 1, 1), date(2023, 12, 31)), cov, ent, ew, bound)
    assert "assumptions (not data)" in note and "not free" in note


# --- the services end to end -------------------------------------------------------------------------------------------------
SHIFT = 3276  # 468 weeks back from 2025-01-02: the fixture's prices cover 2016 to mid-2019 (the first development half)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-i")
    live = root / "live.db"
    make_live_db(live, T=900, N=30, extra=())
    con = sqlite3.connect(live)
    con.execute(f"UPDATE prices SET day = date(day, '-{SHIFT} days')")
    con.commit()
    con.close()
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(root / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    (root / "lab" / "config.json").write_text(json.dumps({"edgar_contact": CONTACT}))
    (root / "restricted.txt").write_text("S3\nname: blocked example\n")
    return root


@pytest.fixture()
def sec(lab, monkeypatch):
    clock = FakeClock()
    fake = FakeSEC(clock)
    monkeypatch.setenv(restricted.USER_FILE_ENV, str(lab / "restricted.txt"))
    for mod in (I1, I2):
        monkeypatch.setattr(mod, "TRANSPORT", fake)
        monkeypatch.setattr(mod, "CLOCK", clock)
        monkeypatch.setattr(mod, "SLEEP", clock.sleep)
    return fake


def table(lab: Path, sid: str, sql: str) -> list[tuple]:
    db = sqlite3.connect(lab / "lab" / "services" / sid / "service.db")
    try:
        return db.execute(sql).fetchall()
    finally:
        db.close()


def check_result(ctx, variants: int) -> dict:
    r = ctx._result
    assert r["variants"] == variants and r["data_hash"] and r["window"] == "development"
    json.dumps(r["metrics"])
    return r["metrics"]


@pytest.mark.requirement(REQ)
def test_the_h_i1_service_studies_the_fixture_events_and_pairs_the_event_trees(lab, sec):
    cfg = {"learner": "ridge", "seeds": [0], "train": True, "max_horizon": 10}
    ctx = ctx_for(lab / "lab", I1.SERVICE_ID, cfg, "i1")
    I1.run(ctx)
    assert not any(f"CIK{cik_of(3):010d}" in u or f"/data/{cik_of(3)}/" in u for u in sec.calls)  # restricted: no request
    assert not any("1000000099" in u for u in sec.calls)
    fetch = dict((s, st) for s, st in table(lab, I1.SERVICE_ID, "SELECT symbol, status FROM i1_fetch"))
    assert fetch["S3"].startswith("restricted") and fetch["S29"].startswith("restricted by name")
    assert fetch["S0"] == "ok" and fetch["S15"] == "no CIK in the ticker map"
    events = table(lab, I1.SERVICE_ID, "SELECT symbol, session, window_end, ear, tone, ear_quintile, in_window FROM i1_events")
    assert {e[0] for e in events} == {f"S{i}" for i in range(10)} - {"S3"}
    inside = [e for e in events if e[6]]
    assert len(inside) >= 10 and all(e[2] >= "2019-01-01" and e[3] is not None and e[5] in range(1, 6) for e in inside)
    assert all(e[1] < e[2] for e in events if e[2])  # known only after the event session's next close
    assert {np.sign(e[4]) for e in events if e[4] is not None} == {-1.0, 1.0}  # both press releases were read
    drift = table(lab, I1.SERVICE_ID, "SELECT proxy, horizon, quintile FROM i1_drift")
    assert len(drift) == 2 * 10 * 5 and {d[0] for d in drift} == {"ear", "tone"}
    assert len(table(lab, I1.SERVICE_ID, "SELECT * FROM i1_spread")) == 2 * 10
    verdicts = table(lab, I1.SERVICE_ID, "SELECT family, variant, against FROM trial_verdicts ORDER BY variant")
    assert verdicts == [("earnings", "trees+events", "trees-rank5"), ("earnings", "trees-rank5", "trees-rank5")]
    m = check_result(ctx, I1.VARIANTS)
    assert m["requests"] == len(sec.calls) and "S3" in m["restricted_dropped"] and set(m["drift"]) == {"ear", "tone"}
    assert m["trees"]["learner"] == "ridge" and m["trees"]["diff"] is not None
    before = len(sec.calls)
    ctx2 = ctx_for(lab / "lab", I1.SERVICE_ID, {**cfg, "train": False}, "i1-again")
    I1.run(ctx2)
    assert len(sec.calls) == before and ctx2._result["metrics"]["requests"] == 0  # everything came from the cache
    assert "trees" not in ctx2._result["metrics"]
    with pytest.raises(LookRefused):
        I1.run(ctx_for(lab / "lab", I1.SERVICE_ID, {"window": "holdout"}, "i1-holdout"))
    no_contact = ctx_for(lab / "lab", I1.SERVICE_ID, cfg, "i1-nocontact")
    no_contact.lab_setting = lambda key: None
    with pytest.raises(edgar.ContactMissing):
        I1.run(no_contact)


@pytest.mark.requirement(REQ)
def test_the_h_i2_service_builds_point_in_time_fundamentals_and_pairs_them_with_price_only(lab, sec):
    ctx = ctx_for(lab / "lab", I2.SERVICE_ID, {"learner": "ridge", "seeds": [0]}, "i2")
    out = I2.run(ctx)
    assert not any(f"CIK{cik_of(3):010d}" in u for u in sec.calls)
    cov = dict(table(lab, I2.SERVICE_ID, "SELECT symbol, status FROM i2_coverage"))
    assert len(cov) == 30 and cov["S3"].startswith("restricted") and cov["S0"] == "ok"
    assert sum(v == "ok" for v in cov.values()) == 9
    feat = table(lab, I2.SERVICE_ID, "SELECT feature, coverage FROM i2_feature_ic")
    assert [f[0] for f in feat] == list(F.FEATURES) and all(0 < f[1] < 1 for f in feat)
    verdicts = table(lab, I2.SERVICE_ID, "SELECT family, variant, against FROM trial_verdicts ORDER BY variant")
    assert verdicts == [("fundamentals", I2.VARIANT, I2.BASELINE), ("fundamentals", I2.BASELINE, I2.BASELINE)]
    m = check_result(ctx, I2.VARIANTS)
    assert m["with_facts"] == 9 and m["learner"] == "ridge" and m["diff"] is not None
    syms = out["trial"]["scores"][I2.BASELINE].shape[1]
    assert out["features"].shape[1:] == (syms, len(F.FEATURES))
    with pytest.raises(LookRefused):
        I2.run(ctx_for(lab / "lab", I2.SERVICE_ID, {"window": "holdout"}, "i2-holdout"))


@pytest.mark.requirement(REQ)
def test_the_h_i3_service_writes_the_bound_and_its_assumptions(lab):
    ctx = ctx_for(lab / "lab", I3.SERVICE_ID, {}, "i3")
    I3.run(ctx)
    kinds = table(lab, I3.SERVICE_ID, "SELECT kind, note FROM i3_bound")
    assert len(kinds) == 12 and all(k == "assumption" and "not data" in n for k, n in kinds)
    cov = table(lab, I3.SERVICE_ID, "SELECT day, share FROM i3_coverage")
    assert cov and cov[0][0] >= "2019-01-01" and all(s == 1.0 for _, s in cov)  # the fixture's names all start together
    assert table(lab, I3.SERVICE_ID, "SELECT year FROM i3_ew_market") == [(2019,)]
    note = (lab / "lab" / "services" / I3.SERVICE_ID / "survivorship.md").read_text()
    assert "Bound from assumptions (not data)" in note
    m = check_result(ctx, I3.VARIANTS)
    assert m["late_entrants"] == 0 and m["bound_worst_per_year"] == pytest.approx(0.035)
    with pytest.raises(LookRefused):
        I3.run(ctx_for(lab / "lab", I3.SERVICE_ID, {"window": "holdout"}, "i3-holdout"))


# --- cards, isolation, no addresses ------------------------------------------------------------------------------------------
CARDS = {"h-i1-edgar-earnings": I1, "h-i2-fundamentals": I2, "h-i3-survivorship": I3}
CREATED = ["src/little_trading/lab/edgar.py", "src/little_trading/lab/earnings_i1.py",
           "src/little_trading/lab/fundamentals_i2.py", "src/little_trading/lab/survivorship_i3.py",
           "src/little_trading/lab/info_cube.py", "src/little_trading/lab/services/h_i1_edgar_earnings.py",
           "src/little_trading/lab/services/h_i2_fundamentals.py", "src/little_trading/lab/services/h_i3_survivorship.py",
           "tests/unit/test_lab_info.py", "docs/lab/edgar.md", *(f"docs/lab/cards/{c}.md" for c in CARDS)]


@pytest.mark.requirement(REQ)
def test_the_cards_are_complete_and_name_their_leak_tests():
    for sid, mod in CARDS.items():
        text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
        assert cards.check(text) == [], sid
        f = cards.parse(text)
        assert f["variants"] == str(mod.VARIANTS) and f["window"] == "development 2019-2023"
        assert f["family"] == mod.FAMILY and f["baseline"] and f["notes"] and f["metric"]
        named = [w.split("::")[1].rstrip(",.()") for w in f["leak test"].split() if "test_lab_info.py::" in w]
        assert named and all(t in globals() for t in named), sid


@pytest.mark.requirement(REQ)
def test_the_services_are_isolated_and_import_no_heavy_package_at_module_level():
    for mod in (I1, I2, I3):
        assert isolation.violations(mod.__name__) == [], mod.__name__
    for name in ("edgar", "earnings_i1", "fundamentals_i2", "survivorship_i3", "info_cube"):
        assert isolation.violations(f"little_trading.lab.{name}") == [], name
    for path in CREATED:
        if path.endswith(".py") and path.startswith("src/"):
            top = [ln for ln in (ROOT / path).read_text().splitlines() if ln.startswith(("import ", "from "))]
            assert not any(pkg in ln for ln in top for pkg in ("transformers", "torch", "sklearn", "lightgbm")), path
    assert I1.TRANSPORT is None and I2.TRANSPORT is None  # the network unless a test replaces the hook


@pytest.mark.requirement(REQ)
def test_no_file_of_this_phase_contains_an_address():
    at = chr(64)
    pattern = re.compile(r"[A-Za-z0-9._%+-]+" + at + r"[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
    files = [ROOT / p for p in CREATED] + sorted(FIX.rglob("*"))
    assert len([f for f in files if f.is_file()]) >= len(CREATED) + 20
    for f in files:
        if f.is_file():
            assert not pattern.search(f.read_text(errors="replace")), f
