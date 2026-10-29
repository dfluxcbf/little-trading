import json
from datetime import datetime, timedelta, timezone

import pytest

from little_trading.data import CsvMarketData, SyntheticMarketData, YahooMarketData
from little_trading.horizons import Horizon, spec_for
from little_trading.models import NewsItem
from little_trading.news import RssNews, aggregate_sentiment, score_text

# --- REQ-000000001: Read live and historical stock prices ---
# --- REQ-000000002: Subscribe to company news ---
# --- REQ-000000003: Trade on day, week and two-week horizons ---

CHART = json.dumps({"chart": {"result": [{
    "timestamp": [1700000000, 1700000300, 1700000600],
    "indicators": {"quote": [{"open": [1, 2, None], "high": [2, 3, None], "low": [1, 2, None],
                              "close": [2, 3, None], "volume": [10, 20, None]}]}}], "error": None}})

RSS = """<rss><channel>
<item><title>ACME beats estimates, shares surge</title><pubDate>Tue, 21 Oct 2026 08:00:00 GMT</pubDate><link>http://x/1</link></item>
<item><title>ACME faces lawsuit</title><pubDate>Mon, 20 Oct 2026 08:00:00 GMT</pubDate></item>
</channel></rss>"""


@pytest.mark.requirement("REQ-000000001")
def test_yahoo_parses_and_skips_null_bars():
    urls = []
    src = YahooMarketData(lambda u: urls.append(u) or CHART)
    bars = src.get_bars("ACME", spec_for("day"))
    assert [b.close for b in bars] == [2, 3]
    assert "ACME" in urls[0] and "interval=5m" in urls[0]


@pytest.mark.requirement("REQ-000000001")
def test_yahoo_error_raises():
    err = json.dumps({"chart": {"result": None, "error": {"code": "Not Found"}}})
    with pytest.raises(ValueError):
        YahooMarketData(lambda u: err).get_bars("ZZZ", spec_for("week"))


@pytest.mark.requirement("REQ-000000001")
def test_csv_source(tmp_path):
    (tmp_path / "ACME.csv").write_text("time,open,high,low,close,volume\n2026-01-01T00:00:00,1,2,1,2,5\n")
    bars = CsvMarketData(str(tmp_path)).get_bars("ACME", spec_for("week"))
    assert bars[0].close == 2 and bars[0].time.tzinfo is not None


@pytest.mark.requirement("REQ-000000001")
def test_synthetic_is_deterministic_per_symbol():
    s = SyntheticMarketData(50)
    a, b = s.get_bars("A", spec_for("week")), s.get_bars("A", spec_for("week"))
    assert a == b and a != s.get_bars("B", spec_for("week"))


@pytest.mark.requirement("REQ-000000002")
def test_sentiment_scoring():
    assert score_text("Profit surge and record growth") == 1.0
    assert score_text("Lawsuit and recall") == -1.0
    assert score_text("Company holds meeting") == 0.0


@pytest.mark.requirement("REQ-000000002")
def test_rss_news_and_recency_weighting():
    items = RssNews(lambda u: RSS).get_news("ACME")
    assert len(items) == 2 and items[0].symbol == "ACME"
    now = datetime(2026, 10, 21, 9, tzinfo=timezone.utc)
    s = aggregate_sentiment(items, now)
    assert 0 < s < 1  # newer positive item outweighs the older negative one
    assert aggregate_sentiment([], now) == 0.0


@pytest.mark.requirement("REQ-000000002")
def test_old_news_decays():
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    fresh = NewsItem("A", "profit surge", now)
    stale_neg = NewsItem("A", "lawsuit", now - timedelta(days=10))
    assert aggregate_sentiment([fresh, stale_neg], now) > 0.9


@pytest.mark.requirement("REQ-000000003")
def test_horizons_are_ordered():
    day, week, two = (spec_for(h) for h in Horizon)
    assert day.interval == "5m" and week.interval == two.interval == "1d"
    assert week.max_hold == 5 and two.max_hold == 10
    assert two.long > week.long
