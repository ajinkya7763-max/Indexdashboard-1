
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from risk_engine import filter_completed_candles, risk_check, performance_metrics

IST = ZoneInfo("Asia/Kolkata")


class ClosedCandleTests(unittest.TestCase):
    def test_excludes_forming_5m_candle(self):
        now = datetime(2026, 11, 24, 10, 3, tzinfo=IST)
        idx = pd.to_datetime(["2026-11-24 09:55", "2026-11-24 10:00"])
        idx = idx.tz_localize(IST)
        df = pd.DataFrame({"close": [1, 2]}, index=idx)
        out = filter_completed_candles(df, 5, now)
        self.assertEqual(list(out.close), [1])

    def test_includes_bar_when_full_interval_elapsed(self):
        now = datetime(2026, 11, 24, 10, 5, tzinfo=IST)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-11-24 10:00", tz=IST)])
        df = pd.DataFrame({"close": [2]}, index=idx)
        self.assertEqual(len(filter_completed_candles(df, 5, now)), 1)

    def test_daily_excludes_current_day(self):
        now = datetime(2026, 11, 24, 16, 0, tzinfo=IST)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-11-23", tz=IST),
                                pd.Timestamp("2026-11-24", tz=IST)])
        df = pd.DataFrame({"close": [1, 2]}, index=idx)
        out = filter_completed_candles(df, 1440, now, daily=True)
        self.assertEqual(list(out.close), [1])


class RiskTests(unittest.TestCase):
    def test_risk_limit_blocks_oversized_trade(self):
        result = risk_check(100, 95, 20, 10000, 0.5, 0, 2, 0, 1, 0, 3)
        self.assertFalse(result["allowed"])

    def test_daily_loss_limit_blocks_trade(self):
        result = risk_check(100, 99, 1, 10000, 1, -200, 1.5, 0, 1, 0, 3)
        self.assertFalse(result["allowed"])

    def test_passes_valid_trade(self):
        result = risk_check(100, 99, 1, 10000, 1, 0, 1.5, 0, 1, 0, 3)
        self.assertTrue(result["allowed"])


class PerformanceTests(unittest.TestCase):
    def test_metrics(self):
        m = performance_metrics(pd.DataFrame({"pnl": [100, -50, 150]}))
        self.assertEqual(m["trades"], 3)
        self.assertAlmostEqual(m["win_rate"], 66.6666667, places=4)
        self.assertEqual(m["net_pnl"], 200)
        self.assertEqual(m["expectancy"], 200/3)
        self.assertEqual(m["profit_factor"], 5)


if __name__ == "__main__":
    unittest.main()
