import io
import math
from unittest import mock

import numpy as np
import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from analytics import dcf
from analytics.frontier import simulate_frontier, weight_changes
from analytics import market_data
from analytics.industry import classify
from analytics.portfolio import align_weights, portfolio_metrics
from stockapp.holdings import parse_csv_fields, read_holdings_file

from .fakes import fake_prices, make_company


class GrowthHelpersTests(SimpleTestCase):
    def test_cagr(self):
        self.assertAlmostEqual(dcf.cagr([121, 110, 100]), 0.10, places=6)
        self.assertTrue(math.isnan(dcf.cagr([100])))
        self.assertTrue(math.isnan(dcf.cagr([100, -5])))

    def test_blend_reweights_missing_estimates(self):
        blended, rows = dcf.blend_growth({"fundamental": 0.10, "historical": float("nan"), "analyst": 0.20})
        # weights 0.4 and 0.3 rescaled to 4/7 and 3/7
        self.assertAlmostEqual(blended, 0.10 * 4 / 7 + 0.20 * 3 / 7)
        self.assertEqual(len(rows), 2)

    def test_blend_clips_outliers(self):
        blended, _ = dcf.blend_growth({"fundamental": 5.0, "historical": 5.0, "analyst": 5.0})
        self.assertLessEqual(blended, dcf.BLENDED_GROWTH_BOUNDS[1])

    def test_longer_not_higher(self):
        rate, years = dcf.growth_schedule(0.08, 0.15)
        self.assertEqual((rate, years), (0.08, 5))
        rate, years = dcf.growth_schedule(0.20, 0.15)
        self.assertEqual(rate, 0.15)
        # 15% for `years` must reach at least the size of 20% for 5 years
        self.assertGreaterEqual(1.15 ** years, 1.20 ** 5 - 1e-9)
        self.assertLess(1.15 ** (years - 1), 1.20 ** 5)
        _, capped = dcf.growth_schedule(0.40, 0.08)
        self.assertEqual(capped, dcf.MAX_HIGH_GROWTH_YEARS)


class ClassificationTests(SimpleTestCase):
    def test_profiles(self):
        self.assertEqual(classify("Financial Services", "Banks - Regional"), "financial")
        self.assertEqual(classify("Financial Services", "Financial Data & Stock Exchanges"), "general")
        self.assertEqual(classify("Basic Materials", "Steel"), "cyclical")
        self.assertEqual(classify("Technology", "Software - Application"), "technology")
        self.assertEqual(classify("Utilities", "Utilities - Regulated Electric"), "utility")


class DCFEngineTests(SimpleTestCase):
    def _model(self, **kw):
        company = make_company(**kw)
        with mock.patch("analytics.dcf.get_company_data", return_value=company), \
                mock.patch("analytics.wacc.get_company_data", return_value=company):
            model = dcf.DCF_Valuation(company.ticker)
            model.wacc_model  # build while patched
        return model

    def test_fcff_runs_and_rerun_is_identical(self):
        m = self._model()
        res = m.run()
        self.assertEqual(res["model"], "fcff")
        self.assertGreater(res["per_share_value"], 0)
        self.assertEqual(len(res["growth_sources"]), 3)
        again = m.run(res["defaults"])
        self.assertAlmostEqual(again["per_share_value"], res["per_share_value"])

    def test_reinvestment_matches_growth(self):
        m = self._model()
        a = m.default_assumptions()
        roic = m.history["ROIC"].iloc[0]
        self.assertAlmostEqual(a["reinvestment_rate"], a["growth_rate"] / roic, places=6)
        self.assertLessEqual(a["terminal_growth"], m.cost_of_capital()["risk_free"])

    def test_fast_grower_gets_more_years_not_higher_rate(self):
        m = self._model(sector="Basic Materials", industry="Steel", analyst=0.45)
        a = m.default_assumptions()
        self.assertLessEqual(a["growth_rate"], m.profile["max_growth"] + 1e-12)
        self.assertGreater(a["high_growth_years"], dcf.BASE_HIGH_GROWTH_YEARS)

    def test_bank_uses_fcfe(self):
        m = self._model(sector="Financial Services", industry="Banks - Regional", financial=True)
        res = m.run()
        self.assertEqual(res["model"], "fcfe")
        self.assertIsNone(res["enterprise_value"])
        self.assertLessEqual(res["assumptions"]["reinvestment_rate"], 1.0)

    def test_validation(self):
        m = self._model()
        with self.assertRaises(dcf.DCFError):
            m.run({"terminal_growth": 0.5})
        with self.assertRaises(dcf.DCFError):
            m.run({"high_growth_years": 0, "fade_years": 0})


class CurrencyAndUnitsTests(SimpleTestCase):
    """Regression: INFY.NS reports in USD but trades in INR. When Yahoo's profile endpoint was
    rate-limited the statements were assumed to be INR and the value came out ~90x too small."""

    def _run(self, company):
        with mock.patch("analytics.dcf.get_company_data", return_value=company), \
                mock.patch("analytics.wacc.get_company_data", return_value=company):
            return dcf.DCF_Valuation(company.ticker).run()

    def test_usd_statements_are_converted_to_price_currency(self):
        inr = self._run(make_company())
        usd = self._run(make_company(reporting="USD", fx=90.0, statement_divisor=90.0))
        self.assertAlmostEqual(usd["per_share_value"], inr["per_share_value"], delta=0.01)
        self.assertEqual(usd["currency"], "INR")
        self.assertTrue(any("converted to INR" in w for w in usd["warnings"]))

    def test_unverified_currency_that_looks_wrong_raises(self):
        company = make_company(statement_divisor=90.0, reporting_assumed=True)
        with self.assertRaises(dcf.DCFError) as ctx:
            self._run(company)
        self.assertIn("currency", str(ctx.exception))

    def test_unverified_currency_that_looks_right_still_values(self):
        res = self._run(make_company(reporting_assumed=True))
        self.assertGreater(res["per_share_value"], 0)

    def test_extreme_value_vs_price_is_flagged(self):
        company = make_company()
        company.info["currentPrice"] = 200_000.0
        res = self._run(company)
        self.assertTrue(any("x the market price" in w for w in res["warnings"]))


class AnalystGrowthTests(SimpleTestCase):
    """Regression: Yahoo's revenue_estimate "growth" for RELIANCE.NS used a wrong year-ago base
    (+151%); growth is now computed from the consensus level vs reported revenue."""

    @staticmethod
    def _est(avg_0y, avg_1y, ccy="INR"):
        return pd.DataFrame({"avg": [avg_0y, avg_1y], "growth": [1.5, 0.8], "currency": [ccy, ccy]},
                            index=["0y", "+1y"])

    def test_uses_levels_not_yahoo_growth_column(self):
        c = make_company()                      # latest revenue 1,300e8
        g = c.growth_from_estimates(self._est(1_400e8, 1_573e8), "revenue")
        self.assertAlmostEqual(g, (1_573 / 1_300) ** 0.5 - 1, places=4)

    def test_converts_statement_currency(self):
        c = make_company(reporting="USD", fx=90.0, statement_divisor=90.0)
        g = c.growth_from_estimates(self._est(1_400e8, 1_573e8, ccy="INR"), "revenue")
        self.assertAlmostEqual(g, (1_573 / 1_300) ** 0.5 - 1, places=4)

    def test_inconsistent_estimates_are_ignored(self):
        c = make_company()
        self.assertIsNone(c.growth_from_estimates(self._est(9_000e8, 12_000e8), "revenue"))
        self.assertIsNone(c.growth_from_estimates(pd.DataFrame(), "revenue"))

    def test_eps_growth(self):
        c = make_company()
        c.income.loc["Net Income"] = [150e8, 130e8, 115e8, 100e8]   # EPS 15 on 1bn shares
        g = c.growth_from_estimates(self._est(16.0, 18.15), "earnings")
        self.assertAlmostEqual(g, (18.15 / 15.0) ** 0.5 - 1, places=4)


TIMESERIES_USD = ('{"timeseries": {"result": [{"meta": {}, "timestamp": [1], "annualTotalRevenue": '
                  '[{"asOfDate": "2025-03-31", "currencyCode": "USD", "reportedValue": {"raw": 1.0}}]}]}}')


class _FakeTicker:
    """Minimal yfinance.Ticker for a USD-reporting stock trading in INR, with "info" blocked."""
    def __init__(self, symbol):
        self.ticker = symbol
        company = make_company(reporting="USD", fx=90.0, statement_divisor=90.0)
        self.income_stmt, self.balance_sheet, self.cashflow = company.income, company.balance, company.cashflow
        self._data = mock.Mock()
        self._data.cache_get.return_value = mock.Mock(text=TIMESERIES_USD)

    @property
    def info(self):
        raise RuntimeError("401 Invalid Crumb")

    def get_history_metadata(self):
        return {"currency": "INR", "longName": "Fake Infosys", "regularMarketPrice": 1500.0}


class CompanyDataFallbackTests(SimpleTestCase):
    def test_blocked_profile_uses_timeseries_currency_and_search_sector(self):
        search = mock.Mock(quotes=[{"symbol": "FAKE.NS", "sector": "Technology",
                                    "industry": "Information Technology Services", "longname": "Fake"}])
        with mock.patch.object(market_data.yf, "Ticker", _FakeTicker), \
                mock.patch.object(market_data.yf, "Search", return_value=search), \
                mock.patch.object(market_data, "fx_rate", return_value=90.0) as fx:
            data = market_data.CompanyData("FAKE.NS")
        self.assertEqual(data.reporting_currency, "USD")
        self.assertFalse(data.reporting_currency_assumed)
        fx.assert_called_with("USD", "INR")
        self.assertEqual(data.fx, 90.0)
        self.assertEqual(data.sector, "Technology")
        self.assertEqual(classify(data.sector, data.industry), "technology")

    def test_unknown_currency_is_flagged(self):
        class NoCurrency(_FakeTicker):
            def __init__(self, symbol):
                super().__init__(symbol)
                self._data.cache_get.side_effect = RuntimeError("429")
        with mock.patch.object(market_data.yf, "Ticker", NoCurrency), \
                mock.patch.object(market_data.yf, "Search", side_effect=RuntimeError("429")):
            data = market_data.CompanyData("FAKE.NS")
        self.assertTrue(data.reporting_currency_assumed)
        self.assertEqual(data.reporting_currency, "INR")


class PortfolioTests(SimpleTestCase):
    def test_metrics(self):
        prices = fake_prices(["A", "B", "C"])
        p, w, skipped = align_weights(prices, ["A", "B", "Z"], [1, 1, 1])
        self.assertEqual(skipped, ["Z"])
        self.assertAlmostEqual(w.sum(), 1.0)
        out = portfolio_metrics(p, w)
        self.assertGreater(out["portfolio_volatility"], 0)
        self.assertGreaterEqual(out["cvar_95"], out["var_95"])

    def test_frontier(self):
        returns = fake_prices(["A", "B", "C"]).pct_change().dropna()
        sim = simulate_frontier(returns, n_portfolios=2000, current_weights=np.ones(3) / 3, seed=0)
        self.assertAlmostEqual(sim["best_weights"].sum(), 1.0)
        self.assertGreaterEqual(sim["best"]["sharpe"], sim["current"]["sharpe"] - 1e-9)
        self.assertLessEqual(sim["min_vol"]["volatility"], sim["best"]["volatility"] + 1e-12)
        rows = weight_changes({"A": 0.5}, sim["best_weights"].to_dict())
        self.assertEqual(len(rows), 3)


class HoldingsTests(SimpleTestCase):
    def test_csv_upload(self):
        f = SimpleUploadedFile("h.csv", b"Symbol,Weight\ninfy.ns,2\nTCS.NS,1\ninfy.ns,1\nBAD,\n")
        symbols, weights = read_holdings_file(f)
        self.assertEqual(symbols, ["INFY.NS", "TCS.NS"])
        self.assertEqual(weights, [3.0, 1.0])

    def test_bad_upload(self):
        with self.assertRaises(ValueError):
            read_holdings_file(SimpleUploadedFile("h.txt", b"x"))
        with self.assertRaises(ValueError):
            read_holdings_file(SimpleUploadedFile("h.csv", b"ticker,amount\nA,1\n"))

    def test_hidden_fields(self):
        self.assertEqual(parse_csv_fields("a, b", "0.5,0.5"), (["A", "B"], [0.5, 0.5]))
        with self.assertRaises(ValueError):
            parse_csv_fields("a,b", "1")
