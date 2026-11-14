import io
import math
from unittest import mock

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from analytics import dcf
from analytics.frontier import simulate_frontier, weight_changes
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
