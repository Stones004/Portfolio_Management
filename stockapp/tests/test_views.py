from unittest import mock

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from analytics.market_data import DataError
from stockapp.models import Portfolio

from .fakes import fake_prices, make_company


def _fake_download(symbols, period="5y"):
    return fake_prices(sorted({s.upper() for s in symbols if s.upper() != "MISSING"}))


class AuthTests(TestCase):
    def test_pages_require_login(self):
        for name in ("home", "efficient_frontier", "dcf"):
            resp = self.client.get(reverse(name))
            self.assertEqual(resp.status_code, 302, name)
            self.assertIn(reverse("login"), resp["Location"])

    def test_register_and_login_pages(self):
        self.assertContains(self.client.get(reverse("login")), "Log in")
        resp = self.client.post(reverse("register"), {
            "username": "newbie", "password1": "a-Strong-pass-123", "password2": "a-Strong-pass-123"})
        self.assertRedirects(resp, reverse("home"))


class LoggedInTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("t", password="pw-12345-x")
        self.client.login(username="t", password="pw-12345-x")


@mock.patch("stockapp.views.portfolio.download_closes", side_effect=_fake_download)
class PortfolioViewTests(LoggedInTestCase):
    def _formset(self, rows):
        data = {"form-TOTAL_FORMS": str(len(rows)), "form-INITIAL_FORMS": "0", "action": "analyze"}
        for i, (s, w) in enumerate(rows):
            data[f"form-{i}-symbol"], data[f"form-{i}-weight"] = s, str(w)
        return data

    def test_get(self, _):
        self.assertContains(self.client.get(reverse("home")), "Holdings")

    def test_analyze_and_save(self, _):
        resp = self.client.post(reverse("home"), self._formset([("AAA", 60), ("BBB", 40), ("MISSING", 10)]))
        self.assertContains(resp, "Sharpe ratio")
        self.assertContains(resp, "No price data for MISSING")
        ctx = resp.context
        resp = self.client.post(reverse("home"), {"action": "save_portfolio", "portfolio_name": "Mine",
                                                  "symbols": ctx["symbols"], "weights": ctx["weights"]})
        p = Portfolio.objects.get(user=self.user, name="Mine")
        self.assertEqual(p.holdings.count(), 2)
        self.assertContains(self.client.get(f"{reverse('home')}?portfolio_id={p.id}"), 'selected>Mine')

    def test_other_users_portfolio_is_hidden(self, _):
        other = User.objects.create_user("o", password="pw-12345-x")
        p = Portfolio.objects.create(user=other, name="Secret")
        resp = self.client.get(f"{reverse('home')}?portfolio_id={p.id}")
        self.assertNotContains(resp, "Secret")

    def test_upload(self, _):
        from django.core.files.uploadedfile import SimpleUploadedFile
        f = SimpleUploadedFile("h.csv", b"symbol,weight\nAAA,1\nBBB,3\n")
        resp = self.client.post(reverse("home"), {"action": "upload_file", "file": f,
                                                  "form-TOTAL_FORMS": "1", "form-INITIAL_FORMS": "0"})
        self.assertContains(resp, "Loaded 2 stocks")


@mock.patch("stockapp.views.frontier.download_closes", side_effect=_fake_download)
class FrontierViewTests(LoggedInTestCase):
    def test_from_home_page(self, _):
        resp = self.client.post(reverse("efficient_frontier"), {"symbols": "AAA,BBB,CCC", "weights": "0.3,0.3,0.4"})
        self.assertContains(resp, "Max-Sharpe return")
        self.assertEqual(len(resp.context["weight_changes"]), 3)

    def test_needs_two_assets(self, _):
        resp = self.client.post(reverse("efficient_frontier"), {"symbols": "AAA", "weights": "1"})
        self.assertContains(resp, "At least two assets")


class DCFViewTests(LoggedInTestCase):
    def setUp(self):
        super().setUp()
        self.company = make_company()
        patches = [mock.patch("analytics.dcf.get_company_data", return_value=self.company),
                   mock.patch("analytics.wacc.get_company_data", return_value=self.company)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _value(self, resp):
        return resp.context["result"]["per_share_value"]

    def test_get(self):
        self.assertContains(self.client.get(reverse("dcf")), "Enter a ticker")

    def test_run_rerun_and_override(self):
        first = self.client.post(reverse("dcf"), {"ticker": "test.ns"})
        self.assertContains(first, "How the growth rate was estimated")
        fields = {k: v for k, _, _, v, _ in first.context["fields"]}
        same = self.client.post(reverse("dcf"), dict(fields, ticker="TEST.NS"))
        self.assertEqual(self._value(same), self._value(first))
        changed = self.client.post(reverse("dcf"), dict(fields, ticker="TEST.NS", discount_rate="20"))
        self.assertNotEqual(self._value(changed), self._value(first))

    def test_invalid_input_keeps_form(self):
        first = self.client.post(reverse("dcf"), {"ticker": "TEST.NS"})
        fields = {k: v for k, _, _, v, _ in first.context["fields"]}
        resp = self.client.post(reverse("dcf"), dict(fields, ticker="TEST.NS", terminal_growth="60"))
        self.assertContains(resp, "Terminal growth must be below")
        self.assertEqual(len(resp.context["fields"]), 7)

    def test_data_error_is_shown(self):
        with mock.patch("analytics.dcf.get_company_data", side_effect=DataError("No financial statements found")):
            resp = self.client.post(reverse("dcf"), {"ticker": "NOPE"})
        self.assertContains(resp, "No financial statements found")
