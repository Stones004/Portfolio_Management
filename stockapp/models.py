from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolios')
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.name}"

class Holding(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='holdings')
    symbol = models.CharField(max_length=50)
    weight = models.FloatField()

    def __str__(self):
        return f"{self.symbol} ({self.weight})"


class Valuation(models.Model):
    """Latest DCF result a user ran for a ticker (one row per user + symbol)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='valuations')
    symbol = models.CharField(max_length=50)
    name = models.CharField(max_length=200, blank=True)
    currency = models.CharField(max_length=10, blank=True)
    model = models.CharField(max_length=10, blank=True)          # "fcff" or "fcfe"
    per_share_value = models.FloatField()
    current_price = models.FloatField(null=True, blank=True)
    upside = models.FloatField(null=True, blank=True)            # intrinsic / price - 1
    custom = models.BooleanField(default=False)                  # run with user-edited assumptions
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['user', 'symbol'], name='one_valuation_per_symbol')]
        ordering = ['symbol']

    def __str__(self):
        return f"{self.user.username} - {self.symbol} {self.per_share_value:.2f}"

    @property
    def verdict(self):
        if self.upside is None:
            return ""
        if self.upside > 0.10:
            return "Undervalued"
        if self.upside < -0.10:
            return "Overvalued"
        return "Fairly valued"

    def as_dict(self):
        return {"symbol": self.symbol, "name": self.name, "currency": self.currency, "model": self.model,
                "per_share_value": self.per_share_value, "current_price": self.current_price,
                "upside": self.upside, "verdict": self.verdict, "custom": self.custom,
                "updated_at": self.updated_at.isoformat()}
