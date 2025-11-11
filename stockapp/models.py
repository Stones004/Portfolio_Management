from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolios')
    name = models.CharField(max_length=100, default='My Portfolio')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.name}"

class Holding(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='holdings')
    symbol = models.CharField(max_length=10)
    weight = models.FloatField()

    def __str__(self):
        return f"{self.symbol} ({self.weight})"
