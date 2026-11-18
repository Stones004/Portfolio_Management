from django.contrib import admin

from .models import Holding, Portfolio, Valuation


class HoldingInline(admin.TabularInline):
    model = Holding
    extra = 0


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "created_at")
    list_filter = ("user",)
    search_fields = ("name", "user__username")
    inlines = [HoldingInline]


@admin.register(Valuation)
class ValuationAdmin(admin.ModelAdmin):
    list_display = ("symbol", "user", "per_share_value", "current_price", "upside", "custom", "updated_at")
    list_filter = ("user", "custom")
    search_fields = ("symbol", "name", "user__username")
