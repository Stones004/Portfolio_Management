from django.apps import AppConfig


class StockappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stockapp'

    def ready(self):
        from . import checks  # noqa: F401  (registers the system checks)
