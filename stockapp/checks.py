"""Startup checks. Shown by `manage.py migrate` / `check`, so they appear in deploy logs."""
import os

from django.conf import settings
from django.core.checks import Warning, register


@register()
def persistent_database(app_configs, **kwargs):
    """Warn when a hosted deploy is using a throwaway SQLite file inside the container."""
    db = settings.DATABASES["default"]
    hosted = bool(os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_HOSTNAME"))
    if not hosted or "sqlite" not in db["ENGINE"] or os.environ.get("SQLITE_PATH"):
        return []
    return [Warning(
        "The app is running on Render with the container's built-in SQLite database. That file is "
        "deleted on every deploy and restart, so registered users and saved portfolios are lost.",
        hint="Create a free Postgres database (e.g. Neon) and set DATABASE_URL in the Render "
             "environment, or mount a persistent disk and set SQLITE_PATH to a file on it.",
        id="stockapp.W001",
    )]
