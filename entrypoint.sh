#!/bin/sh
set -e

echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting gunicorn..."
# DCF/efficient-frontier requests call Yahoo Finance and can exceed gunicorn's 30s default.
# Render sets $PORT; fall back to 8000 for local Docker.
exec gunicorn stocksite.wsgi:application --bind 0.0.0.0:${PORT:-8000} --timeout 120
