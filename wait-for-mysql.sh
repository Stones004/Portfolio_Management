#!/bin/sh

echo "Waiting for MySQL at $DB_HOST:$DB_PORT..."

while ! nc -z "$DB_HOST" "$DB_PORT"; do
  sleep 1
done

echo "MySQL is available!"
exec "$@"
