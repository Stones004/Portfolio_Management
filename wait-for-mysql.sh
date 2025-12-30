#!/bin/sh
set -e

host="$DB_HOST"
shift

until nc -z "$host" 3306; do
  echo "Waiting for MySQL..."
  sleep 2
done

exec "$@"
