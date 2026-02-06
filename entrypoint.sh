#!/bin/sh
set -e

# Fix volume permissions for non-root user
chown -R pika:pika /app/data /app/documents 2>/dev/null || true
mkdir -p /app/data/hf_cache 2>/dev/null || true
chown -R pika:pika /app/data/hf_cache 2>/dev/null || true

# Drop privileges and exec the CMD
exec gosu pika "$@"
