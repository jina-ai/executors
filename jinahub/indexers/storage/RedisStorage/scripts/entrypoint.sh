#!/bin/bash
redis-server --bind 0.0.0.0 --port 6379:6379 --daemonize yes && \
jina executor --uses config.yml $@
