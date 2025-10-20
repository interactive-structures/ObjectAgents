#!/bin/bash


export OPENAI_API_KEY=$(cat ~/secrets/openai-api-key)

if command -v uv >/dev/null 2>&1; then
  exec uv run python src/main.py
else
  echo "Warning: 'uv' not found. Using system python: $(which python)" 1>&2
  exec python src/main.py
fi