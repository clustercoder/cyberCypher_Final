# src/utils/

Shared utility code used across backend modules.

## Files

- `logger.py`: Loguru-based structured logging setup

## Logging Flow

- human-readable colored logs to stderr (developer feedback)
- rotating daily file logs in `logs/` for traceability

## Why Centralized Logger

A single logger configuration avoids inconsistent formats and makes incident timelines easier to correlate across modules.
