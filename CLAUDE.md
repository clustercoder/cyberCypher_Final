# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Final round project for the **TQ Cyber Cypher Hackathon 2026**. This is a Python project (based on `.gitignore` configuration) in its early stages — no source code has been committed yet.

## Development Setup

The `.gitignore` is configured for Python with support for common tooling: `pip`/`uv`/`poetry`/`pdm`/`pipenv`, Django, Flask, pytest, mypy, ruff, and Jupyter notebooks. Set up a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # once a requirements file exists
```

## Commands

Commands will be established as the project develops. Common patterns to follow once files are present:

- **Lint:** `ruff check .` (ruff is configured in `.gitignore`)
- **Type check:** `mypy .`
- **Tests:** `pytest` (run a single test with `pytest path/to/test_file.py::test_name`)
