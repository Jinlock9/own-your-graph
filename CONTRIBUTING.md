# Contributing to own-your-graph

Thanks for your interest! Contributions are welcome — whether it's a bug fix, a new feature, or improving the docs.

## Getting Started

1. Fork the repo and clone it locally
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dev server: `uvicorn main:app --reload --port 8000`
4. Make your changes and test them

## What to Work On

Check the [Issues](../../issues) tab for open tasks. Good first issues are labeled `good first issue`.

Ideas always welcome:
- New chart types (bar, scatter, histogram)
- CSV import/export
- Multiple users / per-user metrics
- REST API for programmatic logging (e.g. from a benchmark script)
- Dark/light theme toggle
- Docker support

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Write a clear description of what you changed and why
- If you're adding a feature, update the README accordingly
- No strict style guide, but try to match the existing code style

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS and Python version

## Questions

Just open an issue and ask. No question is too small.
