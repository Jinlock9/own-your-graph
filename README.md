# own-your-graph 📈

> What you measure, you manage. Own the graph.

A lightweight, self-hosted personal metrics dashboard. Log any number you care about over time — kernel cycle counts, benchmark scores, body weight, sales numbers, whatever — and watch the trend.

The idea: whether you're an engineer optimizing a matmul kernel or a manager tracking a team metric, **everyone should own at least one graph**.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## Features

- **Multiple metrics** — each with its own interactive chart
- **Plotly charts** — zoom, pan, hover, best-value reference line
- **Stats at a glance** — latest value, best value, total change, % improvement
- **"Lower is better" or "Higher is better"** per metric
- **Log entries** with value, timestamp, and an optional note
- **Password protection** — simple single-password session auth
- **Postgres-ready** — swap SQLite for Postgres in one environment variable

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Jinlock9/own-your-graph.git
cd own-your-graph

# 2. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your password
export TRACKER_PASSWORD="your-password-here"
export SECRET_KEY="any-random-string-for-session-signing"

# 5. Run
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000, log in, and create your first metric.

---

## Environment Variables

| Variable           | Default                           | Description                    |
|--------------------|-----------------------------------|--------------------------------|
| `TRACKER_PASSWORD` | `changeme`                        | Login password                 |
| `SECRET_KEY`       | `super-secret-key-change-in-prod` | Session signing key            |
| `DATABASE_URL`     | `sqlite:///./tracker.db`          | SQLAlchemy database URL        |

---

## Deploying

### Switch to PostgreSQL

Just set the `DATABASE_URL` — no code changes needed:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/dbname"
```

### Railway / Fly.io / Render

1. Push your fork to GitHub
2. Connect your repo to the platform
3. Set the environment variables above
4. Deploy — done

---

## Example Use Cases

- **Compiler engineer**: track cycle counts of a matmul kernel across optimization attempts
- **Engineer**: track build times, test suite duration, binary size
- **Manager**: track weekly active users, team velocity, incident count
- **Anyone**: body weight, sleep hours, daily steps — anything with a number

---

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — backend framework
- [SQLAlchemy](https://www.sqlalchemy.org/) — ORM (SQLite by default, Postgres-ready)
- [Plotly](https://plotly.com/python/) — interactive charts
- [Jinja2](https://jinja.palletsprojects.com/) — HTML templates
- [Starlette Sessions](https://www.starlette.io/middleware/) — auth

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs are welcome!

---

## License

[MIT](LICENSE)
