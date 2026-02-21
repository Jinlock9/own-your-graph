import os
import json
from collections import defaultdict
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import plotly.graph_objects as go
import plotly.utils
from itsdangerous import URLSafeTimedSerializer
from starlette.middleware.sessions import SessionMiddleware

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tracker.db")
PASSWORD     = os.getenv("TRACKER_PASSWORD", "changeme")
SECRET_KEY   = os.getenv("SECRET_KEY", "super-secret-key-change-in-prod")

# ── DB setup ──────────────────────────────────────────────────────────────────
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base         = declarative_base()

class Metric(Base):
    __tablename__ = "metrics"
    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String, unique=True, nullable=False)
    description  = Column(Text, nullable=True)
    unit         = Column(String, nullable=True)        # e.g. "cycles", "kg"
    lower_better = Column(Integer, default=1)           # 1 = lower is better (perf), 0 = higher is better
    graph_mode   = Column(String, default='single')     # 'single' or 'multi'
    series_names = Column(Text, nullable=True)          # comma-separated predefined series (multi mode)
    created_at   = Column(DateTime, default=datetime.utcnow)
    entries      = relationship("Entry", back_populates="metric", cascade="all, delete-orphan")

class Entry(Base):
    __tablename__ = "entries"
    id           = Column(Integer, primary_key=True, index=True)
    metric_id    = Column(Integer, ForeignKey("metrics.id"), nullable=False)
    value        = Column(Float, nullable=False)
    note         = Column(Text, nullable=True)
    series_label = Column(String, nullable=True)       # used in multi-graph mode
    recorded_at  = Column(DateTime, default=datetime.utcnow)
    metric       = relationship("Metric", back_populates="entries")

Base.metadata.create_all(bind=engine)

# ── Schema migrations (safe: silently skips if column already exists) ─────────
with engine.connect() as _conn:
    for _stmt in [
        "ALTER TABLE metrics ADD COLUMN graph_mode VARCHAR DEFAULT 'single'",
        "ALTER TABLE entries ADD COLUMN series_label VARCHAR",
        "ALTER TABLE metrics ADD COLUMN series_names TEXT",
    ]:
        try:
            _conn.execute(text(_stmt))
            _conn.commit()
        except Exception:
            pass  # column already exists

# ── Series colour palette ──────────────────────────────────────────────────────
SERIES_COLORS = ["#00e5ff", "#ff4081", "#00e676", "#ffab40", "#ce93d8", "#ff6e40", "#40c4ff"]

def _chart_layout(unit: str | None) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.6)",
        font=dict(color="#c8d0e0", family="JetBrains Mono, monospace"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)",
                   title=unit or "value"),
        margin=dict(l=50, r=30, t=30, b=50),
        hovermode="x unified",
    )

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_auth(request: Request):
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=303, headers={"Location": "/login"})

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if password == PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Wrong password"})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    metrics = db.query(Metric).order_by(Metric.created_at.desc()).all()
    summaries = []
    for m in metrics:
        entries = sorted(m.entries, key=lambda e: e.recorded_at)
        latest  = entries[-1].value if entries else None
        best    = min(e.value for e in entries) if entries and m.lower_better else (max(e.value for e in entries) if entries else None)
        first   = entries[0].value if entries else None
        pct     = ((latest - first) / first * 100) if (first is not None and first != 0 and latest is not None) else None

        # Per-series breakdown for multi-graph mode
        series_summary = []
        if (getattr(m, 'graph_mode', 'single') or 'single') == 'multi' and entries:
            groups: dict[str, list] = defaultdict(list)
            for e in entries:
                groups[e.series_label or 'default'].append(e)
            for label, s_entries in groups.items():
                s_ys     = [e.value for e in s_entries]
                s_latest = s_ys[-1]
                s_first  = s_ys[0]
                s_pct    = ((s_latest - s_first) / s_first * 100) if s_first != 0 else 0
                series_summary.append({
                    "label":    label,
                    "latest":   s_latest,
                    "pct":      s_pct,
                    "improved": (s_pct < 0) if m.lower_better else (s_pct > 0),
                    "count":    len(s_entries),
                })

        summaries.append({
            "metric": m, "count": len(entries),
            "latest": latest, "best": best, "first": first, "pct": pct,
            "series_summary": series_summary,
        })
    return templates.TemplateResponse("dashboard.html", {"request": request, "summaries": summaries})

# ── Create metric ─────────────────────────────────────────────────────────────
@app.get("/metrics/new", response_class=HTMLResponse)
async def new_metric_page(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("new_metric.html", {"request": request, "error": None})

@app.post("/metrics/new")
async def create_metric(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    unit: str = Form(""),
    lower_better: int = Form(1),
    graph_mode: str = Form("single"),
    series_names: str = Form(""),
    db: Session = Depends(get_db)
):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    existing = db.query(Metric).filter(Metric.name == name).first()
    if existing:
        return templates.TemplateResponse("new_metric.html", {"request": request, "error": f'Metric "{name}" already exists.'})
    if graph_mode not in ("single", "multi"):
        graph_mode = "single"
    # Normalise: strip whitespace, drop blanks, deduplicate while preserving order
    if graph_mode == "multi" and series_names:
        seen, cleaned = set(), []
        for s in series_names.split(","):
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                cleaned.append(s)
        series_names = ",".join(cleaned) or None
    else:
        series_names = None
    m = Metric(name=name, description=description or None, unit=unit or None,
               lower_better=lower_better, graph_mode=graph_mode, series_names=series_names)
    db.add(m)
    db.commit()
    db.refresh(m)
    return RedirectResponse(f"/metrics/{m.id}", status_code=302)

# ── Metric detail + chart ─────────────────────────────────────────────────────
@app.get("/metrics/{metric_id}", response_class=HTMLResponse)
async def metric_detail(request: Request, metric_id: int, db: Session = Depends(get_db)):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Metric not found")
    entries = sorted(m.entries, key=lambda e: e.recorded_at)

    chart_json         = None
    series_charts_json = "[]"
    stats              = {}
    series_labels      = []
    series_pct         = []   # per-series vs-first breakdown for multi mode

    # Predefined series for the log dropdown; fall back to names found in entries
    raw_names = getattr(m, 'series_names', None) or ""
    defined_series = [s.strip() for s in raw_names.split(",") if s.strip()]
    if not defined_series and (getattr(m, 'graph_mode', 'single') or 'single') == 'multi':
        defined_series = sorted(set(e.series_label for e in entries if e.series_label))

    if entries:
        ys_all   = [e.value for e in entries]
        best_val = min(ys_all) if m.lower_better else max(ys_all)
        first, last = ys_all[0], ys_all[-1]
        delta = last - first
        pct   = (delta / first * 100) if first != 0 else 0
        stats = {
            "count":    len(entries),
            "latest":   last,
            "best":     best_val,
            "first":    first,
            "delta":    delta,
            "pct":      pct,
            "improved": (delta < 0 if m.lower_better else delta > 0),
        }

        graph_mode = getattr(m, 'graph_mode', 'single') or 'single'

        if graph_mode == 'multi':
            # Collect existing series labels for datalist autocomplete
            series_labels = sorted(set(
                e.series_label for e in entries if e.series_label
            ))

            # Group entries by series_label
            groups: dict[str, list] = defaultdict(list)
            for e in entries:
                groups[e.series_label or 'default'].append(e)

            # ── Per-series vs-first breakdown ─────────────────────────────────
            for label, s_entries in groups.items():
                s_ys = [e.value for e in s_entries]
                s_first, s_last = s_ys[0], s_ys[-1]
                s_delta = s_last - s_first
                s_pct   = (s_delta / s_first * 100) if s_first != 0 else 0
                series_pct.append({
                    "label":    label,
                    "delta":    s_delta,
                    "pct":      s_pct,
                    "improved": (s_delta < 0 if m.lower_better else s_delta > 0),
                })

            # ── Combined chart (all series overlaid) ──────────────────────────
            fig_combined = go.Figure()
            for i, (label, s_entries) in enumerate(groups.items()):
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                xs = [e.recorded_at.strftime("%Y-%m-%d %H:%M") for e in s_entries]
                ys = [e.value for e in s_entries]
                fig_combined.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=7, color=color, line=dict(color="#0a0f1e", width=1.5)),
                    hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
                ))
            layout = _chart_layout(m.unit)
            layout["legend"] = dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
            )
            fig_combined.update_layout(**layout)
            chart_json = json.dumps(fig_combined, cls=plotly.utils.PlotlyJSONEncoder)

            # ── Separate chart per series ─────────────────────────────────────
            sep_list = []
            for i, (label, s_entries) in enumerate(groups.items()):
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                xs = [e.recorded_at.strftime("%Y-%m-%d %H:%M") for e in s_entries]
                ys = [e.value for e in s_entries]
                fig_sep = go.Figure()
                fig_sep.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=7, color=color, line=dict(color="#0a0f1e", width=1.5)),
                    hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
                ))
                s_best = min(ys) if m.lower_better else max(ys)
                fig_sep.add_hline(y=s_best, line_dash="dot", line_color="#ff4081",
                                  annotation_text=f"Best: {s_best}",
                                  annotation_position="bottom right")
                fig_sep.update_layout(**_chart_layout(m.unit))
                sep_list.append({
                    "label": label,
                    "chartData": json.loads(json.dumps(fig_sep, cls=plotly.utils.PlotlyJSONEncoder)),
                })
            series_charts_json = json.dumps(sep_list)

        else:
            # ── Single mode (original behaviour) ─────────────────────────────
            xs = [e.recorded_at.strftime("%Y-%m-%d %H:%M") for e in entries]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=ys_all,
                mode="lines+markers",
                name=m.name,
                line=dict(color="#00e5ff", width=2),
                marker=dict(size=7, color="#00e5ff", line=dict(color="#0a0f1e", width=1.5)),
                hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
            ))
            fig.add_hline(y=best_val, line_dash="dot", line_color="#ff4081",
                          annotation_text=f"Best: {best_val}", annotation_position="bottom right")
            fig.update_layout(**_chart_layout(m.unit))
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Ordered list of series label strings for template iteration
    series_charts_list = [s["label"] for s in json.loads(series_charts_json)] if series_charts_json != "[]" else []

    return templates.TemplateResponse("metric.html", {
        "request": request, "metric": m, "entries": list(reversed(entries)),
        "chart_json": chart_json, "stats": stats,
        "series_charts_json": series_charts_json,
        "series_charts_list": series_charts_list,
        "series_labels": series_labels,
        "series_pct": series_pct,
        "defined_series": defined_series,
    })

# ── Log entry ─────────────────────────────────────────────────────────────────
@app.post("/metrics/{metric_id}/entries")
async def add_entry(
    request: Request,
    metric_id: int,
    value: float = Form(...),
    note: str = Form(""),
    recorded_at: str = Form(""),
    series_label: str = Form(""),
    db: Session = Depends(get_db)
):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id).first()
    if not m:
        raise HTTPException(status_code=404)
    ts = datetime.fromisoformat(recorded_at) if recorded_at else datetime.utcnow()
    e  = Entry(metric_id=metric_id, value=value, note=note or None, recorded_at=ts,
               series_label=series_label or None)
    db.add(e)
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)

# ── Delete entry ──────────────────────────────────────────────────────────────
@app.post("/entries/{entry_id}/delete")
async def delete_entry(request: Request, entry_id: int, db: Session = Depends(get_db)):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    e = db.query(Entry).filter(Entry.id == entry_id).first()
    if not e:
        raise HTTPException(status_code=404)
    metric_id = e.metric_id
    db.delete(e)
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)

# ── Delete metric ─────────────────────────────────────────────────────────────
@app.post("/metrics/{metric_id}/delete")
async def delete_metric(request: Request, metric_id: int, db: Session = Depends(get_db)):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id).first()
    if not m:
        raise HTTPException(status_code=404)
    db.delete(m)
    db.commit()
    return RedirectResponse("/", status_code=302)
