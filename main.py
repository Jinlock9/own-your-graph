import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text
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
    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    unit        = Column(String, nullable=True)        # e.g. "cycles", "kg"
    lower_better= Column(Integer, default=1)           # 1 = lower is better (perf), 0 = higher is better
    created_at  = Column(DateTime, default=datetime.utcnow)
    entries     = relationship("Entry", back_populates="metric", cascade="all, delete-orphan")

class Entry(Base):
    __tablename__ = "entries"
    id         = Column(Integer, primary_key=True, index=True)
    metric_id  = Column(Integer, ForeignKey("metrics.id"), nullable=False)
    value      = Column(Float, nullable=False)
    note       = Column(Text, nullable=True)
    recorded_at= Column(DateTime, default=datetime.utcnow)
    metric     = relationship("Metric", back_populates="entries")

Base.metadata.create_all(bind=engine)

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
        summaries.append({"metric": m, "count": len(entries), "latest": latest, "best": best})
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
    db: Session = Depends(get_db)
):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    existing = db.query(Metric).filter(Metric.name == name).first()
    if existing:
        return templates.TemplateResponse("new_metric.html", {"request": request, "error": f'Metric "{name}" already exists.'})
    m = Metric(name=name, description=description or None, unit=unit or None, lower_better=lower_better)
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

    chart_json = None
    stats      = {}
    if entries:
        xs = [e.recorded_at.strftime("%Y-%m-%d %H:%M") for e in entries]
        ys = [e.value for e in entries]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=m.name,
            line=dict(color="#00e5ff", width=2),
            marker=dict(size=7, color="#00e5ff", line=dict(color="#0a0f1e", width=1.5)),
            hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>"
        ))
        # Best value line
        best_val = min(ys) if m.lower_better else max(ys)
        fig.add_hline(y=best_val, line_dash="dot", line_color="#ff4081",
                      annotation_text=f"Best: {best_val}", annotation_position="bottom right")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,15,30,0.6)",
            font=dict(color="#c8d0e0", family="JetBrains Mono, monospace"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)",
                       title=m.unit or "value"),
            margin=dict(l=50, r=30, t=30, b=50),
            hovermode="x unified",
        )
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        first, last = ys[0], ys[-1]
        delta = last - first
        pct   = (delta / first * 100) if first != 0 else 0
        stats = {
            "count":   len(entries),
            "latest":  last,
            "best":    best_val,
            "first":   first,
            "delta":   delta,
            "pct":     pct,
            "improved": (delta < 0 if m.lower_better else delta > 0),
        }

    return templates.TemplateResponse("metric.html", {
        "request": request, "metric": m, "entries": list(reversed(entries)),
        "chart_json": chart_json, "stats": stats,
    })

# ── Log entry ─────────────────────────────────────────────────────────────────
@app.post("/metrics/{metric_id}/entries")
async def add_entry(
    request: Request,
    metric_id: int,
    value: float = Form(...),
    note: str = Form(""),
    recorded_at: str = Form(""),
    db: Session = Depends(get_db)
):
    if not request.session.get("authenticated"):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id).first()
    if not m:
        raise HTTPException(status_code=404)
    ts = datetime.fromisoformat(recorded_at) if recorded_at else datetime.utcnow()
    e  = Entry(metric_id=metric_id, value=value, note=note or None, recorded_at=ts)
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
