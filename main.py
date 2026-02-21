import os
import json
import csv
import io
import hashlib
import secrets
from collections import defaultdict
from datetime import datetime, date as date_type, time as time_type
from typing import Optional

from fastapi import FastAPI, Request, Form, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import plotly.graph_objects as go
import plotly.utils
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tracker.db")
SECRET_KEY   = os.getenv("SECRET_KEY", "super-secret-key-change-in-prod")

# ── DB setup ──────────────────────────────────────────────────────────────────
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine       = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base         = declarative_base()


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    api_key       = Column(String, unique=True, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    metrics       = relationship("Metric", back_populates="owner")


class Metric(Base):
    __tablename__ = "metrics"
    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True)
    name         = Column(String, nullable=False)
    description  = Column(Text, nullable=True)
    unit         = Column(String, nullable=True)
    lower_better = Column(Integer, default=1)
    graph_mode   = Column(String, default='single')    # 'single' or 'multi'
    series_names = Column(Text, nullable=True)          # comma-separated
    time_unit    = Column(String, default='day')        # 'day', 'week', 'month'
    is_public    = Column(Integer, default=0)           # 0=private, 1=public
    created_at   = Column(DateTime, default=datetime.utcnow)
    owner        = relationship("User", back_populates="metrics")
    entries      = relationship("Entry", back_populates="metric", cascade="all, delete-orphan")
    annotations  = relationship("Annotation", back_populates="metric", cascade="all, delete-orphan")


class Entry(Base):
    __tablename__ = "entries"
    id           = Column(Integer, primary_key=True, index=True)
    metric_id    = Column(Integer, ForeignKey("metrics.id"), nullable=False)
    value        = Column(Float, nullable=False)
    note         = Column(Text, nullable=True)
    series_label = Column(String, nullable=True)
    recorded_at  = Column(DateTime, default=datetime.utcnow)
    metric       = relationship("Metric", back_populates="entries")


class Annotation(Base):
    __tablename__ = "annotations"
    id           = Column(Integer, primary_key=True, index=True)
    metric_id    = Column(Integer, ForeignKey("metrics.id"), nullable=False)
    label        = Column(String, nullable=False)
    annotated_at = Column(DateTime, nullable=False)
    metric       = relationship("Metric", back_populates="annotations")


Base.metadata.create_all(bind=engine)

# ── Schema migrations ─────────────────────────────────────────────────────────
with engine.connect() as _conn:
    for _stmt in [
        "ALTER TABLE metrics ADD COLUMN graph_mode VARCHAR DEFAULT 'single'",
        "ALTER TABLE entries ADD COLUMN series_label VARCHAR",
        "ALTER TABLE metrics ADD COLUMN series_names TEXT",
        "ALTER TABLE metrics ADD COLUMN time_unit VARCHAR DEFAULT 'day'",
        "ALTER TABLE metrics ADD COLUMN user_id INTEGER REFERENCES users(id)",
        "ALTER TABLE metrics ADD COLUMN is_public INTEGER DEFAULT 0",
    ]:
        try:
            _conn.execute(text(_stmt))
            _conn.commit()
        except Exception:
            pass

# Remove the old global UNIQUE constraint on metrics.name so that different
# users can share the same metric name. SQLite doesn't support DROP CONSTRAINT,
# so we recreate the table when the old unique constraint is detected.
def _migrate_drop_metrics_name_unique():
    if not DATABASE_URL.startswith("sqlite"):
        # PostgreSQL supports ALTER TABLE ... DROP CONSTRAINT
        with engine.connect() as conn:
            for cname in ("metrics_name_key", "uq_metrics_name", "ix_metrics_name"):
                try:
                    conn.execute(text(f"ALTER TABLE metrics DROP CONSTRAINT IF EXISTS {cname}"))
                    conn.commit()
                except Exception:
                    pass
        return

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name='metrics'")
        ).fetchone()
        if not row:
            return
        # If name column still has UNIQUE in the CREATE TABLE statement, rebuild
        # the table without it.  PRAGMA table_info does NOT include UNIQUE so we
        # look directly at the stored SQL.
        if "UNIQUE" not in row[0].upper():
            return  # already clean — nothing to do

        pragma = conn.execute(text("PRAGMA table_info(metrics)")).fetchall()
        # pragma row: (cid, name, type, notnull, dflt_value, pk)
        col_names = [r[1] for r in pragma]
        col_defs  = []
        for r in pragma:
            _, cname, ctype, notnull, dflt, pk = r
            if pk:
                col_defs.append(f'"{cname}" {ctype} PRIMARY KEY')
            else:
                defn = f'"{cname}" {ctype}'
                if notnull:
                    defn += " NOT NULL"
                if dflt is not None:
                    defn += f" DEFAULT {dflt}"
                col_defs.append(defn)

        cols_csv = ", ".join(f'"{c}"' for c in col_names)
        conn.execute(text(f"CREATE TABLE metrics_new ({', '.join(col_defs)})"))
        conn.execute(text(f"INSERT INTO metrics_new ({cols_csv}) SELECT {cols_csv} FROM metrics"))
        conn.execute(text("DROP TABLE metrics"))
        conn.execute(text("ALTER TABLE metrics_new RENAME TO metrics"))
        conn.commit()

_migrate_drop_metrics_name_unique()

# Assign any metrics that pre-date multi-user support to the first user
with SessionLocal() as _s:
    _first = _s.query(User).order_by(User.id).first()
    if _first:
        _s.query(Metric).filter(Metric.user_id == None).update({"user_id": _first.id})
        _s.commit()

# ── Password helpers ───────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    key  = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 260_000)
    return f"{salt}:{key.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    try:
        salt, key_hex = hashed.split(':', 1)
    except ValueError:
        return False
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 260_000)
    return secrets.compare_digest(key.hex(), key_hex)


# ── Series colour palette ──────────────────────────────────────────────────────
SERIES_COLORS = ["#00e5ff", "#ff4081", "#00e676", "#ffab40", "#ce93d8", "#ff6e40", "#40c4ff"]


def _add_annotation_vline(fig: go.Figure, x_val: str, label: str) -> None:
    """Draw a vertical annotation line without triggering Plotly's internal
    _mean() call, which fails when x-axis values are strings (date labels)."""
    fig.add_shape(
        type="line",
        x0=x_val, x1=x_val, y0=0, y1=1, yref="paper",
        line=dict(dash="dash", color="#ffab40", width=1.5),
    )
    fig.add_annotation(
        x=x_val, y=0.98, yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=10, color="#ffab40"),
        xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0)",
    )


def _format_date(dt: datetime, time_unit: str) -> str:
    if time_unit == 'week':
        return dt.strftime("%G-W%V")
    elif time_unit == 'month':
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d")


def _parse_recorded_at(ra: str) -> datetime:
    if 'W' in ra:
        d = datetime.strptime(ra + '-1', "%G-W%V-%u").date()
    elif len(ra) == 7:
        d = date_type.fromisoformat(ra + '-01')
    else:
        d = date_type.fromisoformat(ra[:10])
    return datetime.combine(d, time_type.min)


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


# ── Chart + stats computation (shared by private and public views) ─────────────
def _compute_metric_view_data(m: Metric, entries: list, annotations: list) -> dict:
    """Build Plotly charts and summary stats for a metric. Pure function — no DB calls."""
    chart_json         = None
    series_charts_json = "[]"
    stats              = {}
    series_labels      = []
    series_pct         = []

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
        tu = getattr(m, 'time_unit', 'day') or 'day'

        if graph_mode == 'multi':
            series_labels = sorted(set(e.series_label for e in entries if e.series_label))
            groups: dict[str, list] = defaultdict(list)
            for e in entries:
                groups[e.series_label or 'default'].append(e)

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

            fig_combined = go.Figure()
            for i, (label, s_entries) in enumerate(groups.items()):
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                xs = [_format_date(e.recorded_at, tu) for e in s_entries]
                ys = [e.value for e in s_entries]
                fig_combined.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=7, color=color, line=dict(color="#0a0f1e", width=1.5)),
                    hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
                ))
            for ann in annotations:
                _add_annotation_vline(fig_combined, _format_date(ann.annotated_at, tu), ann.label)
            layout = _chart_layout(m.unit)
            layout["legend"] = dict(bgcolor="rgba(0,0,0,0)",
                                    bordercolor="rgba(255,255,255,0.1)", borderwidth=1)
            fig_combined.update_layout(**layout)
            chart_json = json.dumps(fig_combined, cls=plotly.utils.PlotlyJSONEncoder)

            sep_list = []
            for i, (label, s_entries) in enumerate(groups.items()):
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                xs = [_format_date(e.recorded_at, tu) for e in s_entries]
                ys = [e.value for e in s_entries]
                fig_sep = go.Figure()
                fig_sep.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=7, color=color, line=dict(color="#0a0f1e", width=1.5)),
                    hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
                ))
                s_best = min(ys) if m.lower_better else max(ys)
                fig_sep.add_hline(y=s_best, line_dash="dot", line_color="#ff4081",
                                  annotation_text=f"Best: {s_best}",
                                  annotation_position="bottom right")
                for ann in annotations:
                    _add_annotation_vline(fig_sep, _format_date(ann.annotated_at, tu), ann.label)
                fig_sep.update_layout(**_chart_layout(m.unit))
                sep_list.append({
                    "label": label,
                    "chartData": json.loads(json.dumps(fig_sep, cls=plotly.utils.PlotlyJSONEncoder)),
                })
            series_charts_json = json.dumps(sep_list)

        else:
            xs = [_format_date(e.recorded_at, tu) for e in entries]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=ys_all, mode="lines+markers", name=m.name,
                line=dict(color="#00e5ff", width=2),
                marker=dict(size=7, color="#00e5ff", line=dict(color="#0a0f1e", width=1.5)),
                hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
            ))
            fig.add_hline(y=best_val, line_dash="dot", line_color="#ff4081",
                          annotation_text=f"Best: {best_val}", annotation_position="bottom right")
            for ann in annotations:
                _add_annotation_vline(fig, _format_date(ann.annotated_at, tu), ann.label)
            fig.update_layout(**_chart_layout(m.unit))
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    series_charts_list = ([s["label"] for s in json.loads(series_charts_json)]
                          if series_charts_json != "[]" else [])
    return {
        "chart_json":         chart_json,
        "series_charts_json": series_charts_json,
        "series_charts_list": series_charts_list,
        "stats":              stats,
        "series_labels":      series_labels,
        "series_pct":         series_pct,
        "defined_series":     defined_series,
        "time_unit":          getattr(m, 'time_unit', 'day') or 'day',
    }


# ── Dashboard summaries helper ─────────────────────────────────────────────────
def _compute_summaries(metrics: list) -> list:
    summaries = []
    for m in metrics:
        entries = sorted(m.entries, key=lambda e: e.recorded_at)
        latest  = entries[-1].value if entries else None
        best    = (min(e.value for e in entries) if entries and m.lower_better
                   else (max(e.value for e in entries) if entries else None))
        first   = entries[0].value if entries else None
        pct     = ((latest - first) / first * 100
                   if (first is not None and first != 0 and latest is not None) else None)
        series_summary = []
        if (getattr(m, 'graph_mode', 'single') or 'single') == 'multi' and entries:
            groups: dict[str, list] = defaultdict(list)
            for e in entries:
                groups[e.series_label or 'default'].append(e)
            for label, s_entries in groups.items():
                s_ys = [e.value for e in s_entries]
                s_latest, s_first = s_ys[-1], s_ys[0]
                s_pct = ((s_latest - s_first) / s_first * 100) if s_first != 0 else 0
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
    return summaries


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


def _uid(request: Request) -> int | None:
    return request.session.get("user_id")


# ── API key auth dependency ────────────────────────────────────────────────────
def get_api_user(x_api_key: str = Header(None), db: Session = Depends(get_db)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Api-Key header")
    user = db.query(User).filter(User.api_key == x_api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


# ── Pydantic model for REST API ────────────────────────────────────────────────
class EntryCreate(BaseModel):
    value: float
    note: Optional[str] = None
    recorded_at: Optional[str] = None
    series_label: Optional[str] = None


# ── First-time setup (shown only when no accounts exist) ──────────────────────
@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request, db: Session = Depends(get_db)):
    if db.query(User).count() > 0:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("setup.html", {"request": request, "error": None})


@app.post("/setup")
async def setup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    if db.query(User).count() > 0:
        return RedirectResponse("/login", status_code=302)
    if not username.strip() or len(password) < 8:
        return templates.TemplateResponse("setup.html", {
            "request": request,
            "error": "Username is required and password must be at least 8 characters.",
        })
    user = User(username=username.strip(), password_hash=_hash_password(password),
                api_key=secrets.token_urlsafe(32))
    db.add(user)
    db.commit()
    db.refresh(user)
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    return RedirectResponse("/", status_code=302)


# ── Open registration (always available) ──────────────────────────────────────
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, db: Session = Depends(get_db)):
    if db.query(User).count() == 0:
        return RedirectResponse("/setup", status_code=302)
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    if db.query(User).count() == 0:
        return RedirectResponse("/setup", status_code=302)
    username = username.strip()
    if not username or len(password) < 8:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Username is required and password must be at least 8 characters.",
        })
    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "That username is already taken.",
        })
    user = User(username=username, password_hash=_hash_password(password),
                api_key=secrets.token_urlsafe(32))
    db.add(user)
    db.commit()
    db.refresh(user)
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    return RedirectResponse("/", status_code=302)


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, db: Session = Depends(get_db)):
    if db.query(User).count() == 0:
        return RedirectResponse("/setup", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not _verify_password(password, user.password_hash):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password.",
        })
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    return RedirectResponse("/", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ── Settings ──────────────────────────────────────────────────────────────────
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: Session = Depends(get_db),
                         regenerated: str = None):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    user = db.query(User).filter(User.id == _uid(request)).first()
    success = "API key regenerated successfully." if regenerated else None
    return templates.TemplateResponse("settings.html", {
        "request": request, "user": user,
        "success": success, "error": None,
    })


@app.post("/settings/regenerate-key")
async def regenerate_api_key(request: Request, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    user = db.query(User).filter(User.id == _uid(request)).first()
    user.api_key = secrets.token_urlsafe(32)
    db.commit()
    return RedirectResponse("/settings?regenerated=1", status_code=302)


@app.post("/settings/change-password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db)
):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    user = db.query(User).filter(User.id == _uid(request)).first()
    if not _verify_password(current_password, user.password_hash):
        return templates.TemplateResponse("settings.html", {
            "request": request, "user": user,
            "error": "Current password is incorrect.", "success": None,
        })
    if len(new_password) < 8:
        return templates.TemplateResponse("settings.html", {
            "request": request, "user": user,
            "error": "New password must be at least 8 characters.", "success": None,
        })
    user.password_hash = _hash_password(new_password)
    db.commit()
    return templates.TemplateResponse("settings.html", {
        "request": request, "user": user,
        "error": None, "success": "Password changed successfully.",
    })


# ── REST API ──────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
async def api_list_metrics(user: User = Depends(get_api_user), db: Session = Depends(get_db)):
    metrics = db.query(Metric).filter(Metric.user_id == user.id).order_by(Metric.created_at.desc()).all()
    return [
        {"id": m.id, "name": m.name, "unit": m.unit,
         "graph_mode": m.graph_mode, "time_unit": m.time_unit,
         "lower_better": bool(m.lower_better), "is_public": bool(m.is_public),
         "entry_count": len(m.entries)}
        for m in metrics
    ]


@app.post("/api/metrics/{metric_id}/entries", status_code=201)
async def api_add_entry(
    metric_id: int,
    body: EntryCreate,
    user: User = Depends(get_api_user),
    db: Session = Depends(get_db)
):
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == user.id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Metric not found")
    ts = (_parse_recorded_at(body.recorded_at) if body.recorded_at
          else datetime.combine(datetime.utcnow().date(), time_type.min))
    e = Entry(metric_id=metric_id, value=body.value, note=body.note,
              series_label=body.series_label, recorded_at=ts)
    db.add(e)
    db.commit()
    db.refresh(e)
    return {"id": e.id, "metric_id": e.metric_id, "value": e.value,
            "recorded_at": e.recorded_at.isoformat(), "note": e.note,
            "series_label": e.series_label}


# ── CSV Export ─────────────────────────────────────────────────────────────────
@app.get("/metrics/{metric_id}/export.csv")
async def export_csv(request: Request, metric_id: int, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404)
    entries = sorted(m.entries, key=lambda e: e.recorded_at)
    tu = getattr(m, 'time_unit', 'day') or 'day'
    output = io.StringIO()
    writer = csv.writer(output)
    if (getattr(m, 'graph_mode', 'single') or 'single') == 'multi':
        writer.writerow(["date", "series", "value", "note"])
        for e in entries:
            writer.writerow([_format_date(e.recorded_at, tu), e.series_label or '', e.value, e.note or ''])
    else:
        writer.writerow(["date", "value", "note"])
        for e in entries:
            writer.writerow([_format_date(e.recorded_at, tu), e.value, e.note or ''])
    filename = m.name.replace(' ', '_').replace('/', '_') + ".csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Public profile ─────────────────────────────────────────────────────────────
@app.get("/u/{username}", response_class=HTMLResponse)
async def public_profile(request: Request, username: str, db: Session = Depends(get_db)):
    owner = db.query(User).filter(User.username == username).first()
    if not owner:
        raise HTTPException(status_code=404, detail="User not found")
    public_metrics = (db.query(Metric)
                      .filter(Metric.user_id == owner.id, Metric.is_public == 1)
                      .order_by(Metric.created_at.desc()).all())
    summaries = _compute_summaries(public_metrics)
    return templates.TemplateResponse("public_profile.html", {
        "request": request, "owner": owner, "summaries": summaries,
    })


# ── Public metric view (read-only) ─────────────────────────────────────────────
@app.get("/u/{username}/{metric_id}", response_class=HTMLResponse)
async def public_metric(request: Request, username: str, metric_id: int,
                         db: Session = Depends(get_db)):
    owner = db.query(User).filter(User.username == username).first()
    if not owner:
        raise HTTPException(status_code=404)
    m = db.query(Metric).filter(
        Metric.id == metric_id,
        Metric.user_id == owner.id,
        Metric.is_public == 1,
    ).first()
    if not m:
        raise HTTPException(status_code=404)
    entries     = sorted(m.entries, key=lambda e: e.recorded_at)
    annotations = sorted(m.annotations, key=lambda a: a.annotated_at)
    view_data   = _compute_metric_view_data(m, entries, annotations)
    return templates.TemplateResponse("public_metric.html", {
        "request": request, "metric": m, "owner": owner,
        "entries": list(reversed(entries)),
        "annotations": annotations,
        **view_data,
    })


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    metrics = (db.query(Metric)
               .filter(Metric.user_id == _uid(request))
               .order_by(Metric.created_at.desc()).all())
    summaries = _compute_summaries(metrics)
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "summaries": summaries,
    })


# ── Create metric ─────────────────────────────────────────────────────────────
@app.get("/metrics/new", response_class=HTMLResponse)
async def new_metric_page(request: Request):
    if not _uid(request):
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
    time_unit: str = Form("day"),
    is_public: int = Form(0),
    db: Session = Depends(get_db)
):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    existing = db.query(Metric).filter(Metric.name == name, Metric.user_id == _uid(request)).first()
    if existing:
        return templates.TemplateResponse("new_metric.html", {
            "request": request,
            "error": f'You already have a metric named "{name}".',
        })
    if graph_mode not in ("single", "multi"):
        graph_mode = "single"
    if time_unit not in ("day", "week", "month"):
        time_unit = "day"
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
               lower_better=lower_better, graph_mode=graph_mode, series_names=series_names,
               time_unit=time_unit, is_public=is_public, user_id=_uid(request))
    db.add(m)
    db.commit()
    db.refresh(m)
    return RedirectResponse(f"/metrics/{m.id}", status_code=302)


# ── Metric detail ─────────────────────────────────────────────────────────────
@app.get("/metrics/{metric_id}", response_class=HTMLResponse)
async def metric_detail(request: Request, metric_id: int, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404, detail="Metric not found")
    entries     = sorted(m.entries, key=lambda e: e.recorded_at)
    annotations = sorted(m.annotations, key=lambda a: a.annotated_at)
    view_data   = _compute_metric_view_data(m, entries, annotations)
    owner_username = request.session.get("username", "")
    share_url = f"/u/{owner_username}/{m.id}" if m.is_public and owner_username else None
    return templates.TemplateResponse("metric.html", {
        "request": request, "metric": m,
        "entries": list(reversed(entries)),
        "annotations": annotations,
        "share_url": share_url,
        **view_data,
    })


# ── Toggle metric visibility ───────────────────────────────────────────────────
@app.post("/metrics/{metric_id}/toggle-visibility")
async def toggle_visibility(request: Request, metric_id: int, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404)
    m.is_public = 0 if m.is_public else 1
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)


# ── Manage predefined series (multi mode) ─────────────────────────────────────
def _parse_series(m: Metric) -> list[str]:
    return [s.strip() for s in (m.series_names or "").split(",") if s.strip()]


@app.post("/metrics/{metric_id}/series/add")
async def series_add(request: Request, metric_id: int,
                     series_name: str = Form(...), db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404)
    name = series_name.strip()
    names = _parse_series(m)
    if name and name not in names:
        names.append(name)
        m.series_names = ",".join(names)
        db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)


@app.post("/metrics/{metric_id}/series/remove")
async def series_remove(request: Request, metric_id: int,
                         series_name: str = Form(...), db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404)
    names = [s for s in _parse_series(m) if s != series_name.strip()]
    m.series_names = ",".join(names) or None
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)


# ── Annotations ───────────────────────────────────────────────────────────────
@app.post("/metrics/{metric_id}/annotations")
async def add_annotation(request: Request, metric_id: int,
                          label: str = Form(...), annotated_at: str = Form(...),
                          db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404)
    ann = Annotation(metric_id=metric_id, label=label.strip(),
                     annotated_at=_parse_recorded_at(annotated_at))
    db.add(ann)
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)


@app.post("/annotations/{annotation_id}/delete")
async def delete_annotation(request: Request, annotation_id: int, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    ann = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not ann or ann.metric.user_id != _uid(request):
        raise HTTPException(status_code=404)
    metric_id = ann.metric_id
    db.delete(ann)
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)


# ── Log entry ─────────────────────────────────────────────────────────────────
@app.post("/metrics/{metric_id}/entries")
async def add_entry(request: Request, metric_id: int,
                    value: float = Form(...), note: str = Form(""),
                    recorded_at: str = Form(""), series_label: str = Form(""),
                    db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404)
    ts = (_parse_recorded_at(recorded_at) if recorded_at
          else datetime.combine(datetime.utcnow().date(), time_type.min))
    e = Entry(metric_id=metric_id, value=value, note=note or None,
              recorded_at=ts, series_label=series_label or None)
    db.add(e)
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)


# ── Delete entry ──────────────────────────────────────────────────────────────
@app.post("/entries/{entry_id}/delete")
async def delete_entry(request: Request, entry_id: int, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    e = db.query(Entry).filter(Entry.id == entry_id).first()
    if not e or e.metric.user_id != _uid(request):
        raise HTTPException(status_code=404)
    metric_id = e.metric_id
    db.delete(e)
    db.commit()
    return RedirectResponse(f"/metrics/{metric_id}", status_code=302)


# ── Delete metric ─────────────────────────────────────────────────────────────
@app.post("/metrics/{metric_id}/delete")
async def delete_metric(request: Request, metric_id: int, db: Session = Depends(get_db)):
    if not _uid(request):
        return RedirectResponse("/login", status_code=302)
    m = db.query(Metric).filter(Metric.id == metric_id, Metric.user_id == _uid(request)).first()
    if not m:
        raise HTTPException(status_code=404)
    db.delete(m)
    db.commit()
    return RedirectResponse("/", status_code=302)
