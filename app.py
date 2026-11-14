"""
Buffalo Turkey Trot Results Dashboard
"""

import io
import os
import re
import sqlite3
import textwrap
from contextlib import closing
from functools import lru_cache
from html import escape as html_escape
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import flask
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from werkzeug.middleware.proxy_fix import ProxyFix
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc

DB_PATH = Path(__file__).parent / "db" / "turkey_trot.db"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# numpy scalars (e.g. values pulled out of a DataFrame) are otherwise bound as
# BLOBs by sqlite3, so `WHERE year = ?` silently matches nothing.
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.float64, float)


def get_conn():
    return sqlite3.connect(DB_PATH)


def query(sql: str, params=()) -> pd.DataFrame:
    # `with sqlite3.connect()` only manages transactions; closing() releases it.
    with closing(get_conn()) as conn:
        return pd.read_sql(sql, conn, params=params)


def fmt_time(secs):
    if pd.isna(secs):
        return "—"
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def normalize_name(name: str) -> str:
    """Match the normalization applied during DB migration."""
    import re as _re
    if not name:
        return ""
    s = name.strip()
    s = _re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)\s*$", "", s, flags=_re.I)
    parts = s.split()
    if len(parts) > 2:
        parts = [
            p for i, p in enumerate(parts)
            if i == 0 or i == len(parts) - 1
            or not _re.fullmatch(r"[A-Za-z]\.?", p)
        ]
    return " ".join(parts).lower()


def search_names(search: str, limit: int = 50) -> list[dict]:
    """Return distinct (name, name_key) pairs whose name_key contains the search string."""
    key = normalize_name(search)
    df = query(
        """SELECT name, name_key,
                  COUNT(*) AS appearances
           FROM results
           WHERE name_key LIKE ?
           GROUP BY name, name_key
           ORDER BY appearances DESC, name
           LIMIT ?""",
        (f"%{key}%", limit),
    )
    return df.to_dict("records")


def _normalize_time_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Source data mixes '0:24:39' and '24:01'; rebuild display strings from seconds."""
    for col in ("gun_time", "net_time"):
        sec = df[f"{col}_sec"]
        df[col] = [fmt_time(s) if pd.notna(s) else (t or "—")
                   for s, t in zip(sec, df[col])]
    return df


def runner_history(name_key: str) -> pd.DataFrame:
    """Fetch all results matching this normalized key (covers all name variants)."""
    return runner_history_multi({name_key})


def runner_history_multi(keys: set[str]) -> pd.DataFrame:
    """Fetch results for a set of name_keys combined."""
    if not keys:
        return pd.DataFrame()
    ph = ",".join("?" * len(keys))
    df = query(
        f"SELECT year, name, overall_place, sex_place, age_group, ag_rank_of_total, "
        f"gun_time, gun_time_sec, net_time, net_time_sec, city, state, age "
        f"FROM results WHERE name_key IN ({ph}) ORDER BY year",
        tuple(sorted(keys)),
    )
    return _normalize_time_strings(df)


# ---------------------------------------------------------------------------
# Name-link helpers  (persisted in browser localStorage via dcc.Store)
# ---------------------------------------------------------------------------

def find_linked_group(key: str, store: list) -> set[str]:
    """Return all name_keys in the same linked group as `key` (including `key`)."""
    for group in store:
        if key in group:
            return set(group)
    return {key}


def _add_link(key_a: str, key_b: str, store: list) -> list:
    groups = [set(g) for g in store]
    grp_a = next((g for g in groups if key_a in g), None)
    grp_b = next((g for g in groups if key_b in g), None)
    if grp_a is not None and grp_a is grp_b:
        return store
    merged = (grp_a or {key_a}) | (grp_b or {key_b})
    others = [g for g in groups if g is not grp_a and g is not grp_b]
    return [sorted(g) for g in others] + [sorted(merged)]


def _remove_from_link_group(key: str, store: list) -> list:
    result = []
    for group in store:
        if key in group:
            rest = [k for k in group if k != key]
            if len(rest) >= 2:
                result.append(rest)
        else:
            result.append(group)
    return result


def check_link_compat(key_a: str, key_b: str) -> tuple[bool, str]:
    """
    Check whether two name_keys could represent the same person.
    Returns (is_compatible, message).
    """
    df_a = runner_history(key_a).assign(_src="a")
    df_b = runner_history(key_b).assign(_src="b")
    if df_a.empty:
        return True, f'No data found for "{key_a}".'
    if df_b.empty:
        return True, f'No data found for "{key_b}".'

    shared = set(df_a["year"]) & set(df_b["year"])
    if shared:
        return False, (
            f'Both names appear in {", ".join(str(y) for y in sorted(shared))} '
            f"— can't be the same runner."
        )

    combined = pd.concat([df_a, df_b], ignore_index=True)
    persons = split_into_persons(combined)

    for p in persons:
        if "_src" in p.columns and {"a", "b"} <= set(p["_src"]):
            yrs_a = sorted(df_a["year"].tolist())
            yrs_b = sorted(df_b["year"].tolist())
            return True, (
                f"Age-consistent: {key_a} ({yrs_a[0]}–{yrs_a[-1]}) + "
                f"{key_b} ({yrs_b[0]}–{yrs_b[-1]})."
            )

    return False, "Age conflict: these names appear to be different people."


def age_group_bounds(ag: str) -> tuple[float, float] | None:
    """Return (lo, hi) for an age-group label like M20-24, F65-69, M70+."""
    if not ag:
        return None
    m = re.match(r"[MF]\s*(\d+)\s*[-+\s]\s*(\d+)?", str(ag))
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo + 9
    return lo, hi


def age_from_group(ag: str) -> float | None:
    b = age_group_bounds(ag)
    return (b[0] + b[1]) / 2.0 if b else None



def row_age_range(row) -> tuple[float, float] | None:
    """Return (lo, hi) for a row's age. Exact age → (age, age). Group → range. None if unknown."""
    age = row.get("age")
    if pd.notna(age):
        a = float(age)
        return (a, a)
    return age_group_bounds(row.get("age_group", "") or "")


def split_into_persons(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split results for one name_key into groups, each representing one individual.

    Instead of narrowing a range-intersection anchor (which over-constrains after
    many years), we maintain running birth-year estimates for each chain.  A new row
    is consistent with a chain when the chain's estimated age for that year falls
    within the row's age range ±TOLERANCE.  When multiple chains match, we pick the
    best-fitting one (estimated age closest to the row's midpoint).

    Birth-year estimates use exact ages (weight 2) and age-group midpoints (weight 1),
    so a single exact age strongly anchors long pre-2006 chains.
    """
    if df.empty:
        return [df]

    TOLERANCE = 2.0  # years; covers birthday timing + age-group boundary effects

    df = df.sort_values("year").reset_index(drop=True)

    persons: list[list[int]] = []
    # Per-chain: (exact_birth_estimates, group_birth_estimates, fallback_anchor)
    # fallback_anchor = (year, lo, hi) used only when a chain has zero age data
    chain_exact: list[list[float]] = []
    chain_group: list[list[float]] = []
    chain_anchor: list[tuple[int, float, float]] = []

    def _est_birth(p_idx: int) -> float | None:
        ex, gr = chain_exact[p_idx], chain_group[p_idx]
        if ex:
            return sum(ex) / len(ex)
        if gr:
            return sum(gr) / len(gr)
        return None

    for i, row in df.iterrows():
        year = int(row["year"])
        r    = row_age_range(row)
        age  = row.get("age")
        has_exact = pd.notna(age) and age is not None

        best_chain: int | None = None
        best_fit = float("-inf")

        for p_idx in range(len(persons)):
            est_birth = _est_birth(p_idx)

            if r is None:
                # No age info — consistent with any chain (neutral fit, first wins)
                if best_chain is None:
                    best_chain = p_idx
                continue

            row_lo, row_hi = r

            if est_birth is not None:
                est_age = year - est_birth
                consistent = (row_lo - TOLERANCE) <= est_age <= (row_hi + TOLERANCE)
                if consistent:
                    fit = -abs(est_age - (row_lo + row_hi) / 2.0)
                    if fit > best_fit:
                        best_fit = fit
                        best_chain = p_idx
            else:
                # Chain has no age data yet — use fallback range overlap
                anc_yr, anc_lo, anc_hi = chain_anchor[p_idx]
                delta = year - anc_yr
                exp_lo = anc_lo + delta - TOLERANCE
                exp_hi = anc_hi + delta + TOLERANCE
                if row_lo <= exp_hi and row_hi >= exp_lo:
                    if best_chain is None:
                        best_chain = p_idx

        if best_chain is not None:
            p_idx = best_chain
            persons[p_idx].append(i)
            if r is not None:
                row_lo, row_hi = r
                birth_est = year - (row_lo + row_hi) / 2.0
                if has_exact:
                    chain_exact[p_idx].append(year - float(age))
                else:
                    chain_group[p_idx].append(birth_est)
                chain_anchor[p_idx] = (year, row_lo, row_hi)
        else:
            persons.append([i])
            if r is not None:
                birth_est = year - (r[0] + r[1]) / 2.0
                chain_exact.append([year - float(age)] if has_exact else [])
                chain_group.append([] if has_exact else [birth_est])
                chain_anchor.append((year, r[0], r[1]))
            else:
                chain_exact.append([])
                chain_group.append([])
                chain_anchor.append((year, 0.0, 100.0))

    return [df.iloc[indices].reset_index(drop=True) for indices in persons]


def _est_birth_year(person_df: pd.DataFrame) -> float | None:
    """Best birth-year estimate for a chain, prioritising exact ages."""
    exact = []
    group = []
    for _, row in person_df.iterrows():
        age = row.get("age")
        if pd.notna(age) and age is not None:
            exact.append(float(row["year"]) - float(age))
        else:
            r = age_group_bounds(row.get("age_group") or "")
            if r:
                group.append(float(row["year"]) - (r[0] + r[1]) / 2.0)
    if exact:
        return sum(exact) / len(exact)
    if group:
        return sum(group) / len(group)
    return None


def _compute_estimated_ages(person_df: pd.DataFrame) -> list:
    """
    Return one display value per row: exact int where known, '~N' where estimated,
    or '' if the chain has no age data at all.
    """
    est_birth = _est_birth_year(person_df)
    result = []
    for _, row in person_df.iterrows():
        age = row.get("age")
        if pd.notna(age) and age is not None:
            result.append(int(age))
        elif est_birth is not None:
            result.append(f"~{round(float(row['year']) - est_birth)}")
        else:
            result.append("")
    return result


def person_label(name: str, df: pd.DataFrame) -> str:
    """Short label distinguishing this person from others with the same name."""
    est_birth = _est_birth_year(df)
    if est_birth is not None:
        return f"{name} (b. ~{round(est_birth)})"
    return name


@lru_cache(maxsize=1)
def year_summary() -> pd.DataFrame:
    return query("""
        SELECT
            year,
            COUNT(*) AS finishers,
            ROUND(AVG(net_time_sec) / 60.0, 1) AS avg_net_min,
            ROUND(MIN(net_time_sec) / 60.0, 1) AS winner_min,
            ROUND(AVG(CASE WHEN sex='M' THEN net_time_sec END) / 60.0, 1) AS avg_male_min,
            ROUND(AVG(CASE WHEN sex='F' THEN net_time_sec END) / 60.0, 1) AS avg_female_min,
            SUM(CASE WHEN sex='M' THEN 1 ELSE 0 END) AS male_count,
            SUM(CASE WHEN sex='F' THEN 1 ELSE 0 END) AS female_count
        FROM results
        GROUP BY year ORDER BY year
    """)


def two_year_dist_data(year_a: int, year_b: int) -> pd.DataFrame:
    return query(
        "SELECT year, sex, net_time_sec, age, age_group FROM results "
        "WHERE year IN (?, ?) AND net_time_sec IS NOT NULL",
        (year_a, year_b),
    )


def year_data(year: int) -> pd.DataFrame:
    return query(
        "SELECT overall_place, name, age, sex, age_group, "
        "gun_time_sec, net_time_sec, city, state "
        "FROM results WHERE year = ?",
        (year,),
    )


def companions_for_runner(
    name_keys: set[str],
    gun_thresh: int = 2,
    net_thresh: int = 10,
) -> dict[int, list[str]]:
    if not name_keys:
        return {}
    excl = sorted(name_keys)
    ph = ",".join("?" * len(excl))
    params: list = excl + excl + [gun_thresh, gun_thresh, net_thresh, net_thresh]
    df = query(f"""
        WITH my_times AS (
            SELECT year, gun_time_sec, net_time_sec
            FROM results
            WHERE name_key IN ({ph})
              AND gun_time_sec IS NOT NULL
              AND net_time_sec IS NOT NULL
              AND gun_time_sec != net_time_sec
        )
        SELECT
            m.year,
            r.name,
            ABS((r.gun_time_sec - r.net_time_sec) - (m.gun_time_sec - m.net_time_sec)) AS start_delta,
            ABS(r.net_time_sec - m.net_time_sec) AS net_delta
        FROM my_times m
        JOIN results r
          ON r.year = m.year
         AND r.name_key NOT IN ({ph})
         AND (r.gun_time_sec - r.net_time_sec) BETWEEN (m.gun_time_sec - m.net_time_sec) - ? AND (m.gun_time_sec - m.net_time_sec) + ?
         AND r.net_time_sec BETWEEN m.net_time_sec - ? AND m.net_time_sec + ?
        ORDER BY m.year, start_delta + net_delta
    """, tuple(params))
    result: dict[int, list[str]] = {}
    for year, grp in df.groupby("year"):
        result[int(year)] = grp["name"].tolist()
    return result


@lru_cache(maxsize=1)
def names_by_year() -> dict[int, frozenset[str]]:
    df = query("SELECT year, LOWER(TRIM(name)) AS name FROM results")
    return {int(yr): frozenset(g["name"]) for yr, g in df.groupby("year")}


@lru_cache(maxsize=1)
def retention_matrix() -> pd.DataFrame:
    """For each pair of years, count runners appearing in both."""
    nby = names_by_year()
    years = sorted(nby)
    records = []
    for yr_a in years:
        for yr_b in years:
            if yr_a <= yr_b:
                overlap = len(nby[yr_a] & nby[yr_b])
                records.append({"year_a": yr_a, "year_b": yr_b, "shared": overlap})
                if yr_a != yr_b:
                    records.append({"year_a": yr_b, "year_b": yr_a, "shared": overlap})
    return pd.DataFrame(records)


@lru_cache(maxsize=1)
def streak_distribution() -> pd.DataFrame:
    """How many runners have a longest consecutive-year streak of N?"""
    nby = names_by_year()
    years = sorted(nby)
    streak_counts: dict[int, int] = {}
    for n in set().union(*nby.values()):
        best = cur = 0
        for yr in years:
            cur = cur + 1 if n in nby[yr] else 0
            best = max(best, cur)
        streak_counts[best] = streak_counts.get(best, 0) + 1
    return pd.DataFrame(sorted(streak_counts.items()), columns=["streak", "runners"])


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

class TrotDash(dash.Dash):
    """Adds per-link title and Open Graph tags to the server-rendered page."""

    def interpolate_index(self, **kwargs):
        meta = share_meta_tags()
        if meta:
            title, tags = meta
            kwargs["title"] = html_escape(title)
            kwargs["metas"] = kwargs.get("metas", "") + "\n" + tags
        return super().interpolate_index(**kwargs)


app = TrotDash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Buffalo Turkey Trot",
)
server = app.server  # for deployment
# Render terminates HTTPS at its proxy; trust X-Forwarded-Proto/Host so
# absolute URLs in share links and preview tags come out as https://<site>.
server.wsgi_app = ProxyFix(server.wsgi_app, x_proto=1, x_host=1)

# Google Analytics (GA4). Set GA_MEASUREMENT_ID (e.g. G-XXXXXXXXXX) in the
# hosting environment; when it's unset (local dev) no tracking code is emitted.
GA_MEASUREMENT_ID = os.environ.get("GA_MEASUREMENT_ID", "").strip()
if re.fullmatch(r"G-[A-Z0-9]+", GA_MEASUREMENT_ID):
    app.index_string = app.index_string.replace(
        "{%metas%}",
        "{%metas%}\n"
        f'<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>\n'
        "<script>\n"
        "  window.dataLayer = window.dataLayer || [];\n"
        "  function gtag(){dataLayer.push(arguments);}\n"
        "  gtag('js', new Date());\n"
        # Shared links put runner names in the query string and page title;
        # GA forbids personal info, so report only the bare path and site name.
        f"  gtag('config', '{GA_MEASUREMENT_ID}', {{\n"
        "    page_location: location.origin + location.pathname,\n"
        "    page_title: 'Buffalo Turkey Trot'\n"
        "  });\n"
        "</script>",
        1,
    )

def _backfill_name_keys() -> None:
    """Rows added after the name_key migration (e.g. 2020) have NULL keys and
    never show up in runner search. Fill them in using the same normalization."""
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, name FROM results WHERE name_key IS NULL AND name IS NOT NULL"
        ).fetchall()
        if rows:
            conn.executemany(
                "UPDATE results SET name_key = ? WHERE id = ?",
                [(normalize_name(name), rid) for rid, name in rows],
            )
            conn.commit()


_backfill_name_keys()

YEARS = query("SELECT DISTINCT year FROM results ORDER BY year")["year"].tolist()

# Corral analysis needs chip timing (gun time differs from net time). Early
# years were gun-timed only, and 2006/2020 have no gun times at all.
CHIP_YEARS = query("""
    SELECT year FROM results
    GROUP BY year
    HAVING SUM(gun_time_sec IS NOT NULL AND net_time_sec IS NOT NULL
               AND gun_time_sec != net_time_sec) > 0
    ORDER BY year
""")["year"].tolist()

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

HEADER = dbc.Navbar(
    dbc.Container(
        dbc.NavbarBrand("🦃 Buffalo Turkey Trot Results", className="fs-4 fw-bold"),
        fluid=True,
    ),
    color="warning",
    dark=False,
    className="mb-4",
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

lookup_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Search for a runner", className="fw-semibold"),
            dcc.Dropdown(
                id="runner-search",
                options=[],
                placeholder="Start typing a name…",
                searchable=True,
                clearable=True,
                className="mb-3",
            ),
        ], md=6),
        dbc.Col([
            html.Label("Add a second runner to compare", className="fw-semibold"),
            dcc.Dropdown(
                id="runner-search-2",
                options=[],
                placeholder="Optional: compare with…",
                searchable=True,
                clearable=True,
                className="mb-3",
            ),
        ], md=6),
    ]),
    # Share bar: shares the selected runner(s), or the site itself if none.
    html.Div([
        # Handled by assets/share.js (a native click listener), because the
        # share sheet must open inside the real tap gesture.
        html.Button("📲 Share to Instagram Story", id="ig-story-btn", n_clicks=0,
                    className="btn btn-sm me-2",
                    style={"background": "linear-gradient(45deg,#f09433,#e6683c,#dc2743,#cc2366,#bc1888)",
                           "border": "none", "color": "white", "fontWeight": 600}),
        dbc.Button("🔗 Share link", id="share-btn", color="warning", size="sm",
                   n_clicks=0, className="me-2"),
        dbc.DropdownMenu(
            label="⬇ Download image",
            color="secondary", size="sm", className="d-inline-block",
            children=[
                # The server sends these as attachments, so a plain link downloads.
                dbc.DropdownMenuItem("Link preview / Facebook (1200×630)",
                                     id="download-landscape", href="", external_link=True),
                dbc.DropdownMenuItem("Instagram post (1080×1350)",
                                     id="download-portrait", href="", external_link=True),
                dbc.DropdownMenuItem("Instagram story (1080×1920)",
                                     id="download-story", href="", external_link=True),
            ],
        ),
        html.Span(id="share-status", className="ms-2 small text-success"),
        html.Div(id="ig-status", className="small text-muted mt-1"),
    ], id="share-bar", className="mb-3"),
    html.Div(id="runner-output"),
    # Link section — always in DOM, hidden until a runner is selected
    html.Div([
        html.Hr(className="my-3"),
        html.Small("Link name variants", className="fw-semibold text-muted text-uppercase"),
        html.Div(id="linked-names-display", className="mt-1 mb-2"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id="link-candidate",
                placeholder="Search for another name to link…",
                searchable=True,
                clearable=True,
            ), md=8),
            dbc.Col(
                dbc.Button("Link", id="link-add-btn", color="success", size="sm",
                           n_clicks=0, disabled=True),
                md=2, className="d-flex align-items-center ps-0",
            ),
        ], className="g-2"),
        html.Div(id="link-preview", className="mt-2"),
    ], id="link-section", style={"display": "none"}),
    dcc.Store(id="name-links-store", storage_type="local", data=[]),
], fluid=True)


_YEAR_OPTS = [{"label": str(y), "value": y} for y in YEARS]

year_tab = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Graph(id="year-finishers"), md=6),
        dbc.Col(dcc.Graph(id="year-avg-time"), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="year-gender-split"), md=12),
    ]),
    html.Hr(className="my-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Year A", className="fw-semibold"),
            dcc.Dropdown(
                id="dist-year-a",
                options=_YEAR_OPTS,
                value=YEARS[-2] if len(YEARS) > 1 else YEARS[-1],
                clearable=False,
            ),
        ], md=2),
        dbc.Col([
            html.Label("Year B", className="fw-semibold"),
            dcc.Dropdown(
                id="dist-year-b",
                options=_YEAR_OPTS,
                value=YEARS[-1],
                clearable=False,
            ),
        ], md=2),
        dbc.Col([
            html.Label("Sex", className="fw-semibold"),
            dbc.RadioItems(
                id="dist-sex-filter",
                options=[
                    {"label": "All",   "value": "Both"},
                    {"label": "Men",   "value": "M"},
                    {"label": "Women", "value": "F"},
                ],
                value="Both",
                inline=True,
                className="mt-1",
            ),
        ], md=2),
        dbc.Col([
            html.Label("Age bracket", className="fw-semibold"),
            dcc.Dropdown(
                id="dist-age-bracket",
                options=[
                    {"label": "All ages",  "value": "All"},
                    {"label": "Under 20",  "value": "0-19"},
                    {"label": "20s",       "value": "20-29"},
                    {"label": "30s",       "value": "30-39"},
                    {"label": "40s",       "value": "40-49"},
                    {"label": "50s",       "value": "50-59"},
                    {"label": "60+",       "value": "60+"},
                ],
                value="All",
                clearable=False,
            ),
        ], md=4),
        dbc.Col([
            html.Label("View", className="fw-semibold"),
            dbc.RadioItems(
                id="dist-norm",
                options=[
                    {"label": "% of field", "value": "percent"},
                    {"label": "Count",       "value": ""},
                ],
                value="percent",
                inline=True,
                className="mt-1",
            ),
        ], md=2),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="year-dist"), md=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="year-dist-stats"), lg=8, xl=6),
    ], className="mb-3"),
], fluid=True)


corral_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Year", className="fw-semibold"),
            html.Small(
                " (chip-timed years only: earlier races recorded gun time alone)",
                className="text-muted",
            ),
            dcc.Slider(
                id="corral-year-slider",
                min=CHIP_YEARS[0], max=CHIP_YEARS[-1],
                step=None,
                marks={y: f"'{y % 100:02d}" for y in CHIP_YEARS},
                value=CHIP_YEARS[-1],
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-2",
            ),
        ]),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="corral-scatter"), md=8),
        dbc.Col([
            html.Div(id="corral-stats", className="mt-4"),
        ], md=4),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="corral-by-ag"), md=12),
    ]),
], fluid=True)


retention_tab = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Graph(id="retention-heatmap"), md=7),
        dbc.Col(dcc.Graph(id="retention-yoy"), md=5),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="retention-streak"), md=12),
    ]),
], fluid=True)


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    HEADER,
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(lookup_tab, label="Runner Lookup", tab_id="tab-lookup"),
            dbc.Tab(year_tab, label="Year Overview", tab_id="tab-year"),
            dbc.Tab(corral_tab, label="Corral Analysis", tab_id="tab-corral"),
            dbc.Tab(retention_tab, label="Retention", tab_id="tab-retention"),
        ], id="main-tabs", active_tab="tab-lookup"),
    ], fluid=True),
    html.Footer(
        html.A("Made by Tony", href="https://antoniosirianni.com", target="_blank",
               style={"color": "#aaa", "textDecoration": "none", "fontSize": "0.8rem"}),
        style={"textAlign": "center", "padding": "1.5rem 0", "marginTop": "1rem"}
    ),
], className="bg-light min-vh-100")


# ---------------------------------------------------------------------------
# Runner Lookup callbacks
# ---------------------------------------------------------------------------

def runner_card(name: str, df: pd.DataFrame, color: str, label: str | None = None) -> dbc.Card:
    if df.empty:
        return dbc.Alert(f"No results found for {name}.", color="warning")

    display_name = label or name
    df = df.copy()
    df["gap_sec"] = df["gun_time_sec"] - df["net_time_sec"]
    df["net_min"] = df["net_time_sec"] / 60

    # Time trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["net_min"],
        mode="lines+markers",
        name="Net time",
        line=dict(color=color, width=2),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Net: %{customdata[0]}<br>Overall: %{customdata[1]}<br>Age group: %{customdata[2]}<extra></extra>",
        customdata=list(zip(df["net_time"], df["overall_place"], df["age_group"])),
    ))
    fig.update_layout(
        title=display_name,
        xaxis_title="Year",
        yaxis_title="Net time (min)",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(t=40, b=30, l=50, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(tickvals=df["year"].tolist(), tickformat="d")

    # Stats table — show estimated ages for pre-2006 rows where exact age is NULL
    base_cols = ["year", "net_time", "gun_time", "overall_place", "sex_place",
                 "age_group", "ag_rank_of_total", "city", "state"]
    table_df = df[base_cols].copy()
    table_df.insert(7, "Age", _compute_estimated_ages(df))
    col_names = ["Year", "Net", "Gun", "Overall", "Sex place",
                 "Age group", "AG rank", "Age", "City", "State"]
    # Show name variant column only when multiple variants are present
    if df["name"].nunique() > 1:
        table_df.insert(1, "Name", df["name"])
        col_names.insert(1, "Name")
    table_df.columns = col_names

    return dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Hr(),
            dash_table.DataTable(
                data=table_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in table_df.columns],
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 13, "padding": "4px 8px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                sort_action="native",
            ),
        ])
    ], className="mb-3 shadow-sm")


def _name_options(search: str | None, selected: str | None) -> list[dict]:
    keys: set[str] = set()
    if selected:
        keys.add(selected)
    if search and len(search) >= 1:
        key_search = normalize_name(search)
        df = query(
            "SELECT DISTINCT name_key FROM results WHERE name_key LIKE ? LIMIT 50",
            (f"%{key_search}%",),
        )
        keys.update(df["name_key"].tolist())

    if not keys:
        return []

    # Single query for all variants across all matching keys
    ph = ",".join("?" * len(keys))
    variants_df = query(
        f"SELECT name_key, name, COUNT(*) AS c "
        f"FROM results WHERE name_key IN ({ph}) "
        f"GROUP BY name_key, name ORDER BY name_key, c DESC",
        tuple(keys),
    )

    by_key: dict[str, list[str]] = {}
    for _, row in variants_df.iterrows():
        by_key.setdefault(row["name_key"], []).append(row["name"])

    options = []
    for key, variants in sorted(by_key.items()):
        primary = variants[0]
        others  = [v for v in variants[1:] if v != primary]
        label   = primary if not others else f"{primary}  (also: {', '.join(others[:3])})"
        options.append({"label": label, "value": key})
    return options


@app.callback(
    Output("runner-search", "options"),
    Input("runner-search", "search_value"),
    # Input (not State) so a value set from a shared link gets its label.
    Input("runner-search", "value"),
)
def update_search_options(search, selected):
    return _name_options(search, selected)


@app.callback(
    Output("runner-search-2", "options"),
    Input("runner-search-2", "search_value"),
    Input("runner-search-2", "value"),
)
def update_search_options_2(search, selected):
    return _name_options(search, selected)


def fmt_margin(sec: float) -> str:
    sec = int(abs(sec))
    if sec < 60:
        return f"{sec}s"
    return f"{sec // 60}:{sec % 60:02d}"


def head_to_head_card(name1: str, df1: pd.DataFrame,
                      name2: str, df2: pd.DataFrame) -> dbc.Card:
    d1 = df1[["year", "net_time_sec", "net_time"]].dropna(subset=["net_time_sec"])
    d2 = df2[["year", "net_time_sec", "net_time"]].dropna(subset=["net_time_sec"])
    merged = d1.merge(d2, on="year", suffixes=("_1", "_2")).sort_values("year")

    if merged.empty:
        return dbc.Card(dbc.CardBody(
            dbc.Alert(f"{name1} and {name2} have no years in common.", color="secondary")
        ), className="mt-3")

    n = len(merged)
    merged["margin_sec"] = merged["net_time_sec_1"] - merged["net_time_sec_2"]
    merged["winner"] = merged["margin_sec"].apply(
        lambda d: name1 if d < 0 else (name2 if d > 0 else "Tie")
    )
    merged["margin_str"] = merged["margin_sec"].apply(fmt_margin)

    w1   = (merged["winner"] == name1).sum()
    w2   = (merged["winner"] == name2).sum()
    ties = (merged["winner"] == "Tie").sum()

    # --- Series summary text ---
    short1 = name1.split()[0]
    short2 = name2.split()[0]

    if w1 > w2:
        rec = f"{w1}–{w2}" + (f"–{ties}" if ties else "")
        headline = f"{short1} leads the series {rec}"
    elif w2 > w1:
        rec = f"{w2}–{w1}" + (f"–{ties}" if ties else "")
        headline = f"{short2} leads the series {rec}"
    else:
        rec = f"{w1}–{w2}" + (f"–{ties}" if ties else "")
        headline = f"Dead even — {rec}"

    headline += f" across {n} shared race{'s' if n > 1 else ''}."

    # Biggest margin
    contested = merged[merged["winner"] != "Tie"]
    extras = []
    if not contested.empty:
        big = contested.loc[contested["margin_sec"].abs().idxmax()]
        extras.append(
            f"Biggest beatdown: {big['winner'].split()[0]} by {fmt_margin(big['margin_sec'])} in {big['year']}."
        )

    # Closest race
    close = merged.loc[merged["margin_sec"].abs().idxmin()]
    if abs(close["margin_sec"]) <= 10:
        extras.append(f"Photo finish in {close['year']}: only {fmt_margin(close['margin_sec'])} apart!")
    elif abs(close["margin_sec"]) <= 60:
        extras.append(f"Closest race: {fmt_margin(close['margin_sec'])} in {close['year']}.")

    # Win streak
    recent = merged.sort_values("year", ascending=False)
    streak_w = recent.iloc[0]["winner"]
    if streak_w != "Tie":
        streak = sum(1 for _ in (
            r for _, r in recent.iterrows()
            if r["winner"] == streak_w
        ) if True)
        # count only leading streak
        streak = 0
        for _, row in recent.iterrows():
            if row["winner"] == streak_w:
                streak += 1
            else:
                break
        if streak >= 2:
            extras.append(f"{streak_w.split()[0]} has won the last {streak} in a row.")

    summary = " ".join([headline] + extras)

    # --- Margin bar chart ---
    bar_colors = merged["margin_sec"].apply(
        lambda v: "#e6500a" if v < 0 else ("#1a6eb5" if v > 0 else "#aaa")
    )
    hover = merged.apply(
        lambda r: (f"{r['winner'].split()[0]} wins by {fmt_margin(r['margin_sec'])}"
                   if r["winner"] != "Tie" else "Tie"),
        axis=1,
    )
    fig = go.Figure(go.Bar(
        x=merged["year"],
        y=(-merged["margin_sec"] / 60).round(2),   # positive = name1 wins
        marker_color=bar_colors.tolist(),
        hovertemplate="<b>%{x}</b>  %{customdata}<extra></extra>",
        customdata=hover.tolist(),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#666", line_width=1)
    fig.update_layout(
        title=dict(text=f"<b>{short1}</b> <span style='color:#e6500a'>▲</span>  vs  "
                        f"<span style='color:#1a6eb5'>▼</span> <b>{short2}</b>  —  margin per year",
                   font_size=13),
        xaxis=dict(tickvals=merged["year"].tolist(), tickformat="d"),
        yaxis_title="Minutes (+ = {} wins)".format(short1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=260, margin=dict(t=45, b=30, l=55, r=15),
        showlegend=False,
    )

    # --- Per-year table ---
    rows = []
    for _, r in merged.iterrows():
        t1, t2 = r["net_time_1"], r["net_time_2"]
        w = r["winner"]
        rows.append({
            "Year":   int(r["year"]),
            name1:    ("🏆 " if w == name1 else "") + t1,
            name2:    ("🏆 " if w == name2 else "") + t2,
            "Gap":    ("🤝 Tie" if w == "Tie"
                       else f"{w.split()[0]} +{fmt_margin(r['margin_sec'])}"),
        })
    tdf = pd.DataFrame(rows)

    return dbc.Card([
        dbc.CardHeader(html.H5(f"⚔️  {short1} vs {short2}", className="mb-0 fw-bold")),
        dbc.CardBody([
            dbc.Alert(summary, color="warning", className="mb-3 fst-italic"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Hr(className="my-2"),
            dash_table.DataTable(
                data=tdf.to_dict("records"),
                columns=[{"name": c, "id": c} for c in tdf.columns],
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 13, "padding": "4px 10px", "textAlign": "center"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa", "textAlign": "center"},
                style_data_conditional=[
                    {"if": {"column_id": name1, "filter_query": f"{{{name1}}} contains 🏆"},
                     "backgroundColor": "#fff3e0", "fontWeight": "bold"},
                    {"if": {"column_id": name2, "filter_query": f"{{{name2}}} contains 🏆"},
                     "backgroundColor": "#e3f2fd", "fontWeight": "bold"},
                ],
            ),
        ]),
    ], className="mt-3 shadow")


@app.callback(
    Output("runner-output", "children"),
    Input("runner-search", "value"),
    Input("runner-search-2", "value"),
    Input("name-links-store", "data"),
)
def update_runner_output(key1, key2, store_data):
    if not key1 and not key2:
        return dbc.Alert(
            "Search for a runner above to see their history across all Turkey Trot years.",
            color="info",
        )

    store_data = store_data or []
    COLORS = ["#e6500a", "#1a6eb5", "#2ca02c", "#9467bd"]
    col_width = 6 if key2 else 12
    cards = []
    color_idx = 0
    h2h_persons: list[tuple[str, pd.DataFrame]] = []

    for key in [key1, key2]:
        if not key:
            continue
        all_keys = find_linked_group(key, store_data)
        df = runner_history_multi(all_keys)
        primary_name = df["name"].value_counts().idxmax() if not df.empty else key
        persons = split_into_persons(df)
        multi = len(persons) > 1
        for p_idx, person_df in enumerate(persons):
            label = person_label(primary_name, person_df) if multi else None
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1
            cards.append(dbc.Col(runner_card(primary_name, person_df, color, label), md=col_width))
            if p_idx == 0:
                display = label or primary_name
                h2h_persons.append((display, person_df))

    output = [dbc.Row(cards)]

    if len(h2h_persons) == 2:
        (n1, p1), (n2, p2) = h2h_persons
        output.append(head_to_head_card(n1, p1, n2, p2))

    return output


# ---------------------------------------------------------------------------
# Name-link callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("link-section", "style"),
    Input("runner-search", "value"),
)
def toggle_link_section(key1):
    return {} if key1 else {"display": "none"}


@app.callback(
    Output("linked-names-display", "children"),
    Input("runner-search", "value"),
    Input("name-links-store", "data"),
)
def render_linked_names(key1, store):
    if not key1:
        return html.Div()
    linked = find_linked_group(key1, store or []) - {key1}
    if not linked:
        return html.Small("No linked names.", className="text-muted")
    chips = []
    for k in sorted(linked):
        chips.append(
            dbc.Button(
                [k, html.Span(" ×", className="ms-1")],
                id={"type": "remove-link-btn", "key": k},
                size="sm", color="secondary", outline=True,
                n_clicks=0, className="me-1 mb-1 py-0",
                style={"fontSize": "12px"},
            )
        )
    return html.Div([html.Small("Linked: ", className="text-muted me-1"), *chips])


@app.callback(
    Output("link-candidate", "options"),
    Input("link-candidate", "search_value"),
    State("link-candidate", "value"),
)
def update_link_candidate_options(search, selected):
    return _name_options(search, selected)


@app.callback(
    Output("link-preview", "children"),
    Output("link-add-btn", "disabled"),
    Input("link-candidate", "value"),
    State("runner-search", "value"),
    State("name-links-store", "data"),
    prevent_initial_call=True,
)
def preview_link(candidate, primary, store):
    if not candidate or not primary:
        return html.Div(), True
    if candidate == primary:
        return dbc.Alert("That's the same name.", color="secondary", className="py-1 small"), True
    linked = find_linked_group(primary, store or [])
    if candidate in linked:
        return dbc.Alert("Already linked.", color="info", className="py-1 small"), True
    compat, msg = check_link_compat(primary, candidate)
    if compat:
        return (
            dbc.Alert([html.Strong("Compatible. "), msg], color="success", className="py-1 small"),
            False,
        )
    return (
        dbc.Alert([html.Strong("Conflict. "), msg], color="danger", className="py-1 small"),
        True,
    )


@app.callback(
    Output("name-links-store", "data"),
    Output("link-candidate", "value"),
    Input("link-add-btn", "n_clicks"),
    State("runner-search", "value"),
    State("link-candidate", "value"),
    State("name-links-store", "data"),
    prevent_initial_call=True,
)
def do_add_link(n, primary, candidate, store):
    if not n or not primary or not candidate:
        raise dash.exceptions.PreventUpdate
    return _add_link(primary, candidate, store or []), None


@app.callback(
    Output("name-links-store", "data", allow_duplicate=True),
    Input({"type": "remove-link-btn", "key": ALL}, "n_clicks"),
    State("name-links-store", "data"),
    prevent_initial_call=True,
)
def do_remove_link(n_clicks_list, store):
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate
    return _remove_from_link_group(triggered["key"], store or [])


# ---------------------------------------------------------------------------
# Sharing: links (?runner=…&vs=…), preview image, and link-preview meta tags
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _key_exists(key: str) -> bool:
    return bool(query("SELECT 1 AS x FROM results WHERE name_key = ? LIMIT 1", (key,)).shape[0])


def parse_share_args(args) -> tuple[str | None, str | None]:
    """Read and validate runner keys from a query-string mapping."""
    def _clean(v):
        v = (v or "").strip().lower()[:100]
        return v if v and _key_exists(v) else None
    k1, k2 = _clean(args.get("runner")), _clean(args.get("vs"))
    if not k1 and k2:
        k1, k2 = k2, None
    if k1 == k2:
        k2 = None
    return k1, k2


def share_query(k1: str | None, k2: str | None) -> str:
    params = {k: v for k, v in (("runner", k1), ("vs", k2)) if v}
    return urlencode(params)


def _share_person(key: str) -> tuple[str, pd.DataFrame]:
    """Display label and results for the first person under a name key,
    matching what the Runner Lookup tab shows first."""
    df = runner_history(key)
    primary = df["name"].value_counts().idxmax()
    persons = split_into_persons(df)
    person = persons[0]
    label = person_label(primary, person) if len(persons) > 1 else primary
    return label, person


def h2h_record(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[int, int, int]:
    """(wins_1, wins_2, ties) across years both runners finished."""
    d1 = df1[["year", "net_time_sec"]].dropna()
    d2 = df2[["year", "net_time_sec"]].dropna()
    m = d1.merge(d2, on="year", suffixes=("_1", "_2"))
    diff = m["net_time_sec_1"] - m["net_time_sec_2"]
    return int((diff < 0).sum()), int((diff > 0).sum()), int((diff == 0).sum())


def share_summary(k1: str | None, k2: str | None) -> tuple[str, str]:
    """(title, one-line description) for a shared link."""
    if not k1:
        return ("Buffalo Turkey Trot Results",
                f"Look up any runner's results from {YEARS[0]} to {YEARS[-1]}.")
    n1, p1 = _share_person(k1)
    if not k2:
        best = p1.loc[p1["net_time_sec"].idxmin()] if p1["net_time_sec"].notna().any() else None
        races = len(p1)
        desc = f"{races} Turkey Trot{'s' if races != 1 else ''}"
        desc += f" ({int(p1['year'].min())}–{int(p1['year'].max())})" if races > 1 else f" ({int(p1['year'].iloc[0])})"
        if best is not None:
            desc += f" · best {fmt_time(best['net_time_sec'])} in {int(best['year'])}"
        return f"{n1} · Buffalo Turkey Trot", desc
    n2, p2 = _share_person(k2)
    w1, w2, ties = h2h_record(p1, p2)
    n = w1 + w2 + ties
    s1, s2 = n1.split()[0], n2.split()[0]
    if n == 0:
        desc = "No shared races yet. Compare their Turkey Trot histories."
    elif w1 == w2:
        desc = f"Dead even at {w1}–{w2}" + (f"–{ties}" if ties else "") + f" across {n} shared races."
    else:
        lead, hi, lo = (s1, w1, w2) if w1 > w2 else (s2, w2, w1)
        desc = f"{lead} leads {hi}–{lo}" + (f"–{ties}" if ties else "") + f" across {n} shared race{'s' if n != 1 else ''}."
    return f"{n1} vs {n2} · Buffalo Turkey Trot", desc


_CARD_COLORS = ["#e6500a", "#1a6eb5"]

# Card layouts. Positions are fractions of the figure (0 = bottom, 1 = top);
# font sizes are in points at 100 dpi, so 1 pt ≈ 1.4 px.
CARD_SIZES = {
    # Link previews (Facebook, iMessage, Slack, X): 1200×630
    "landscape": dict(fig=(12, 6.3), band=0.12, header_fs=17, stack=False,
                      head_y=0.78, head_fs=34, desc_y=0.68, desc_fs=18, wrap=90,
                      ax=[0.09, 0.15, 0.87, 0.45], tick_fs=11, label_fs=12,
                      url_y=0.04, url_fs=12, url_band=False),
    # Instagram feed post (4:5): 1080×1350
    "portrait": dict(fig=(10.8, 13.5), band=0.07, header_fs=22, stack=True,
                     head_y=0.88, head_fs=50, desc_y=0.70, desc_fs=24, wrap=42,
                     ax=[0.15, 0.17, 0.78, 0.43], tick_fs=16, label_fs=17,
                     url_y=0.05, url_fs=22, url_band=True, url_to_bottom=True),
    # Instagram story (9:16): 1080×1920. Instagram's own UI covers roughly the
    # top 13% and bottom 18%, so the important content stays between those.
    "story": dict(fig=(10.8, 19.2), band=0.16, header_y=0.875, header_fs=26, stack=True,
                  head_y=0.80, head_fs=56, desc_y=0.64, desc_fs=28, wrap=38,
                  ax=[0.16, 0.30, 0.77, 0.27], tick_fs=18, label_fs=19,
                  url_y=0.225, url_fs=24, url_band=True, url_to_bottom=True),
}


def _fit_text(fig, text_artist, max_frac: float = 0.92, min_fs: float = 14) -> None:
    """Shrink a text artist's font until it fits within max_frac of the width."""
    renderer = fig.canvas.get_renderer()
    width = text_artist.get_window_extent(renderer).width
    max_width = max_frac * fig.bbox.width
    if width > max_width:
        text_artist.set_fontsize(max(min_fs, text_artist.get_fontsize() * max_width / width))


@lru_cache(maxsize=512)
def build_preview_png(k1: str | None, k2: str | None, share_url: str,
                      size: str = "landscape") -> bytes:
    """Shareable PNG card in one of CARD_SIZES. Uses the object-oriented
    Matplotlib API (no pyplot), which is safe across threads."""
    L = CARD_SIZES.get(size, CARD_SIZES["landscape"])
    fig = Figure(figsize=L["fig"], dpi=100, facecolor="#fffaf3")
    FigureCanvasAgg(fig)
    x0 = 0.06 if L["stack"] else 0.04

    # Header band
    fig.patches.append(Rectangle((0, 1 - L["band"]), 1, L["band"], transform=fig.transFigure,
                                 facecolor="#f39c12", zorder=0))
    fig.text(x0, L.get("header_y", 1 - L["band"] / 2), "BUFFALO TURKEY TROT RESULTS",
             fontsize=L["header_fs"],
             weight="bold", color="#3b2a10", va="center")

    title, desc = share_summary(k1, k2)
    ax = fig.add_axes(L["ax"])
    ax.set_facecolor("#fffaf3")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if not k1:
        df = year_summary()
        heading = (f"Every finisher\nsince {YEARS[0]}" if L["stack"]
                   else f"Every finisher since {YEARS[0]}")
        ax.bar(df["year"], df["finishers"], color="#e6500a", width=0.8)
        ax.set_ylabel("Finishers", fontsize=L["label_fs"], color="#555")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    else:
        people = [_share_person(k1)] + ([_share_person(k2)] if k2 else [])
        # Tall cards stack the names; wide cards keep them on one line.
        heading = ("\nvs\n" if L["stack"] else "  vs  ").join(n for n, _ in people)
        for (name, p), color in zip(people, _CARD_COLORS):
            p = p.dropna(subset=["net_time_sec"])
            ax.plot(p["year"], p["net_time_sec"] / 60, marker="o",
                    lw=2.5 if not L["stack"] else 3.5, ms=7 if not L["stack"] else 10,
                    color=color, label=name)
        ax.invert_yaxis()  # faster = higher
        ax.set_ylabel("Net time", fontsize=L["label_fs"], color="#555")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_time(v * 60)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10 if not L["stack"] else 6))
        if k2:
            ax.legend(frameon=False, fontsize=L["label_fs"], loc="best")

    head = fig.text(x0, L["head_y"], heading, fontsize=L["head_fs"], weight="bold",
                    color="#222", va="center" if not L["stack"] else "top",
                    linespacing=1.15)
    _fit_text(fig, head, max_frac=0.88 if L["stack"] else 0.92)
    desc_text = textwrap.fill(desc, L["wrap"])
    if L["stack"]:
        # Headings are 1–3 lines tall; keep the summary just below the heading.
        renderer = fig.canvas.get_renderer()
        head_bottom = head.get_window_extent(renderer).y0 / fig.bbox.height
        desc_artist = fig.text(x0, head_bottom - 0.035, desc_text, fontsize=L["desc_fs"],
                               color="#555", va="top", linespacing=1.3)
        # Let the chart grow up into whatever space the text leaves free.
        desc_bottom = desc_artist.get_window_extent(renderer).y0 / fig.bbox.height
        ax_x, ax_y, ax_w, _ = L["ax"]
        ax.set_position([ax_x, ax_y, ax_w, desc_bottom - 0.06 - ax_y])
    else:
        fig.text(x0, L["desc_y"], desc_text, fontsize=L["desc_fs"],
                 color="#555", va="center", linespacing=1.3)

    ax.tick_params(labelsize=L["tick_fs"], colors="#555")
    ax.grid(axis="y", color="#e8e0d4", lw=0.8)
    ax.set_axisbelow(True)

    # Site address. On Instagram it can't be tapped, so tall cards make it a
    # prominent call to action.
    display_url = re.sub(r"^https?://", "", share_url).rstrip("/")
    if L["url_band"]:
        band_h = L["url_fs"] * 4.2 / (L["fig"][1] * 100)
        band_top = L["url_y"] + band_h / 2
        band_bottom = 0 if L.get("url_to_bottom") else L["url_y"] - band_h / 2
        fig.patches.append(Rectangle((0, band_bottom), 1, band_top - band_bottom,
                                     transform=fig.transFigure, facecolor="#3b2a10", zorder=0))
        fig.text(0.5, L["url_y"] + band_h * 0.17, "Look up your results",
                 fontsize=L["url_fs"] * 0.75, color="#f8d9a0", ha="center", va="center")
        url = fig.text(0.5, L["url_y"] - band_h * 0.17, display_url, fontsize=L["url_fs"],
                       weight="bold", color="white", ha="center", va="center")
        _fit_text(fig, url, max_frac=0.9)
    else:
        fig.text(x0, L["url_y"], display_url, fontsize=L["url_fs"], color="#8a6d3b", va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


def _share_url_for(k1, k2) -> str:
    root = flask.request.url_root  # honours X-Forwarded-* via ProxyFix
    q = share_query(k1, k2)
    return root + ("?" + q if q else "")


@server.route("/preview.png")
def preview_png():
    k1, k2 = parse_share_args(flask.request.args)
    size = flask.request.args.get("size", "landscape")
    if size not in CARD_SIZES:
        size = "landscape"
    png = build_preview_png(k1, k2, _share_url_for(k1, k2), size)
    resp = flask.Response(png, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    if flask.request.args.get("download"):
        fname = re.sub(r"[^a-z0-9]+", "-", "-vs-".join(k for k in (k1, k2) if k) or "turkey-trot")
        suffix = "" if size == "landscape" else f"-{size}"
        resp.headers["Content-Disposition"] = f'attachment; filename="{fname}{suffix}.png"'
    return resp


def share_meta_tags() -> tuple[str, str] | None:
    """(title, extra <meta> HTML) for the page being requested, so link
    previews (iMessage, Slack, Facebook, X…) show the right card. Crawlers
    don't run JavaScript, so this must be in the server-rendered HTML."""
    if not flask.has_request_context():
        return None
    k1, k2 = parse_share_args(flask.request.args)
    title, desc = share_summary(k1, k2)
    q = share_query(k1, k2)
    root = flask.request.url_root
    image = root + "preview.png" + ("?" + q if q else "")
    page = root + ("?" + q if q else "")
    e = lambda s: html_escape(s, quote=True)
    tags = [
        f'<meta name="description" content="{e(desc)}">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{e(title)}">',
        f'<meta property="og:description" content="{e(desc)}">',
        f'<meta property="og:url" content="{e(page)}">',
        f'<meta property="og:image" content="{e(image)}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{e(title)}">',
        f'<meta name="twitter:description" content="{e(desc)}">',
        f'<meta name="twitter:image" content="{e(image)}">',
    ]
    return title, "\n".join(tags)


@app.callback(
    Output("runner-search", "value"),
    Output("runner-search-2", "value"),
    # Dash 4 dropdowns drop a value that isn't in their options, so the
    # options have to arrive in the same update as the value.
    Output("runner-search", "options", allow_duplicate=True),
    Output("runner-search-2", "options", allow_duplicate=True),
    Output("main-tabs", "active_tab"),
    Input("url", "search"),
    prevent_initial_call="initial_duplicate",
)
def load_shared_link(search):
    """Opening a shared link pre-fills the runners and shows Runner Lookup."""
    k1, k2 = parse_share_args({k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()})
    if not k1:
        raise dash.exceptions.PreventUpdate
    return (k1, k2, _name_options(None, k1),
            _name_options(None, k2) if k2 else dash.no_update, "tab-lookup")


@app.callback(
    Output("share-bar", "style"),
    Output("download-landscape", "href"),
    Output("download-portrait", "href"),
    Output("download-story", "href"),
    Input("runner-search", "value"),
    Input("runner-search-2", "value"),
)
def update_share_bar(k1, k2):
    # With no runner selected, the buttons share the site's general card.
    k1, k2 = (k1, k2) if k1 else (k2, None)
    q = share_query(k1, k2)
    prefix = f"/preview.png?{q}&" if q else "/preview.png?"
    return ({}, *(f"{prefix}size={s}&download=1" for s in ("landscape", "portrait", "story")))


app.clientside_callback(
    """
    async function(n, k1, k2) {
        if (!n) { return window.dash_clientside.no_update; }
        if (!k1) { k1 = k2; k2 = null; }
        const p = new URLSearchParams();
        if (k1) { p.set("runner", k1); }
        if (k2 && k2 !== k1) { p.set("vs", k2); }
        const q = p.toString();
        const url = location.origin + location.pathname + (q ? "?" + q : "");
        const isMobile = /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent);
        if (isMobile && navigator.share) {
            try { await navigator.share({title: "Buffalo Turkey Trot results", url: url}); }
            catch (e) { /* user cancelled */ }
            return "";
        }
        try {
            await navigator.clipboard.writeText(url);
            return "✓ Link copied";
        } catch (e) {
            window.prompt("Copy this link:", url);
            return "";
        }
    }
    """,
    Output("share-status", "children"),
    Input("share-btn", "n_clicks"),
    State("runner-search", "value"),
    State("runner-search-2", "value"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Year Overview callbacks
# ---------------------------------------------------------------------------

def _require_tab(active: str | None, wanted: str) -> None:
    """Only draw a tab's charts while it is visible.

    Plotly measures its container when a figure is drawn; inside a hidden
    (display:none) tab that width is wrong, so charts come out clipped or
    squashed. Drawing on tab activation also stops hidden tabs from
    recomputing every time the user switches tabs.
    """
    if active != wanted:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("year-finishers", "figure"),
    Output("year-avg-time", "figure"),
    Output("year-gender-split", "figure"),
    Input("main-tabs", "active_tab"),
)
def update_year_tab(tab="tab-year"):
    _require_tab(tab, "tab-year")
    df = year_summary()

    fig_fin = px.bar(df, x="year", y="finishers",
                     title="Finishers per year",
                     labels={"year": "Year", "finishers": "Finishers"},
                     color_discrete_sequence=["#e6500a"])
    fig_fin.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          showlegend=False, height=300)

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=df["year"], y=df["avg_male_min"],
                                  name="Male avg", mode="lines+markers",
                                  line=dict(color="#1a6eb5")))
    fig_time.add_trace(go.Scatter(x=df["year"], y=df["avg_female_min"],
                                  name="Female avg", mode="lines+markers",
                                  line=dict(color="#e63480")))
    fig_time.add_trace(go.Scatter(x=df["year"], y=df["avg_net_min"],
                                  name="Overall avg", mode="lines+markers",
                                  line=dict(color="#555", dash="dot")))
    fig_time.update_layout(title="Average net time by year (min)",
                           xaxis_title="Year", yaxis_title="Minutes",
                           yaxis=dict(autorange="reversed"),
                           plot_bgcolor="white", paper_bgcolor="white",
                           height=300)

    fig_gender = go.Figure()
    fig_gender.add_trace(go.Bar(x=df["year"], y=df["male_count"], name="Male",
                                marker_color="#1a6eb5"))
    fig_gender.add_trace(go.Bar(x=df["year"], y=df["female_count"], name="Female",
                                marker_color="#e63480"))
    fig_gender.update_layout(barmode="stack", title="Gender split by year",
                              xaxis_title="Year", yaxis_title="Finishers",
                              plot_bgcolor="white", paper_bgcolor="white",
                              height=300)

    return fig_fin, fig_time, fig_gender


def _filter_age_bracket(df: pd.DataFrame, bracket: str) -> pd.DataFrame:
    """Filter a results dataframe to rows matching the given age bracket string."""
    if bracket == "All":
        return df
    if bracket.endswith("+"):
        lo, hi = int(bracket[:-1]), 999.0
    else:
        lo_s, hi_s = bracket.split("-")
        lo, hi = int(lo_s), int(hi_s)

    exact = pd.to_numeric(df["age"], errors="coerce")
    has_exact = exact.notna()

    # Exact age: simple range check
    exact_ok = has_exact & (exact >= lo) & (exact <= hi)

    # Age-group bounds: ranges must overlap [lo, hi]
    ag_lo = df["age_group"].map(lambda ag: age_group_bounds(ag)[0] if age_group_bounds(ag) else 0.0)
    ag_hi = df["age_group"].map(lambda ag: age_group_bounds(ag)[1] if age_group_bounds(ag) else 999.0)
    ag_ok = (~has_exact) & (ag_lo <= hi) & (ag_hi >= lo)

    return df[exact_ok | ag_ok]


@app.callback(
    Output("year-dist", "figure"),
    Output("year-dist-stats", "children"),
    Input("dist-year-a", "value"),
    Input("dist-year-b", "value"),
    Input("dist-sex-filter", "value"),
    Input("dist-age-bracket", "value"),
    Input("dist-norm", "value"),
    Input("main-tabs", "active_tab"),
)
def update_dist(year_a, year_b, sex_filter, age_bracket, histnorm, tab="tab-year"):
    _require_tab(tab, "tab-year")
    raw = two_year_dist_data(year_a, year_b)

    def _prep(yr):
        d = raw[raw["year"] == yr].copy()
        if sex_filter != "Both":
            d = d[d["sex"] == sex_filter]
        d = _filter_age_bracket(d, age_bracket)
        d["net_min"] = d["net_time_sec"] / 60.0
        return d.dropna(subset=["net_min"])

    dfa = _prep(year_a)
    dfb = _prep(year_b)

    same_year = (year_a == year_b)

    COLOR_A = "#e6500a"
    COLOR_B = "#1a6eb5"

    # Shared bin edges across both datasets so bars align and widths match
    all_mins = pd.concat([dfa["net_min"], dfb["net_min"]]).dropna()
    bin_size = 1.0  # minutes per bin
    bin_start = (all_mins.min() // bin_size) * bin_size if not all_mins.empty else 0
    bin_end   = (all_mins.max() // bin_size + 1) * bin_size if not all_mins.empty else 120
    shared_xbins = dict(start=bin_start, end=bin_end, size=bin_size)

    yaxis_label = "% of finishers" if histnorm == "percent" else "Count"
    sex_label = {"M": "Men", "F": "Women", "Both": "All runners"}[sex_filter]

    # Same year picked twice → draw it once.
    series = [(dfa, year_a, COLOR_A)] + ([] if same_year else [(dfb, year_b, COLOR_B)])

    fig = go.Figure()
    for df_i, yr, color in series:
        if df_i.empty:
            continue
        med = df_i["net_min"].median()
        fig.add_trace(go.Histogram(
            x=df_i["net_min"],
            name=f"{yr} (median {med:.1f} min)",
            histnorm=histnorm,
            xbins=shared_xbins,
            marker_color=color,
            opacity=0.55 if not same_year else 0.75,
        ))
        # Median line; the value lives in the legend so labels never collide.
        fig.add_vline(x=med, line_dash="dash", line_color=color, line_width=1.5)

    if not fig.data:
        fig.add_annotation(text="No finishers match these filters.",
                           showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(size=14, color="#888"))

    title_years = str(year_a) if same_year else f"{year_a} vs {year_b}"
    age_label = "" if age_bracket == "All" else f" · {age_bracket}"
    fig.update_layout(
        barmode="overlay",
        title=f"Finish time distribution — {title_years} — {sex_label}{age_label}",
        xaxis_title="Net time (min)",
        yaxis_title=yaxis_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        legend=dict(x=0.99, y=0.97, xanchor="right", bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=50, b=40, l=55, r=20),
    )

    # Stats summary table
    def _stats(df_i, yr):
        if df_i.empty:
            return None
        v = df_i["net_min"]
        return {
            "Year": yr,
            "n": f"{len(df_i):,}",
            "Median": f"{v.median():.1f}",
            "Mean": f"{v.mean():.1f}",
            "P25": f"{v.quantile(0.25):.1f}",
            "P75": f"{v.quantile(0.75):.1f}",
            "P90": f"{v.quantile(0.90):.1f}",
        }

    stat_rows = [s for s in (_stats(d, y) for d, y, _ in series) if s]

    stats_block = html.Div([
        html.P(f"{sex_label}", className="fw-semibold text-muted small mb-2"),
        dash_table.DataTable(
            data=stat_rows,
            columns=[{"name": c, "id": c} for c in ["Year","n","Median","Mean","P25","P75","P90"]],
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": 12, "padding": "4px 6px", "textAlign": "center"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa",
                          "fontSize": 11, "textAlign": "center"},
            style_data_conditional=[
                {"if": {"row_index": 0}, "color": COLOR_A, "fontWeight": "bold"},
                {"if": {"row_index": 1}, "color": COLOR_B, "fontWeight": "bold"},
            ],
        ),
    ], className="mt-2")

    return fig, stats_block


# ---------------------------------------------------------------------------
# Corral Analysis callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("corral-scatter", "figure"),
    Output("corral-stats", "children"),
    Output("corral-by-ag", "figure"),
    Input("corral-year-slider", "value"),
    Input("main-tabs", "active_tab"),
)
def update_corral(year, tab="tab-corral"):
    _require_tab(tab, "tab-corral")
    df = year_data(year)
    df = df.dropna(subset=["gun_time_sec", "net_time_sec"])
    if df.empty or (df["gun_time_sec"] == df["net_time_sec"]).all():
        empty = go.Figure()
        empty.add_annotation(text=f"No chip-timing data for {year}.", showarrow=False,
                             xref="paper", yref="paper", x=0.5, y=0.5,
                             font=dict(size=14, color="#888"))
        empty.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300,
                            xaxis_visible=False, yaxis_visible=False)
        msg = dbc.Alert(f"{year} only recorded gun times, so wave gaps can't be computed.",
                        color="secondary")
        return empty, msg, empty
    df["wave_gap_sec"] = df["gun_time_sec"] - df["net_time_sec"]
    df["net_min"] = df["net_time_sec"] / 60
    df["wave_gap_min"] = df["wave_gap_sec"] / 60

    fig_scatter = px.scatter(
        df.sample(min(3000, len(df)), random_state=42),
        x="net_min", y="wave_gap_min",
        color="sex",
        opacity=0.4,
        title=f"{year}: Corral seeding — wave gap vs. finish time",
        labels={"net_min": "Net time (min)", "wave_gap_min": "Wave delay (min)", "sex": "Sex"},
        color_discrete_map={"M": "#1a6eb5", "F": "#e63480"},
        hover_data=["name", "age_group"],
    )
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=400)
    fig_scatter.update_traces(marker=dict(size=4))

    # Zone of shame: the slowest quarter by net time (>= P75) who started with
    # the front quarter of the field (wave delay <= P25). Computed on the full
    # field, not the plotted sample.
    net_p75 = df["net_min"].quantile(0.75)
    gap_p25 = df["wave_gap_min"].quantile(0.25)
    in_zone = (df["net_min"] >= net_p75) & (df["wave_gap_min"] <= gap_p25)
    n_zone = int(in_zone.sum())
    pct_zone = 100 * n_zone / len(df)
    fig_scatter.add_shape(
        type="rect", layer="above", line_width=1, line_dash="dot",
        line_color="rgba(200, 30, 30, 0.7)", fillcolor="rgba(230, 60, 60, 0.15)",
        x0=net_p75, x1=df["net_min"].max(),
        y0=min(0.0, df["wave_gap_min"].min()), y1=gap_p25,
    )
    fig_scatter.add_annotation(
        x=df["net_min"].max(), y=gap_p25, xanchor="right", yanchor="bottom",
        text=f"<b>Zone of shame</b> ({n_zone:,} runners)", showarrow=False,
        font=dict(color="#b22222", size=12),
    )

    # Stats
    n_front_jump = (df["wave_gap_sec"] < 60).sum()
    pct_fj = 100 * n_front_jump / len(df)
    corr = df["wave_gap_sec"].corr(df["net_time_sec"])
    stats = [
        html.H6(f"{year} Seeding Stats"),
        html.P(f"Total finishers: {len(df):,}"),
        html.P(f"Median wave gap: {df['wave_gap_sec'].median():.0f}s"),
        html.P(f"<60s gap (near-front): {n_front_jump:,} ({pct_fj:.1f}%)"),
        html.P(f"Correlation (gap vs net time): {corr:.3f}"),
        html.P([
            html.Span("Zone of shame: ", className="fw-semibold", style={"color": "#b22222"}),
            f"{n_zone:,} ({pct_zone:.1f}%)",
            html.Br(),
            html.Small(
                f"Slowest 25% (net ≥ {fmt_time(net_p75 * 60)}) who started in the "
                f"front 25% (wave delay ≤ {fmt_time(gap_p25 * 60)})",
                className="text-muted",
            ),
        ]),
        dbc.Alert(
            "A strong positive correlation means slower runners are correctly "
            "seeding further back. Near-zero or negative suggests front-jumping.",
            color="info", className="mt-2", style={"fontSize": 12}
        ),
    ]

    # By age group — median wave gap
    ag_df = (df.groupby("age_group", as_index=False)
               .agg(median_gap=("wave_gap_sec", "median"),
                    median_net=("net_time_sec", "median"),
                    count=("net_time_sec", "count"))
               .sort_values("median_net"))
    fig_ag = px.bar(
        ag_df, x="age_group", y="median_gap",
        title=f"{year}: Median wave gap by age group",
        labels={"age_group": "Age group", "median_gap": "Median wave gap (sec)"},
        color="median_net", color_continuous_scale="RdYlGn_r",
        hover_data=["count"],
    )
    fig_ag.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=350,
                         xaxis_tickangle=-45)

    return fig_scatter, stats, fig_ag


# ---------------------------------------------------------------------------
# Retention callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("retention-heatmap", "figure"),
    Output("retention-yoy", "figure"),
    Output("retention-streak", "figure"),
    Input("main-tabs", "active_tab"),
)
def update_retention(tab="tab-retention"):
    _require_tab(tab, "tab-retention")
    ret = retention_matrix()

    # Pivot for heatmap. The diagonal (a year vs itself = its whole field) is
    # an order of magnitude bigger than any overlap and washes out the color
    # scale, so blank it out.
    pivot = ret.pivot(index="year_a", columns="year_b", values="shared").astype(float)
    for yr in pivot.index:
        pivot.loc[yr, yr] = np.nan
    fig_heat = px.imshow(
        pivot,
        title="Runner overlap between years (shared finishers, diagonal hidden)",
        labels={"color": "Shared runners"},
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig_heat.update_layout(height=420, xaxis_title="Year B", yaxis_title="Year A")

    # Year-over-year retention rate (consecutive years)
    years = sorted(ret["year_a"].unique())
    yoy = []
    for i in range(len(years) - 1):
        ya, yb = years[i], years[i + 1]
        count_a = ret[(ret["year_a"] == ya) & (ret["year_b"] == ya)]["shared"].values[0]
        count_ab = ret[(ret["year_a"] == ya) & (ret["year_b"] == yb)]["shared"].values[0]
        yoy.append({"years": f"{ya}→{yb}", "retention_pct": 100 * count_ab / count_a})
    yoy_df = pd.DataFrame(yoy)
    fig_yoy = px.bar(
        yoy_df, x="years", y="retention_pct",
        title="Year-over-year return rate (%)",
        labels={"years": "", "retention_pct": "% returned next year"},
        color_discrete_sequence=["#e6500a"],
    )
    fig_yoy.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=300, xaxis_tickangle=-45)

    streak_df = streak_distribution()
    fig_streak = px.bar(
        streak_df, x="streak", y="runners",
        title="Longest consecutive-year streak per runner",
        labels={"streak": "Years in a row", "runners": "Number of runners (log scale)"},
        color_discrete_sequence=["#1a6eb5"],
        log_y=True,  # one-timers outnumber long streaks ~1000:1
    )
    fig_streak.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300)

    return fig_heat, fig_yoy, fig_streak


if __name__ == "__main__":
    app.run(debug=True, port=8050)
