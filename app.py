"""
Buffalo Turkey Trot Results Dashboard
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc

DB_PATH = Path(__file__).parent / "db" / "turkey_trot.db"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_conn():
    return sqlite3.connect(DB_PATH)


def query(sql: str, params=()) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params)


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


def runner_history(name_key: str) -> pd.DataFrame:
    """Fetch all results matching this normalized key (covers all name variants)."""
    return query(
        "SELECT year, name, overall_place, sex_place, age_group, ag_rank_of_total, "
        "gun_time, gun_time_sec, net_time, net_time_sec, city, state, age "
        "FROM results WHERE name_key = ? ORDER BY year",
        (name_key,),
    )


def runner_history_multi(keys: set[str]) -> pd.DataFrame:
    """Fetch results for a set of name_keys combined."""
    if not keys:
        return pd.DataFrame()
    ph = ",".join("?" * len(keys))
    return query(
        f"SELECT year, name, overall_place, sex_place, age_group, ag_rank_of_total, "
        f"gun_time, gun_time_sec, net_time, net_time_sec, city, state, age "
        f"FROM results WHERE name_key IN ({ph}) ORDER BY year",
        tuple(sorted(keys)),
    )


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


def retention_matrix() -> pd.DataFrame:
    """For each pair of years, count runners appearing in both."""
    years_df = query("SELECT DISTINCT year FROM results ORDER BY year")
    years = years_df["year"].tolist()
    names_by_year = {}
    for yr in years:
        df = query("SELECT LOWER(TRIM(name)) AS name FROM results WHERE year=?", (yr,))
        names_by_year[yr] = set(df["name"])

    records = []
    for yr_a in years:
        for yr_b in years:
            if yr_a <= yr_b:
                overlap = len(names_by_year[yr_a] & names_by_year[yr_b])
                records.append({"year_a": yr_a, "year_b": yr_b, "shared": overlap})
                if yr_a != yr_b:
                    records.append({"year_a": yr_b, "year_b": yr_a, "shared": overlap})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Buffalo Turkey Trot",
)
server = app.server  # for deployment

YEARS = query("SELECT DISTINCT year FROM results ORDER BY year")["year"].tolist()

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def fmt_time(secs):
    if pd.isna(secs):
        return "—"
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


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
        dbc.Col(dcc.Graph(id="year-dist"), md=9),
        dbc.Col(html.Div(id="year-dist-stats"), md=3),
    ]),
], fluid=True)


corral_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Year", className="fw-semibold"),
            dcc.Slider(
                id="corral-year-slider",
                min=YEARS[0], max=YEARS[-1],
                step=None,
                marks={y: str(y) for y in YEARS},
                value=YEARS[-1],
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
    State("runner-search", "value"),
)
def update_search_options(search, selected):
    return _name_options(search, selected)


@app.callback(
    Output("runner-search-2", "options"),
    Input("runner-search-2", "search_value"),
    State("runner-search-2", "value"),
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
# Year Overview callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("year-finishers", "figure"),
    Output("year-avg-time", "figure"),
    Output("year-gender-split", "figure"),
    Input("main-tabs", "active_tab"),
)
def update_year_tab(tab):
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
)
def update_dist(year_a, year_b, sex_filter, age_bracket, histnorm):
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
    bin_size = 0.5  # minutes per bin
    bin_start = (all_mins.min() // bin_size) * bin_size if not all_mins.empty else 0
    bin_end   = (all_mins.max() // bin_size + 1) * bin_size if not all_mins.empty else 120
    shared_xbins = dict(start=bin_start, end=bin_end, size=bin_size)

    fig = go.Figure()
    for df_i, yr, color in [(dfa, year_a, COLOR_A), (dfb, year_b, COLOR_B)]:
        if df_i.empty:
            continue
        fig.add_trace(go.Histogram(
            x=df_i["net_min"],
            name=str(yr),
            histnorm=histnorm,
            xbins=shared_xbins,
            marker_color=color,
            opacity=0.55 if not same_year else 0.75,
        ))

    # Median lines
    yaxis_label = "% of finishers" if histnorm == "percent" else "Count"
    sex_label = {"M": "Men", "F": "Women", "Both": "All runners"}[sex_filter]

    for df_i, yr, color, side in [
        (dfa, year_a, COLOR_A, "top right"),
        (dfb, year_b, COLOR_B, "top left"),
    ]:
        if df_i.empty:
            continue
        med = df_i["net_min"].median()
        fig.add_vline(
            x=med, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=f"{yr} med {med:.1f}m",
            annotation_font_color=color,
            annotation_position=side,
        )

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
        legend=dict(x=0.78, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
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

    stat_rows = [s for s in [_stats(dfa, year_a), _stats(dfb, year_b)] if s]

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
    ], className="mt-4")

    return fig, stats_block


# ---------------------------------------------------------------------------
# Corral Analysis callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("corral-scatter", "figure"),
    Output("corral-stats", "children"),
    Output("corral-by-ag", "figure"),
    Input("corral-year-slider", "value"),
)
def update_corral(year):
    df = year_data(year)
    df = df.dropna(subset=["gun_time_sec", "net_time_sec"])
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
def update_retention(tab):
    ret = retention_matrix()

    # Pivot for heatmap
    pivot = ret.pivot(index="year_a", columns="year_b", values="shared")
    fig_heat = px.imshow(
        pivot,
        title="Runner overlap between years (shared finishers)",
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

    # Streak analysis — how many runners appeared in N consecutive years?
    names_by_year = {}
    for yr in years:
        df_yr = query("SELECT LOWER(TRIM(name)) AS name FROM results WHERE year=?", (yr,))
        names_by_year[yr] = set(df_yr["name"])

    all_runner_names = set().union(*names_by_year.values())
    streak_counts: dict[int, int] = {}
    for n in all_runner_names:
        appeared = [1 if n in names_by_year[yr] else 0 for yr in years]
        # longest consecutive run
        best = cur = 0
        for v in appeared:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        streak_counts[best] = streak_counts.get(best, 0) + 1

    streak_df = pd.DataFrame(
        sorted(streak_counts.items()), columns=["streak", "runners"]
    )
    fig_streak = px.bar(
        streak_df, x="streak", y="runners",
        title="Longest consecutive-year streak per runner",
        labels={"streak": "Years in a row", "runners": "Number of runners"},
        color_discrete_sequence=["#1a6eb5"],
    )
    fig_streak.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300)

    return fig_heat, fig_yoy, fig_streak


if __name__ == "__main__":
    app.run(debug=True, port=8050)
