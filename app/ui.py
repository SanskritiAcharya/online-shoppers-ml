"""Streamlit app for the marketing team. Runs on its own (no FastAPI needed), so it can go on
Streamlit Community Cloud. It loads models/model.joblib through src/predict.py.

    streamlit run app/ui.py      then open http://localhost:8501
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from src import config, evaluate, predict

st.set_page_config(page_title="Who should see the discount pop-up?", layout="wide")

ACCENT, MUTED = "#1d5f8a", "#a3abb6"

INPUT_LABELS = {
    "Administrative": "Account pages", "Administrative_Duration": "Time on account pages",
    "Informational": "Info pages", "Informational_Duration": "Time on info pages",
    "ProductRelated": "Product pages", "ProductRelated_Duration": "Time on product pages",
    "BounceRates": "Bounce rate", "ExitRates": "Exit rate", "SpecialDay": "Near a holiday",
    "Month": "Month", "VisitorType": "Visitor type", "OperatingSystems": "Operating system",
    "Browser": "Browser", "Region": "Region", "TrafficType": "Traffic source", "Weekend": "Weekend",
}
VISITOR_PLAIN = {"Returning_Visitor": "Returning", "New_Visitor": "New", "Other": "Other"}
MONTH_NAMES = {"Feb": "February", "Mar": "March", "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
               "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"}

# Starting points for the form. Durations are in minutes here and turned into seconds for the model.
PRESETS = {
    "Serious": {"Administrative": 6, "admin_min": 3.0, "Informational": 1, "info_min": 0.5,
                        "ProductRelated": 60, "product_min": 40.0, "exit_pct": 1.0, "bounce_pct": 0.3,
                        "Month": "Nov", "VisitorType": "Returning_Visitor", "Weekend": False, "SpecialDay": 0.0},
    "Casual": {"Administrative": 1, "admin_min": 0.5, "Informational": 0, "info_min": 0.0,
                       "ProductRelated": 12, "product_min": 6.0, "exit_pct": 4.0, "bounce_pct": 1.0,
                       "Month": "May", "VisitorType": "Returning_Visitor", "Weekend": False, "SpecialDay": 0.0},
    "Bounce": {"Administrative": 0, "admin_min": 0.0, "Informational": 0, "info_min": 0.0,
                     "ProductRelated": 2, "product_min": 0.5, "exit_pct": 15.0, "bounce_pct": 12.0,
                     "Month": "Mar", "VisitorType": "Returning_Visitor", "Weekend": True, "SpecialDay": 0.0},
}
DEVICE_DEFAULTS = {"OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2}

CSS = """
<style>
:root {
  --accent: #1d5f8a; --accent-soft: #eaf1f6; --ink: #1b1f27; --ink-2: #4a5361; --ink-3: #737c8a;
  --line: #e2e5ea; --line-faint: #eef0f3; --well: #f7f8fa;
}
html { -webkit-tap-highlight-color: transparent; -webkit-text-size-adjust: 100%; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], #MainMenu, footer, .stDeployButton { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1200px; padding: 2.25rem 2rem 4rem; }
[data-testid="stMarkdownContainer"] p { line-height: 1.55; }

.app-head { margin-bottom: 22px; }
.app-head .eyebrow { font-size: 13px; color: var(--ink-3); font-weight: 500; margin-bottom: 6px; }
.app-head .title { font-size: 28px; line-height: 1.2; font-weight: 650; color: var(--ink); letter-spacing: -0.01em; }
.app-head .sub { font-size: 15px; color: var(--ink-2); margin-top: 6px; max-width: 760px; line-height: 1.5; }

.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 8px; }
.kpi { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px 15px; background: #fff; }
.kpi .label { font-size: 13px; color: var(--ink-2); font-weight: 500; }
.kpi .value { font-size: 26px; font-weight: 650; color: var(--ink); margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi .value .unit { font-size: 15px; color: var(--ink-3); font-weight: 500; }
.kpi .note { font-size: 12.5px; color: var(--ink-3); margin-top: 2px; line-height: 1.4; }
.kpi.lead .value { color: var(--accent); }
.kpi.muted .value { color: var(--ink-3); }

div[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { padding: 10px 14px; user-select: none; -webkit-user-select: none; touch-action: manipulation; }
button[data-baseweb="tab"] p { font-size: 15px !important; font-weight: 500; }
div[data-baseweb="tab-highlight"] { height: 2px; }
div[data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 22px; }
button, [role="button"] { touch-action: manipulation; }

.section-title { font-size: 17px; font-weight: 600; color: var(--ink); margin: 6px 0 4px; }
.section-note { font-size: 13.5px; color: var(--ink-3); margin-bottom: 12px; line-height: 1.5; }
.group-label { font-size: 12.5px; font-weight: 600; color: var(--ink-3); margin: 2px 0 6px; letter-spacing: 0.01em; }

.pill { display: inline-block; font-size: 12.5px; font-weight: 600; padding: 4px 10px; border-radius: 999px;
  white-space: nowrap; line-height: 1.3; }
.pill.accent { background: var(--accent-soft); color: var(--accent); }
.pill.plain { background: var(--well); color: var(--ink-2); border: 1px solid var(--line); }

.result-card { border: 1px solid var(--line); border-radius: 12px; padding: 20px 22px; }
.result-card .label { font-size: 13px; color: var(--ink-2); font-weight: 500; }
.result-card .big { font-size: 44px; font-weight: 650; color: var(--ink); line-height: 1.1; margin: 6px 0 10px;
  font-variant-numeric: tabular-nums; }
.result-card .action { font-size: 14.5px; color: var(--ink-2); margin-top: 14px; line-height: 1.5; }
.scale { position: relative; height: 8px; border-radius: 4px; background: var(--line-faint); margin: 22px 0 6px; }
.scale .above { position: absolute; top: 0; bottom: 0; right: 0; background: var(--accent-soft); border-radius: 0 4px 4px 0; }
.scale .mark { position: absolute; top: -5px; width: 3px; height: 18px; border-radius: 2px; background: var(--accent); }
.scale .avg { position: absolute; top: -2px; width: 1px; height: 12px; background: var(--ink-3); }
.scale-labels { position: relative; height: 16px; font-size: 12px; color: var(--ink-3); }
.scale-labels span { position: absolute; white-space: nowrap; }

.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0; border: 1px solid var(--line);
  border-radius: 10px; margin: 4px 0 18px; }
.stat { padding: 12px 16px; border-left: 1px solid var(--line-faint); }
.stat:first-child { border-left: none; }
.stat .k { font-size: 12.5px; color: var(--ink-3); font-weight: 500; }
.stat .v { font-size: 20px; font-weight: 650; color: var(--ink); margin-top: 2px; font-variant-numeric: tabular-nums; }
.stat .v.accent { color: var(--accent); }
.stat .s { font-size: 12.5px; color: var(--ink-3); margin-top: 1px; }

.tbl-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; overscroll-behavior-x: contain; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 14px; color: var(--ink); }
table.tbl th { font-size: 12.5px; font-weight: 500; color: var(--ink-3); text-align: left; padding: 8px 12px;
  border-bottom: 1px solid var(--line); white-space: nowrap; }
table.tbl td { padding: 10px 12px; border-bottom: 1px solid var(--line-faint); vertical-align: middle; }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.tbl th.num { text-align: right; }
table.tbl .dim { color: var(--ink-3); }
table.tbl .name { font-weight: 500; }
table.tbl tr.hl td { background: var(--accent-soft); font-weight: 600; }
table.tbl tr.group td { font-size: 12.5px; font-weight: 600; color: var(--ink-3); background: var(--well);
  padding-top: 7px; padding-bottom: 7px; }
table.tbl tr.cut td { border-bottom: 2px solid var(--accent); }
.prob { display: flex; align-items: center; justify-content: flex-end; gap: 10px; }
.prob .track { width: 110px; height: 6px; border-radius: 3px; background: var(--line-faint); overflow: hidden; flex: none; }
.prob .fill { height: 100%; background: var(--accent); border-radius: 3px; }
.prob .fill.low { background: var(--ink-3); opacity: 0.45; }
.prob .val { width: 38px; text-align: right; }
.yes { white-space: nowrap; display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; color: var(--accent); }
.yes::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--accent); }
.no { font-size: 13px; color: var(--ink-3); }

.facts { margin: 0; padding-left: 18px; color: var(--ink-2); font-size: 14px; line-height: 1.55; }
.facts li { margin-bottom: 6px; }
.decision { border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
.decision .row { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0;
  border-bottom: 1px solid var(--line-faint); font-size: 14px; color: var(--ink-2); }
.decision .row:last-child { border-bottom: none; }
.decision .row b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.foot { font-size: 12.5px; color: var(--ink-3); margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--line-faint);
  line-height: 1.5; }
[data-testid="stExpander"] details { border-radius: 10px; }
[data-testid="stExpander"] summary p { font-weight: 500; }

/* iOS zooms into inputs under 16px, so touch screens get 16px */
@media (pointer: coarse) {
  input, textarea, select, [data-baseweb="select"] div { font-size: 16px !important; }
}
@media (max-width: 900px) {
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .stat:nth-child(3) { border-left: none; }
  .stat:nth-child(n+3) { border-top: 1px solid var(--line-faint); }
}
@media (max-width: 640px) {
  /* on a phone the decision goes above the form, so the answer is the first thing you see */
  [data-testid="stHorizontalBlock"]:has(.result-card) > div:nth-child(2) { order: -1; }
  .block-container, [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 3rem; }
  .app-head .title { font-size: 23px; }
  .app-head .sub { font-size: 14.5px; }
  .hide-sm { display: none; }
  .prob .track { display: none; }
  table.tbl.wide { min-width: 520px; }
  table.tbl { font-size: 13.5px; }
  table.tbl td, table.tbl th { padding: 9px 6px; }
  .kpi { padding: 12px 12px 13px; }
  .kpi .value { font-size: 22px; }
  .result-card .big { font-size: 38px; }
  .stat .v { font-size: 18px; }
}
</style>
"""


def esc(text):
    return html.escape(str(text))


def pct(p, digits=0):
    return f"{p:.{digits}%}"


def prob_cell(p, threshold):
    low = " low" if p < threshold else ""
    return (f'<div class="prob"><div class="track"><div class="fill{low}" style="width:{min(p, 1) * 100:.1f}%"></div></div>'
            f'<span class="val">{pct(p)}</span></div>')


def table_html(headers, rows, row_classes=None, wide=False):
    """headers: list of (label, css class). rows: lists of html strings. A row can also be ("group", text)
    for a full-width group label."""
    head = "".join(f'<th class="{cls}">{esc(label)}</th>' for label, cls in headers)
    body = []
    for i, row in enumerate(rows):
        row_cls = row_classes[i] if row_classes else ""
        if isinstance(row, tuple) and row[0] == "group":
            body.append(f'<tr class="group"><td colspan="{len(headers)}">{esc(row[1])}</td></tr>')
            continue
        cells, j = [], 0
        for cell in row:
            # a cell can be ("span", html, n) to run across n columns
            if isinstance(cell, tuple):
                cells.append(f'<td class="dim" colspan="{cell[2]}">{cell[1]}</td>')
                j += cell[2]
            else:
                cells.append(f'<td class="{headers[j][1]}">{cell}</td>')
                j += 1
        cells = "".join(cells)
        body.append(f'<tr class="{row_cls}">{cells}</tr>')
    wide_cls = " wide" if wide else ""
    return (f'<div class="tbl-wrap"><table class="tbl{wide_cls}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


@st.cache_resource
def get_model():
    return predict.load()


@st.cache_data
def test_sessions():
    """Held-out test sessions with the final model's probability (written by train.py)."""
    df = pd.read_csv(config.REPORTS_DIR / "heldout_predictions.csv")
    return df[df["split"] == "test"].reset_index(drop=True)


@st.cache_data
def results_table():
    return pd.read_csv(config.REPORTS_DIR / "results_table.csv").set_index("run")


@st.cache_data
def importance_table():
    return pd.read_csv(config.REPORTS_DIR / "permutation_importance.csv")


model, meta = get_model()
threshold = meta["threshold"]
test_m = meta["test_metrics"]
results = results_table()
tests = test_sessions()
rule_b = results.loc["baseline_rule_B_products_and_low_exit"]
with_pv = results[results["kind"] == "comparison"].iloc[0]
base_rate = test_m["prevalence"]

st.html(CSS)
st.html(f"""
<div class="app-head">
  <div class="eyebrow">Online store, 12,205 browsing sessions, UCI Online Shoppers dataset</div>
  <div class="title">Who should see the discount pop-up?</div>
  <div class="sub">When a visitor is about to leave, the model scores how close they are to buying. The top 10% of
  sessions get a discount pop-up. Everyone else leaves without one.</div>
</div>
<div class="kpis">
  <div class="kpi lead"><div class="label">Pop-ups that reach buyers</div><div class="value">{pct(test_m['precision_at_10pct'])}</div>
    <div class="note">Top 10% of {len(tests):,} test sessions. A random 10%: {pct(base_rate)}</div></div>
  <div class="kpi"><div class="label">PR-AUC on test sessions</div><div class="value">{test_m['pr_auc']:.2f}</div>
    <div class="note">Best rule without page value: {rule_b['test_pr_auc']:.2f}</div></div>
  <div class="kpi"><div class="label">Buyers reached</div><div class="value">{pct(test_m['recall_at_10pct'])}</div>
    <div class="note">Of all test buyers, with pop-ups on 10% of sessions</div></div>
  <div class="kpi muted"><div class="label">PR-AUC if page value were used</div><div class="value">{with_pv['test_pr_auc']:.2f}</div>
    <div class="note">Not deployable. The column partly records the purchase itself</div></div>
</div>
""")

tab_score, tab_tests, tab_how = st.tabs(["Score a session", "Test sessions", "How the model works"])

# ---------------- Score a session ----------------


def apply_preset():
    preset = PRESETS[st.session_state["preset"] or "Serious"]
    for key, value in preset.items():
        st.session_state[key] = value


if "ProductRelated" not in st.session_state:
    st.session_state["preset"] = "Serious"
    apply_preset()
    for key, value in DEVICE_DEFAULTS.items():
        st.session_state[key] = value

with tab_score:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.html('<div class="section-title">Session at exit intent</div>'
                '<div class="section-note">What the site knows when the visitor moves to leave. '
                'Page value is left out on purpose.</div>')
        st.segmented_control("Start from an example session", list(PRESETS), key="preset", on_change=apply_preset)
        with st.container(border=True):
            st.html('<div class="group-label">Pages viewed so far</div>')
            a, b = st.columns(2)
            a.number_input("Product pages", min_value=0, max_value=700, step=1, key="ProductRelated")
            b.number_input("Minutes on product pages", min_value=0.0, max_value=600.0, step=1.0, format="%.1f", key="product_min")
            a, b = st.columns(2)
            a.number_input("Account pages", min_value=0, max_value=30, step=1, key="Administrative")
            b.number_input("Minutes on account pages", min_value=0.0, max_value=60.0, step=0.5, format="%.1f", key="admin_min")
            a, b = st.columns(2)
            a.number_input("Info pages", min_value=0, max_value=25, step=1, key="Informational")
            b.number_input("Minutes on info pages", min_value=0.0, max_value=45.0, step=0.5, format="%.1f", key="info_min")
            st.html('<div class="group-label" style="margin-top:6px">How those pages usually perform</div>')
            a, b = st.columns(2)
            a.slider("Exit rate of pages visited (%)", 0.0, 20.0, step=0.5, key="exit_pct",
                     help="Google Analytics: how often other visitors leave the site from these pages")
            b.slider("Bounce rate of pages visited (%)", 0.0, 20.0, step=0.5, key="bounce_pct")
            st.html('<div class="group-label" style="margin-top:6px">Visit</div>')
            a, b = st.columns(2)
            a.selectbox("Month", config.MONTHS, format_func=lambda m: MONTH_NAMES[m], key="Month")
            b.selectbox("Visitor type", config.VISITOR_TYPES, format_func=lambda v: VISITOR_PLAIN[v], key="VisitorType")
            a, b = st.columns(2)
            a.select_slider("Closeness to a holiday", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], key="SpecialDay",
                            help="1.0 means the session is right before a holiday such as Valentine's Day")
            with b:
                st.write("")
                st.toggle("Weekend", key="Weekend")
            with st.expander("Device and traffic source"):
                a, b = st.columns(2)
                a.selectbox("Operating system code", list(range(1, 9)), key="OperatingSystems")
                b.selectbox("Browser code", list(range(1, 14)), key="Browser")
                a, b = st.columns(2)
                a.selectbox("Region code", list(range(1, 10)), key="Region")
                b.selectbox("Traffic source code", list(range(1, 21)), key="TrafficType")

    s = st.session_state
    record = {
        "Administrative": int(s["Administrative"]), "Administrative_Duration": float(s["admin_min"]) * 60,
        "Informational": int(s["Informational"]), "Informational_Duration": float(s["info_min"]) * 60,
        "ProductRelated": int(s["ProductRelated"]), "ProductRelated_Duration": float(s["product_min"]) * 60,
        "BounceRates": float(s["bounce_pct"]) / 100, "ExitRates": float(s["exit_pct"]) / 100,
        "SpecialDay": float(s["SpecialDay"]), "Month": s["Month"], "VisitorType": s["VisitorType"],
        "OperatingSystems": int(s["OperatingSystems"]), "Browser": int(s["Browser"]), "Region": int(s["Region"]),
        "TrafficType": int(s["TrafficType"]), "Weekend": bool(s["Weekend"]),
    }
    result = predict.predict(record)[0]
    p = result["purchase_probability"]
    if result["show_popup"]:
        pill = '<span class="pill accent">Show the pop-up</span>'
        action = "This visitor looks close to buying and is about to leave. A discount has a real chance to save the sale."
    else:
        pill = '<span class="pill plain">No pop-up</span>'
        action = ("Below the pop-up line. Let this visitor leave and keep the discount for sessions closer to buying.")
    times_avg = p / base_rate

    contrib = pd.DataFrame([{"input": INPUT_LABELS[col], "change": value}
                            for col, value in predict.drivers(record).items()])
    contrib = contrib[contrib["change"].abs() >= 0.005].sort_values("change", ascending=False)
    contrib["direction"] = contrib["change"].map(lambda v: "Raises chance" if v > 0 else "Lowers chance")

    with right:
        st.html(f"""
<div class="section-title">Decision</div>
<div class="section-note">The pop-up line is set so that about 10% of sessions cross it.</div>
<div class="result-card">
  <div class="label">Purchase probability</div>
  <div class="big">{pct(p)}</div>
  {pill}
  <div class="scale">
    <div class="above" style="left:{threshold * 100:.1f}%"></div>
    <div class="avg" style="left:{base_rate * 100:.1f}%"></div>
    <div class="mark" style="left:calc({min(p, 0.995) * 100:.1f}% - 1px)"></div>
  </div>
  <div class="scale-labels">
    <span class="hide-sm" style="left:0">0%</span>
    <span style="left:{base_rate * 100:.1f}%;transform:translateX(-50%)">Average {pct(base_rate)}</span>
    <span style="left:{threshold * 100:.1f}%">Pop-up line {pct(threshold)}</span>
    <span class="hide-sm" style="right:0">100%</span>
  </div>
  <div class="action">{esc(action)} That is {times_avg:.1f} times the average session.</div>
</div>
""")
        st.html('<div class="section-title" style="margin-top:22px">What moves this prediction</div>'
                '<div class="section-note">Change in probability when one input is reset to a typical session '
                '(median of each number, most common category). Blue raises the chance, grey lowers it.</div>')
        if contrib.empty:
            st.caption("No single input moves the probability by more than half a point.")
        else:
            chart = alt.Chart(contrib).mark_bar(cornerRadiusEnd=3, height=16).encode(
                x=alt.X("change:Q", title=None, axis=alt.Axis(format="+.0%", tickCount=5)),
                y=alt.Y("input:N", sort=None, title=None, axis=alt.Axis(labelLimit=200, labelPadding=8)),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises chance", "Lowers chance"],
                                                               range=[ACCENT, MUTED]), legend=None),
                tooltip=[alt.Tooltip("input:N", title="Input"), alt.Tooltip("change:Q", format="+.1%", title="Change")],
            ).properties(height=34 * len(contrib) + 36).configure_view(stroke=None).configure_axis(
                grid=False, domainColor="#e2e5ea", labelColor="#4a5361", labelFontSize=12.5, tickColor="#e2e5ea",
                labelFont="Inter")
            st.altair_chart(chart, width="stretch")

# ---------------- Test sessions ----------------
with tab_tests:
    st.html(f'<div class="section-title">{len(tests):,} held-out sessions, ranked by the model</div>'
            '<div class="section-note">The model never saw these sessions while training. Pick a pop-up budget and '
            'see who would get one, and who really bought.</div>')
    c1, c2, c3 = st.columns([1.2, 1, 1])
    budget = c1.slider("Pop-up budget, share of sessions", 5, 30, 10, step=1, format="%d%%")
    month_pick = c2.selectbox("Month", ["All months"] + config.MONTHS,
                              format_func=lambda m: m if m == "All months" else MONTH_NAMES[m])
    visitor_pick = c3.selectbox("Visitor type", ["All visitors"] + config.VISITOR_TYPES,
                                format_func=lambda v: v if v == "All visitors" else VISITOR_PLAIN[v])

    shown = tests
    if month_pick != "All months":
        shown = shown[shown["Month"] == month_pick]
    if visitor_pick != "All visitors":
        shown = shown[shown["VisitorType"] == visitor_pick]
    shown = shown.sort_values("p_buy", ascending=False).reset_index(drop=True)
    n_popups = int(round(budget / 100 * len(shown)))
    flagged = shown.head(n_popups)
    buyers_total = int(shown[config.TARGET].sum())
    hits = int(flagged[config.TARGET].sum())
    precision = hits / n_popups if n_popups else 0
    recall = hits / buyers_total if buyers_total else 0
    random_hits = n_popups * (buyers_total / len(shown)) if len(shown) else 0

    st.html(f"""
<div class="stats">
  <div class="stat"><div class="k">Pop-ups shown</div><div class="v">{n_popups:,}</div>
    <div class="s">{budget}% of {len(shown):,} sessions</div></div>
  <div class="stat"><div class="k">Went to buyers</div><div class="v accent">{hits:,}</div>
    <div class="s">{pct(precision)} of pop-ups</div></div>
  <div class="stat"><div class="k">At random</div><div class="v">{random_hits:,.0f}</div>
    <div class="s">Same budget, no model</div></div>
  <div class="stat"><div class="k">Buyers reached</div><div class="v">{pct(recall)}</div>
    <div class="s">of {buyers_total:,} who bought</div></div>
</div>
""")

    page_size = 15
    rows_to_show = st.session_state.get("show_rows", page_size)
    top = shown.head(rows_to_show)
    rows, classes = [], []
    for i, r in top.iterrows():
        rows.append([
            f'<span class="dim">{i + 1}</span>',
            esc(MONTH_NAMES[r["Month"]][:3]),
            esc(VISITOR_PLAIN[r["VisitorType"]]),
            f'{int(r["ProductRelated"])}',
            f'{r["ProductRelated_Duration"] / 60:.0f}',
            f'{r["ExitRates"] * 100:.1f}%',
            prob_cell(r["p_buy"], threshold),
            '<span class="yes">Pop-up</span>' if i < n_popups else '<span class="no">None</span>',
            '<span class="yes">Bought</span>' if r[config.TARGET] == 1 else '<span class="no">Left</span>',
        ])
        classes.append("cut" if i == n_popups - 1 and i < len(top) - 1 else "")
    st.html(table_html([("#", "num"), ("Month", ""), ("Visitor", "hide-sm"), ("Product pages", "num"),
                        ("Minutes", "num hide-sm"), ("Exit rate", "num hide-sm"), ("Probability", "num"),
                        ("Decision", "hide-sm"), ("Outcome", "")], rows, classes))
    if rows_to_show < len(shown):
        if st.button(f"Show {min(page_size, len(shown) - rows_to_show)} more"):
            st.session_state["show_rows"] = rows_to_show + page_size
            st.rerun()
    st.caption(f"Ranked by purchase probability. With the whole test set, the fixed pop-up line ({pct(threshold)}) "
               f"flags {pct(test_m['flag_rate_at_threshold'], 1)} of sessions, close to the planned 10%. In November "
               "it would flag far more, so the line has to be re-set each month (see the monitoring runbook).")

# ---------------- How the model works ----------------
with tab_how:
    st.html('<div class="section-title">Baselines against the models</div>'
            '<div class="section-note">CV is 5 stratified folds on the 9,764 training sessions. Test is the 2,441 '
            'held-out sessions, opened once. Precision and recall are at a 10% pop-up budget.</div>')
    final_run = results[results["kind"] == "final"].index[0]
    groups = [
        ("Without page value (deployable)", [
            ("baseline_constant_nobody_buys", "Constant: nobody buys"),
            ("baseline_rule_B_products_and_low_exit", "Rule B: 20+ product pages, exit rate under 2.5%"),
            ("baseline_logreg_4_features", "Logistic regression, 4 columns"),
            ("tuned_logistic_regression_no_pagevalues", "Logistic regression, tuned"),
            ("tuned_random_forest_no_pagevalues", "Random forest, tuned"),
            (final_run, "Gradient boosting, tuned (final model)"),
        ]),
        ("With page value (not deployable)", [
            ("baseline_rule_A_pagevalue_above_0", "Rule A: page value above zero"),
            (with_pv.name, "Random forest, tuned, with page value"),
        ]),
    ]

    def fmt(value, style):
        if pd.isna(value):
            return '<span class="dim">CV only</span>'
        return {"auc": f"{value:.3f}", "pct": pct(value)}[style]

    rows, classes = [], []
    for group_name, runs in groups:
        rows.append(("group", group_name))
        classes.append("")
        for run, label in runs:
            r = results.loc[run]
            spread = "" if pd.isna(r["cv_pr_auc_std"]) or r["cv_pr_auc_std"] == 0 else \
                f' <span class="dim">&plusmn; {r["cv_pr_auc_std"]:.3f}</span>'
            first = [f'<span class="name">{esc(label)}</span>', fmt(r["cv_pr_auc_mean"], "auc") + spread]
            if pd.isna(r["test_pr_auc"]):
                rows.append(first + [("span", "Tuning run, CV only. The test set was opened once, for the final model.", 4)])
            else:
                rows.append(first + [fmt(r["test_pr_auc"], "auc"), fmt(r["test_roc_auc"], "auc"),
                                     fmt(r["test_precision_at_10pct"], "pct"), fmt(r["test_recall_at_10pct"], "pct")])
            classes.append("hl" if run == final_run else "")
    st.html(table_html([("Approach", ""), ("CV PR-AUC", "num hide-sm"), ("Test PR-AUC", "num"),
                        ("Test ROC-AUC", "num hide-sm"), ("Precision", "num"), ("Recall", "num hide-sm")], rows, classes))
    st.caption(f"Accuracy is not shown on purpose. \"Nobody buys\" is {pct(results.loc['baseline_constant_nobody_buys', 'test_accuracy_at_0_5'])} "
               "accurate and finds no one.")

    st.write("")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.html('<div class="section-title">What the model relies on</div>'
                '<div class="section-note">Drop in test PR-AUC when one input is shuffled. Exit rate, time on product '
                'pages and the month do most of the work. The six inputs not shown move it by less than 0.001.</div>')
        importance = importance_table().sort_values("importance_mean", ascending=False).head(10)
        importance["input"] = importance["feature"].map(INPUT_LABELS)
        importance["shown"] = importance["importance_mean"].clip(lower=0)
        chart = alt.Chart(importance).mark_bar(color=ACCENT, cornerRadiusEnd=3, height=13).encode(
            x=alt.X("shown:Q", title=None, axis=alt.Axis(format=".2f", tickCount=4)),
            y=alt.Y("input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelPadding=8)),
            tooltip=[alt.Tooltip("input:N", title="Input"),
                     alt.Tooltip("importance_mean:Q", format=".3f", title="Mean drop"),
                     alt.Tooltip("importance_std:Q", format=".3f", title="Std")],
        ).properties(height=320).configure_view(stroke=None).configure_axis(
            grid=False, domainColor="#e2e5ea", labelColor="#4a5361", labelFontSize=12.5, tickColor="#e2e5ea",
            labelFont="Inter")
        st.altair_chart(chart, width="stretch")
    with right:
        leak = pd.read_csv(config.REPORTS_DIR / "leakage_check_pagevalues.csv", index_col=0)["value"]
        slices = pd.read_csv(config.REPORTS_DIR / "slices.csv").set_index(["slice", "group"])
        nov_flag = slices.loc[("Month", "Nov"), "flag_rate"]
        feb_oct_flag = slices.loc[("Season", "Feb-Oct"), "flag_rate"]
        st.html(f"""
<div class="section-title">Why page value is left out</div>
<div class="section-note">Google Analytics gives a page value when a purchase follows the page in the same session.
So a buying session's own purchase feeds the number. At exit intent that purchase has not happened yet.</div>
<div class="decision">
  <div class="row"><span>Buy rate when page value is above zero</span><b>{pct(leak['buy_rate_pagevalue_above_0'])}</b></div>
  <div class="row"><span>Buy rate when it is zero</span><b>{pct(leak['buy_rate_pagevalue_0'])}</b></div>
  <div class="row"><span>Buyers with a page value</span><b>{pct(leak['share_of_buyers_with_pagevalue'])}</b></div>
  <div class="row"><span>Non-buyers with a page value</span><b>{pct(leak['share_of_nonbuyers_with_pagevalue'])}</b></div>
  <div class="row"><span>CV PR-AUC without it, with it</span><b>{leak['best_cv_pr_auc_without_pagevalues']:.2f}, {leak['best_cv_pr_auc_with_pagevalues']:.2f}</b></div>
</div>
<div class="section-title" style="margin-top:20px">Limitations</div>
<ul class="facts">
  <li>Every counter covers the whole session, not just the part before exit intent, so all scores are a bit optimistic.</li>
  <li>The model predicts who buys, not who buys because of the pop-up. Only a live test with a hold-out group can measure that.</li>
  <li>One fixed pop-up line flags {pct(nov_flag)} of November sessions and {pct(feb_oct_flag)} of February to October ones. Re-set it monthly.</li>
  <li>One store, one year, no January or April sessions.</li>
</ul>
""")

    st.html(f"""<div class="foot">Data: Sakar, Polat, Katircioglu and Kastro (2019), Online Shoppers Purchasing Intention
Dataset, UCI Machine Learning Repository (id 468), CC BY 4.0, accessed 26 November 2026. Model {esc(meta['model_version'])}
({esc(meta['model_family'].replace('_', ' '))}), MLflow run {esc(meta['mlflow_run_id'][:8])}, pop-up line {threshold:.3f}.</div>""")
