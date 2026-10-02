"""
SalesSense AI - B2B sales lead-scoring assistant (Streamlit app).

Run locally:   streamlit run app.py
Gemini key:    put GEMINI_API_KEY in .streamlit/secrets.toml (never in this file)
"""

import html
import json
import os
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ai import DEFAULT_MODEL, AIUnavailable, LeadAnalysis, analyse_lead, certainty_flags
from scoring import (COMPANY_SIZES, DEFAULT_ACTIONS, INTERACTIONS, REQUIRED_COLUMNS,
                     SCORING_RULES, URGENCY_LEVELS, YES_NO, score_dataframe,
                     score_lead, validate_lead)

DATA_DIR = Path(__file__).parent / "data"
TIER_COLORS = {"HOT": "#d4462b", "WARM": "#d99a0b", "COLD": "#2a78d6"}
TIER_ICONS = {"HOT": "🔥", "WARM": "🌤️", "COLD": "❄️"}
FALLBACK_MSG = ("AI recommendation is currently unavailable. "
                "The rule-based lead assessment is still available.")

st.set_page_config(page_title="SalesSense AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.score-card {border-radius: 12px; padding: 18px 22px; color: white; margin-bottom: 8px;}
.score-card .num {font-size: 44px; font-weight: 700; line-height: 1.1;}
.score-card .tier {font-size: 20px; font-weight: 600; letter-spacing: 1px;}
.score-card .sub {font-size: 14px; opacity: 0.9;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Settings: API key and model come from Streamlit secrets or env vars
# ---------------------------------------------------------------
def get_setting(name, default=""):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file at all
        pass
    return os.environ.get(name, default)


API_KEY = get_setting("GEMINI_API_KEY")
MODEL = get_setting("GEMINI_MODEL", DEFAULT_MODEL)

if "ai_results" not in st.session_state:
    st.session_state.ai_results = {}   # lead key -> AI next action (for the export)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_analysis(lead_json, score, tier, breakdown_json, model):
    """Caches Gemini answers so Streamlit reruns don't spend extra API calls.
    Failures are not cached, so a retry really retries."""
    result = analyse_lead(json.loads(lead_json), score, tier,
                          json.loads(breakdown_json), API_KEY, model)
    return result.model_dump()


def lead_key(lead):
    return f"{lead.get('Lead_ID', '')}|{lead['Company']}"


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.title("📈 SalesSense AI")
    st.caption("AI-assisted B2B lead scoring and recommendations")
    st.divider()
    st.subheader("AI status")
    simulate_outage = st.toggle(
        "Simulate AI outage", value=False,
        help="Test mode: blocks all Gemini calls to show the app still works without AI.")
    if simulate_outage:
        st.warning("AI outage simulated. Rule-based scoring only.")
    elif API_KEY:
        st.success(f"Gemini connected\n\nModel: `{MODEL}`")
    else:
        st.info("No Gemini API key found. Running in rule-based mode.")
    st.divider()
    st.caption("Decision support only. Scores are based on business rules, not a "
               "prediction of conversion. The salesperson makes the final decision.")
    st.caption("🔒 Privacy: lead details you submit for AI analysis, including sales "
               "notes, are sent to Google's Gemini API. Nothing is saved by this app. "
               "Use synthetic or consented data only.")


# ---------------------------------------------------------------
# Shared result display
# ---------------------------------------------------------------
def show_score_card(lead, score, tier):
    st.markdown(
        f"""<div class="score-card" style="background:{TIER_COLORS[tier]}">
        <div class="sub">{html.escape(lead['Company'])} · {html.escape(lead['Industry'])}</div>
        <div class="num">{score}/100</div>
        <div class="tier">{TIER_ICONS[tier]} {tier} LEAD</div></div>""",
        unsafe_allow_html=True)


def show_breakdown(breakdown):
    st.markdown("**Why this score** (rule-based, fully traceable)")
    table = pd.DataFrame(breakdown)
    table["Points"] = table.apply(lambda r: f"{r['Points']} / {r['Max']}", axis=1)
    st.dataframe(table[["Factor", "Points", "Reason"]], hide_index=True,
                 width="stretch")


def show_ai_analysis(lead, score, tier, breakdown):
    st.markdown("#### 🤖 AI analysis (Gemini)")
    if simulate_outage or not API_KEY:
        st.warning(FALLBACK_MSG)
        st.markdown(f"**Default next step for a {tier} lead:** {DEFAULT_ACTIONS[tier]}")
        return

    try:
        with st.spinner("Asking Gemini..."):
            result = cached_analysis(json.dumps(lead, sort_keys=True), score, tier,
                                     json.dumps(breakdown), MODEL)
    except AIUnavailable as exc:
        st.warning(f"{FALLBACK_MSG}\n\nReason: {exc}")
        st.markdown(f"**Default next step for a {tier} lead:** {DEFAULT_ACTIONS[tier]}")
        return

    st.session_state.ai_results[lead_key(lead)] = result["next_action"]

    if result["notes_conflict"]:
        st.error("⚠️ The sales notes conflict with the score. "
                 "Review the risks below before acting on this lead.")

    st.write(result["summary"])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Key factors**")
        for factor in result["key_factors"]:
            st.markdown(f"- {factor}")
        st.markdown("**Recommended next action**")
        st.info(result["next_action"])
    with col2:
        st.markdown("**Suggested sales approach**")
        st.write(result["sales_approach"])
        st.markdown("**Risks and information gaps**")
        for risk in result["risks_and_gaps"]:
            st.markdown(f"- {risk}")

    if certainty_flags(LeadAnalysis(**result)):
        st.warning("Guardrail: this answer contains over-confident wording. "
                   "No lead is certain to convert.")
    st.caption("AI-generated explanation. It can be wrong. The score above was not "
               "changed by AI, and the final decision rests with the salesperson.")


# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
st.title("SalesSense AI")
st.markdown("Score B2B leads with transparent rules, then get an AI-generated "
            "explanation and next step.")

tab_single, tab_batch, tab_how = st.tabs(
    ["🧾 Score a lead", "📊 Upload leads & dashboard", "ℹ️ How scoring works"])

# ----- Tab 1: single lead -----
with tab_single:
    with st.form("lead_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            company = st.text_input("Company name *")
            industry = st.text_input("Industry", placeholder="e.g. IT Services")
            size = st.selectbox("Company size (employees)", COMPANY_SIZES, index=2)
            vendor = st.text_input("Current vendor / solution",
                                   placeholder="e.g. Competitor A, In-house tool, None")
        with c2:
            decision_maker = st.radio("Decision maker involved?", YES_NO, index=1, horizontal=True)
            demo = st.radio("Demo requested?", YES_NO, index=1, horizontal=True)
            urgency = st.selectbox("Purchase urgency", URGENCY_LEVELS, index=1)
            interaction = st.selectbox("Previous interaction", INTERACTIONS, index=1)
        with c3:
            # min_value is not set on purpose, so the validation step can catch negatives
            visits = st.number_input("Website visits (last 30 days)", value=0, step=1)
            opens = st.number_input("Email opens (last 30 days)", value=0, step=1)
            notes = st.text_area(
                "Sales notes (optional)", height=120,
                placeholder="Anything from calls or emails. Not used in the score; "
                            "the AI reads it to spot risks.")
        submitted = st.form_submit_button("Analyze lead", type="primary")

    if submitted:
        lead, errors = validate_lead({
            "Company": company, "Industry": industry, "Company_Size": size,
            "Decision_Maker": decision_maker, "Demo_Requested": demo,
            "Purchase_Urgency": urgency, "Website_Visits": visits,
            "Email_Opens": opens, "Current_Vendor": vendor,
            "Previous_Interaction": interaction, "Sales_Notes": notes})
        if errors:
            st.session_state.pop("single", None)
            for err in errors:
                st.error(err)
        else:
            st.session_state.single = lead

    if "single" in st.session_state:
        lead = st.session_state.single
        score, tier, breakdown = score_lead(lead)
        left, right = st.columns([1, 2])
        with left:
            show_score_card(lead, score, tier)
            st.progress(score / 100)
        with right:
            show_breakdown(breakdown)
        st.divider()
        show_ai_analysis(lead, score, tier, breakdown)

# ----- Tab 2: upload + dashboard + export -----
with tab_batch:
    up_col, sample_col = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader("Upload leads (CSV or Excel)", type=["csv", "xlsx"])
    with sample_col:
        st.write("")
        if st.button("Load sample data (30 leads)"):
            st.session_state.batch_source = "sample_leads.csv"
        if st.button("Load sample with errors"):
            st.session_state.batch_source = "sample_leads_with_errors.csv"
        template = pd.DataFrame(columns=["Lead_ID"] + REQUIRED_COLUMNS + ["Sales_Notes"])
        st.download_button("Download blank template", template.to_csv(index=False),
                           "salessense_template.csv", "text/csv")

    raw_df = None
    try:
        if uploaded is not None:
            if uploaded.name.lower().endswith(".xlsx"):
                raw_df = pd.read_excel(uploaded)
            else:
                raw_df = pd.read_csv(uploaded)
        elif "batch_source" in st.session_state:
            raw_df = pd.read_csv(DATA_DIR / st.session_state.batch_source)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")

    if raw_df is None:
        st.info("Upload a file or load the sample data to see the dashboard.")
    else:
        try:
            scored, skipped = score_dataframe(raw_df)
        except ValueError as exc:
            st.error(str(exc))
            st.caption("Download the blank template above to see the expected columns.")
            scored, skipped = pd.DataFrame(), []

        if skipped:
            st.warning(f"{len(skipped)} row(s) failed validation and were not scored:")
            st.dataframe(pd.DataFrame(skipped), hide_index=True, width="stretch")

        if not scored.empty:
            counts = scored["Tier"].value_counts()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total leads scored", len(scored))
            m2.metric("🔥 Hot", int(counts.get("HOT", 0)))
            m3.metric("🌤️ Warm", int(counts.get("WARM", 0)))
            m4.metric("❄️ Cold", int(counts.get("COLD", 0)))
            m5.metric("Average score", f"{scored['Score'].mean():.0f}")

            chart_col, top_col = st.columns([1, 2])
            with chart_col:
                st.markdown("**Lead distribution by tier**")
                dist = pd.DataFrame({"Tier": ["HOT", "WARM", "COLD"],
                                     "Leads": [int(counts.get(t, 0)) for t in ["HOT", "WARM", "COLD"]]})
                bars = alt.Chart(dist).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Tier:N", sort=["HOT", "WARM", "COLD"], title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Leads:Q", title="Number of leads"),
                    color=alt.Color("Tier:N", legend=None, scale=alt.Scale(
                        domain=list(TIER_COLORS), range=list(TIER_COLORS.values()))),
                    tooltip=["Tier", "Leads"])
                labels = bars.mark_text(dy=-8).encode(text="Leads:Q", color=alt.value("#52514e"))
                st.altair_chart((bars + labels).properties(height=260), width="stretch")
            with top_col:
                st.markdown("**Top 5 priority leads**")
                st.dataframe(scored.head(5)[["Company", "Industry", "Score", "Tier", "Reasons"]],
                             hide_index=True, width="stretch")

            st.markdown("**All scored leads**")
            tiers = st.multiselect("Filter by tier", ["HOT", "WARM", "COLD"],
                                   default=["HOT", "WARM", "COLD"])
            view = scored[scored["Tier"].isin(tiers)]
            st.dataframe(view[["Lead_ID", "Company", "Industry", "Score", "Tier", "Reasons",
                               "Recommended_Action"]],
                         hide_index=True, width="stretch")

            st.divider()
            st.markdown("#### 🤖 AI analysis for one lead")
            st.caption("AI runs one lead at a time, on request, to stay within "
                       "free-tier API limits.")
            options = {f"{r['Company']} ({r['Score']}, {r['Tier']})": i
                       for i, r in scored.iterrows()}
            choice = st.selectbox("Choose a lead", list(options))
            if st.button("Generate AI analysis"):
                st.session_state.batch_pick = choice
            if st.session_state.get("batch_pick") in options:
                row = scored.loc[options[st.session_state.batch_pick]]
                lead = {k: (int(v) if k in ("Website_Visits", "Email_Opens") else v)
                        for k, v in row.items()
                        if k not in ("Score", "Tier", "Reasons", "Recommended_Action")}
                score, tier, breakdown = score_lead(lead)
                show_score_card(lead, score, tier)
                show_breakdown(breakdown)
                show_ai_analysis(lead, score, tier, breakdown)

            st.divider()
            export = scored.copy()
            export["AI_Next_Action"] = [st.session_state.ai_results.get(lead_key(r), "")
                                        for r in export.to_dict("records")]
            export["Assessed_On"] = date.today().isoformat()
            export = export[["Lead_ID", "Company", "Industry", "Score", "Tier", "Reasons",
                             "Recommended_Action", "AI_Next_Action", "Assessed_On"]]
            st.download_button("⬇️ Download CRM export (CSV)", export.to_csv(index=False),
                               f"salessense_export_{date.today().isoformat()}.csv",
                               "text/csv", type="primary")

# ----- Tab 3: how it works -----
with tab_how:
    st.markdown("""
**How a lead moves through the app**

Lead entry or upload → input validation → rule-based score (0-100) → Hot / Warm / Cold →
Gemini explanation and next step → dashboard and CRM export
""")
    st.markdown("**Scoring rules (total 100 points)**")
    st.dataframe(pd.DataFrame(SCORING_RULES, columns=["Factor", "Max points", "How points are given"]),
                 hide_index=True, width="stretch")
    st.markdown("**Tiers:** HOT = 80-100 · WARM = 50-79 · COLD = 0-49")
    st.markdown("""
**What the AI does:** explains the score, lists key factors, suggests a next action and
sales approach, and reads the free-text sales notes to flag risks the score cannot see.

**What the AI does not do:** it never changes the score, never predicts that a lead will
convert, never contacts anyone, and never makes pricing, legal or contract decisions.

**Known limits:** the rules and weights are assumptions that should be validated against
historical win/loss data. High engagement does not always mean buying intent. The AI only
knows what is entered and can still be wrong.
""")
