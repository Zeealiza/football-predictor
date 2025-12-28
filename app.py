import streamlit as st
import pandas as pd
import numpy as np
import joblib
from duckduckgo_search import DDGS
from datetime import datetime

# ================== APP CONFIG ==================
st.set_page_config(
    page_title="AI Match Master Pro",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ AI Match Master Pro")
st.caption("Professional Football Prediction Engine")

# ================== CACHED LOADERS ==================

@st.cache_resource
def load_league_model(league_key):
    return joblib.load(f"models/{league_key}.pkl")

@st.cache_data(ttl=1800)
def get_intel(team):
    try:
        with DDGS() as ddgs:
            res = list(ddgs.text(f"{team} injuries suspension", max_results=2))
            return res if res else [{"body": "No major updates"}]
    except:
        return [{"body": "Intel unavailable"}]

# ================== FEATURE ENGINEERING ==================

def add_table_position(df):
    df["points"] = np.select(
        [df.FTR == "H", df.FTR == "D", df.FTR == "A"],
        [3, 1, 0]
    )
    table = (
        df.groupby("HomeTeam")["points"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    table["position"] = range(1, len(table) + 1)
    return df.merge(table, on="HomeTeam", how="left")

def add_form(df, window=5):
    df["form"] = (
        df.groupby("HomeTeam")["points"]
        .transform(lambda x: x.rolling(window).mean())
        .fillna(df["points"].mean())
    )
    return df

def rest_days(df):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["rest_days"] = (
        df.groupby("HomeTeam")["Date"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(2, 14)
    )
    return df

# ================== ACCUMULATOR ENGINE ==================

def generate_accumulator(matches, min_prob=0.65):
    acc = []
    for m in matches:
        if m["confidence"] >= min_prob:
            acc.append(m)
    return acc

# ================== LEAGUE MAP ==================

LEAGUES = {
    "Premier League": "EPL",
    "La Liga": "LaLiga",
    "Bundesliga": "Bundesliga",
    "Serie A": "SerieA",
    "Ligue 1": "Ligue1"
}

league_name = st.sidebar.selectbox("Competition", list(LEAGUES.keys()))
league_key = LEAGUES[league_name]

df, models, FEATURES = load_league_model(league_key)
rf_res, rf_o25, rf_gg = models.values()

teams = sorted(df.HomeTeam.unique())

# ================== UI ==================

c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("🏠 Home Team", teams)
    h_abs = st.slider("Home Absences", 0, 5, 0)

with c2:
    away = st.selectbox("🚩 Away Team", teams)
    a_abs = st.slider("Away Absences", 0, 5, 0)

use_intel = st.sidebar.checkbox("Use Live Team News", False)

# ================== PREDICTION ==================

if st.button("🚀 RUN AI PREDICTION"):
    h = df[df.HomeTeam == home].iloc[-1]
    a = df[df.HomeTeam == away].iloc[-1]

    x = np.array([h[f] for f in FEATURES]).reshape(1, -1)

    p_res = rf_res.predict_proba(x)[0]
    p_o25 = rf_o25.predict_proba(x)[0][1]
    p_gg = rf_gg.predict_proba(x)[0][1]

    # Adjustments
    pos_factor = (a.position - h.position) * 0.015
    fatigue_factor = (h.rest_days - a.rest_days) * 0.02
    abs_factor = (a_abs - h_abs) * 0.08

    home_win = np.clip(p_res[2] + pos_factor + fatigue_factor + abs_factor, 0, 1)
    away_win = np.clip(p_res[0] - pos_factor - fatigue_factor - abs_factor, 0, 1)
    draw = max(0, 1 - home_win - away_win)

    # ================== DISPLAY ==================

    st.divider()
    st.subheader("🏆 Match Outcome")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} Win", f"{home_win*100:.1f}%")
    c2.metric("Draw", f"{draw*100:.1f}%")
    c3.metric(f"{away} Win", f"{away_win*100:.1f}%")

    st.subheader("⚽ Goals Market")
    g1, g2 = st.columns(2)
    g1.success(f"Over 2.5 Goals: {p_o25*100:.1f}%")
    g2.info(f"BTTS: {p_gg*100:.1f}%")

    # ================== FINAL VERDICT ==================

    probs = {
        home: home_win,
        "Draw": draw,
        away: away_win
    }

    outcome = max(probs, key=probs.get)
    confidence = probs[outcome]

    st.divider()
    st.header("🧠 AI FINAL VERDICT")
    st.warning(f"**{outcome}** — Confidence: **{confidence*100:.1f}%**")

    if p_o25 > 0.65:
        st.markdown("📈 Expect goals in this match.")
    else:
        st.markdown("📉 Likely a tight encounter.")

    # ================== INTEL ==================

    if use_intel:
        st.subheader("🗞️ Team News")
        n1, n2 = st.columns(2)
        with n1:
            st.write(f"**{home}**")
            for n in get_intel(home):
                st.caption(f"• {n['body'][:120]}...")
        with n2:
            st.write(f"**{away}**")
            for n in get_intel(away):
                st.caption(f"• {n['body'][:120]}...")

    # ================== ACCUMULATOR ==================

    st.divider()
    st.subheader("📊 Accumulator Candidate")

    acc = generate_accumulator([
        {"match": f"{home} vs {away}", "pick": outcome, "confidence": confidence}
    ])

    if acc:
        for a in acc:
            st.success(f"{a['match']} → {a['pick']} ({a['confidence']*100:.1f}%)")
    else:
        st.info("No high-confidence accumulator picks")
