import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Match Master: Live Edition", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor + Live Pulse")

# --- DATA REPOSITORY (VERIFIED 25/26) ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
BASE_EXTRA = "https://www.football-data.co.uk/new_leagues/"

league_urls = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England: Premier League": f"{BASE_MAIN}E0.csv",
    "🇪🇸 Spain: La Liga": f"{BASE_MAIN}SP1.csv",
    "🇩🇪 Germany: Bundesliga": f"{BASE_MAIN}D1.csv",
    "🇮🇹 Italy: Serie A": f"{BASE_MAIN}I1.csv",
    "🇫🇷 France: Ligue 1": f"{BASE_MAIN}F1.csv",
    "🇳🇱 Netherlands: Eredivisie": f"{BASE_MAIN}N1.csv",
    "🇧🇪 Belgium: Pro League": f"{BASE_MAIN}B1.csv",
    "🇵🇹 Portugal: Liga Portugal": f"{BASE_MAIN}P1.csv",
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland: Premiership": f"{BASE_MAIN}SC0.csv",
    "🇦🇹 Austria: Bundesliga": f"{BASE_EXTRA}AUT.csv",
    "🇨🇭 Switzerland: Super League": f"{BASE_EXTRA}SWZ.csv",
    "🇬🇷 Greece: Super League": f"{BASE_EXTRA}GREECE.csv",
    "🇹🇷 Turkey: Süper Lig": f"{BASE_EXTRA}TUR.csv",
    "🇩🇰 Denmark: Superliga": f"{BASE_EXTRA}DNK.csv"
}

# --- REAL-TIME INTELLIGENCE ---
def get_live_data(query_type, home, away):
    """Fetches Live Scores, News, or Lineups via DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            if query_type == "score":
                q = f"live score {home} vs {away} today"
            elif query_type == "lineup":
                q = f"{home} vs {away} confirmed starting lineups"
            else:
                q = f"{home} {away} team news injuries"
            return list(ddgs.text(q, max_results=2))
    except: return []

# --- PREDICTION ENGINE ---
@st.cache_data(ttl=3600)
def load_and_train(url):
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(4, closed='left').mean().fillna(x.mean()))
        model = RandomForestClassifier(n_estimators=100).fit(df[["FTHG_roll", "FTAG_roll"]], df["target_res"])
        return df, model
    except: return None, None

# --- UI INTERFACE ---
league = st.sidebar.selectbox("Select Competition", list(league_urls.keys()))
data, model = load_and_train(league_urls[league])

if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    with c1:
        h_team = st.selectbox("🏠 Home", teams, index=0)
        h_abs = st.slider(f"Absences ({h_team})", 0, 5, 0)
    with c2:
        a_team = st.selectbox("🚩 Away", teams, index=1 if len(teams)>1 else 0)
        a_abs = st.slider(f"Absences ({a_team})", 0, 5, 0)

    # --- LIVE ACTION SECTION ---
    st.divider()
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        if st.button("📡 Check Live Score"):
            score = get_live_data("score", h_team, a_team)
            if score: st.info(f"**Current Pulse:** {score[0]['body'][:160]}...")
            else: st.warning("No live data found. Match may not have started.")
    with col_l2:
        if st.checkbox("🔍 See Lineups (1hr before)"):
            lineups = get_live_data("lineup", h_team, a_team)
            for l in lineups: st.caption(f"📋 {l['body'][:120]}...")

    # --- FINAL PREDICTION ---
    if st.button("🚀 PREDICT WITH HUMAN FACTORS"):
        h_row = data[data["HomeTeam"] == h_team].iloc[-1]
        probs = model.predict_proba([[h_row["FTHG_roll"], h_row["FTAG_roll"]]])[0]
        
        # Math adjustments (10% per absence)
        final_h = max(0, min(1, probs[2] - (h_abs * 0.10) + (a_abs * 0.10)))
        final_a = max(0, min(1, probs[0] - (a_abs * 0.10) + (h_abs * 0.10)))
        
        st.divider()
        st.subheader("Final AI Verdict")
        r1, r2, r3 = st.columns(3)
        r1.metric(h_team, f"{final_h*100:.1f}%")
        r2.metric(a_team, f"{final_a*100:.1f}%")
        r3.metric("Draw", f"{(1 - final_h - final_a)*100:.1f}%")
        
