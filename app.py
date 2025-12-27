import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor + Live Pulse")

# --- DATA REPOSITORY ---
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

# --- INTELLIGENCE TOOLS ---
def get_live_intel(h, a, mode="news"):
    try:
        with DDGS() as ddgs:
            if mode == "score": q = f"live score {h} vs {a}"
            elif mode == "lineup": q = f"{h} vs {a} confirmed starting lineups"
            else: q = f"{h} {a} team news injuries"
            return list(ddgs.text(q, max_results=2))
    except: return []

def get_h2h_swing(df, h, a):
    h2h = df[((df['HomeTeam']==h) & (df['AwayTeam']==a)) | ((df['HomeTeam']==a) & (df['AwayTeam']==h))].tail(3)
    if len(h2h) < 1: return 0
    hw = len(h2h[((h2h['HomeTeam']==h) & (h2h['FTR']=='H')) | ((h2h['AwayTeam']==h) & (h2h['FTR']=='A'))])
    return 0.10 if hw >= 2 else (-0.10 if (len(h2h)-hw) >= 2 else 0)

# --- CORE ENGINE ---
@st.cache_data(ttl=3600)
def load_and_train(url):
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(4, closed='left').mean().fillna(x.mean()))
        
        feats = ["FTHG_roll", "FTAG_roll"]
        model = RandomForestClassifier(n_estimators=100)
        model.fit(df[feats], df["target_res"])
        return df, model, feats
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return None, None, None

# --- UI INTERFACE ---
sel_league = st.sidebar.selectbox("Select Competition", list(league_urls.keys()))
data, rf_model, predictors = load_and_train(league_urls[sel_league])

if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
        h_abs = st.slider(f"Key Absences ({home})", 0, 5, 0)
    with c2:
        away = st.selectbox("🚩 Away Team", teams, index=1 if len(teams)>1 else 0)
        a_abs = st.slider(f"Key Absences ({away})", 0, 5, 0)

    st.divider()
    lc1, lc2 = st.columns(2)
    with lc1:
        if st.button("📡 Check Live Score"):
            pulse = get_live_intel(home, away, "score")
            if pulse: st.info(pulse[0]['body'][:160])
    with lc2:
        if st.checkbox("🔍 See Lineups (1hr before)"):
            lines = get_live_intel(home, away, "lineup")
            for l in lines: st.caption(f"📋 {l['body'][:120]}...")

    if st.button("🚀 PREDICT NOW"):
        h_row = data[data["HomeTeam"] == home].iloc[-1]
        probs = rf_model.predict_proba([[h_row[p] for p in predictors]])[0]
        
        # Apply 10% Absence Penalty & H2H Swing
        swing = get_h2h_swing(data, home, away)
        final_h = max(0, min(1, probs[2] - (h_abs * 0.10) + (a_abs * 0.10) + swing))
        final_a = max(0, min(1, probs[0] - (a_abs * 0.10) + (h_abs * 0.10) - swing))
        final_d = max(0, 1 - final_h - final_a)

        st.divider()
        res1, res2, res3 = st.columns(3)
        res1.metric(f"{home} Win", f"{final_h*100:.1f}%")
        res2.metric(f"{away} Win", f"{final_a*100:.1f}%")
        res3.metric("Draw", f"{final_d*100:.1f}%")
        
