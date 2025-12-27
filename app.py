import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS
from datetime import datetime
st.cache_data.clear()


# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor: Win/Loss & Goals")

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

# --- CORE ENGINE ---
@st.cache_data(ttl=3600)
def load_and_train(url):
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
        
        # Binary Targets for Goals
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(4, closed='left').mean().fillna(x.mean()))
        
        feats = ["FTHG_roll", "FTAG_roll"]
        m_res = RandomForestClassifier(n_estimators=100).fit(df[feats], df["target_res"])
        m_goal = RandomForestClassifier(n_estimators=100).fit(df[feats], df["target_o25"])
        m_gg = RandomForestClassifier(n_estimators=100).fit(df[feats], df["target_gg"])
        return df, m_res, m_goal, m_gg, feats
    except: return None, None, None, None, None

# --- UI INTERFACE ---
sel_league = st.sidebar.selectbox("Competition", list(league_urls.keys()))
data, rf_res, rf_goal, rf_gg, predictors = load_and_train(league_urls[sel_league])

if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
        h_abs = st.slider(f"Absences ({home})", 0, 5, 0)
    with c2:
        away = st.selectbox("🚩 Away Team", teams, index=1 if len(teams)>1 else 0)
        a_abs = st.slider(f"Absences ({away})", 0, 5, 0)

        # ... (Your Data Loading Code stays the same) ...

    if st.button("🚀 RUN FULL PREDICTION"):
        # 1. Fetch AI Predictions (Goals & Result)
        h_row = data[data["HomeTeam"] == home].iloc[-1]
        p_res = rf_res.predict_proba([[h_row[p] for p in predictors]])[0]
        p_o25 = rf_goal.predict_proba([[h_row[p] for p in predictors]])[0][1]
        p_gg = rf_gg.predict_proba([[h_row[p] for p in predictors]])[0][1]

        # 2. Fetch Live Intel (Lineups & News) - MOVING THIS INSIDE THE BUTTON
        with st.spinner("Fetching Live Lineups & News..."):
            h_news = get_intel(home)
            a_news = get_intel(away)
            h_lineup = get_intel(home, mode="lineup")
            a_lineup = get_intel(away, mode="lineup")

        # --- DISPLAY SECTION (Tablet Optimized) ---
        st.divider()
        
        # 🏆 Result
        st.subheader("🏆 Match Winner AI")
        c1, c2, c3 = st.columns(3)
        c1.metric(home, f"{p_res[2]*100:.1f}%")
        c2.metric("Draw", f"{p_res[1]*100:.1f}%")
        c3.metric(away, f"{p_res[0]*100:.1f}%")

        # ⚽ Goals
        st.subheader("⚽ Goal Predictions")
        st.success(f"**Over 2.5 Goals:** {p_o25*100:.1f}%")
        st.info(f"**Both Teams Score (GG):** {p_gg*100:.1f}%")

        # 🗞️ Live Intel (Now stays visible!)
        st.subheader("🗞️ Real-Time Intelligence")
        col_news1, col_news2 = st.columns(2)
        with col_news1:
            st.write(f"**{home} News:**")
            for n in h_news: st.caption(f"• {n['body'][:100]}...")
        with col_news2:
            st.write(f"**{away} News:**")
            for n in a_news: st.caption(f"• {n['body'][:100]}...")
                
