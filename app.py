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

    if st.button("🚀 RUN FULL PREDICTION"):
        # 1. Get the data for the last match to use as AI input
        h_row = data[data["HomeTeam"] == home].iloc[-1]
        input_feats = [[h_row[p] for p in predictors]]
        
        # 2. Run all AI models
        p_res = rf_res.predict_proba(input_feats)[0]
        p_o25 = rf_goal.predict_proba(input_feats)[0][1]
        p_gg = rf_gg.predict_proba(input_feats)[0][1]

        # 3. Apply the 10% Absence Adjustment
        f_h = max(0, min(1, p_res[2] - (h_abs * 0.10) + (a_abs * 0.10)))
        f_a = max(0, min(1, p_res[0] - (a_abs * 0.10) + (h_abs * 0.10)))
        f_d = max(0, 1 - f_h - f_a)

        st.divider()
        
        # --- SECTION 1: WIN/DRAW/LOSS ---
        st.header("🏆 Match Winner AI")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} Win", f"{f_h*100:.1f}%")
        c2.metric("Draw", f"{f_d*100:.1f}%")
        c3.metric(f"{away} Win", f"{f_a*100:.1f}%")

        # --- SECTION 2: GOALS (FORCED VISIBILITY) ---
        st.write("---")
        st.header("⚽ Goals & BTTS Analysis")
        
        # We use a colorful box here to make sure it shows up
        st.success(f"**Over 2.5 Goals Probability:** {p_o25*100:.1f}%")
        st.info(f"**Both Teams to Score (GG) Probability:** {p_gg*100:.1f}%")
        
        # Additional backup display
        st.write(f"### 🔥 AI Over 2.5 Prediction: {p_o25*100:.1f}%")
        st.write(f"### 🎯 AI BTTS Prediction: {p_gg*100:.1f}%")
