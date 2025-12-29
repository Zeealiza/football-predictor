import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --- CONFIG & RAM MANAGEMENT ---
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
Path("models").mkdir(exist_ok=True)

def clear_ram():
    gc.collect()

# --- 1. REFINED DATA REPOSITORY (Top Divisions Only) ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
BASE_EXTRA = "https://www.football-data.co.uk/new_leagues/"

LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": f"{BASE_MAIN}E0.csv",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 League 1": f"{BASE_MAIN}E2.csv",
    "🇪🇸 La Liga": f"{BASE_MAIN}SP1.csv",
    "🇩🇪 Bundesliga": f"{BASE_MAIN}D1.csv",
    "🇮🇹 Serie A": f"{BASE_MAIN}I1.csv",
    "🇫🇷 Ligue 1": f"{BASE_MAIN}F1.csv",
    "🇳🇱 Eredivisie": f"{BASE_MAIN}N1.csv",
    "🇧🇪 Pro League": f"{BASE_MAIN}B1.csv",
    "🇵🇹 Liga Portugal": f"{BASE_MAIN}P1.csv",
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Premiership": f"{BASE_MAIN}SC0.csv",
    "🇦🇹 Austria": f"{BASE_EXTRA}AUT.csv",
    "🇨🇭 Switzerland": f"{BASE_EXTRA}SWZ.csv",
    "🇬🇷 Greece": f"{BASE_EXTRA}GREECE.csv",
    "🇹🇷 Turkey": f"{BASE_EXTRA}TUR.csv",
    "🇩🇰 Denmark": f"{BASE_EXTRA}DNK.csv"
}

# --- 2. OPTIMIZED AI ENGINE ---
def get_league_code(url):
    return url.split("/")[-1].replace(".csv", "")

def train_and_save(url):
    code = get_league_code(url)
    res_path, goal_path = f"models/{code}_res.pkl", f"models/{code}_goal.pkl"
    
    if not os.path.exists(res_path):
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
            
            # Feature: Last 5 Games Performance
            for s in ["FTHG", "FTAG"]:
                df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(
                    lambda x: x.rolling(5, closed='left').mean().fillna(x.mean())
                )
            
            y_res = df["FTR"].map({"H": 2, "D": 1, "A": 0})
            y_goal = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
            X = df[["FTHG_roll", "FTAG_roll"]]

            # Model params tuned for 1GB RAM stability
            m_res = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_res)
            m_goal = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_goal)

            with open(res_path, "wb") as f: pickle.dump(m_res, f)
            with open(goal_path, "wb") as f: pickle.dump(m_goal, f)
            del df, X, y_res, y_goal
            clear_ram()
        except Exception:
            pass

@st.cache_resource
def load_tools(url):
    train_and_save(url)
    code = get_league_code(url)
    with open(f"models/{code}_res.pkl", "rb") as f: res = pickle.load(f)
    with open(f"models/{code}_goal.pkl", "rb") as f: goal = pickle.load(f)
    return res, goal

@st.cache_data(ttl=1800)
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    for s in ["FTHG", "FTAG"]:
        df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(
            lambda x: x.rolling(5, closed='left').mean().fillna(x.mean())
        )
    return df

# --- 3. UI ---
sel_league = st.sidebar.selectbox("Select Competition", list(LEAGUES.keys()))
target_url = LEAGUES[sel_league]

data = load_data(target_url)
rf_res, rf_goal = load_tools(target_url)

teams = sorted(data["HomeTeam"].unique())
c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("🏠 Home Team", teams)
    h_odd = st.number_input(f"SportyBet {home} Odd", 1.01, 50.0, 2.0)
with c2:
    away = st.selectbox("🚩 Away Team", teams, index=1)
    a_odd = st.number_input(f"SportyBet {away} Odd", 1.01, 50.0, 2.0)

if st.button("🚀 RUN AI ANALYSIS"):
    h_row = data[data["HomeTeam"] == home].iloc[-1]
    a_row = data[data["HomeTeam"] == away].iloc[-1]
    
    X_input = [[h_row["FTHG_roll"], h_row["FTAG_roll"]]]
    p_res = rf_res.predict_proba(X_input)[0]
    p_goal = rf_goal.predict_proba(X_input)[0][1]

    # QUALITY NEUTRALIZER (Corrected Form Bias)
    # If Away quality is elite (>1.8 avg goals), reduce Home Win weight
    ha_bias = 0.12 if a_row["FTHG_roll"] < 1.8 else 0.02
    f_h = max(0, p_res[2] - ha_bias)
    f_a = min(1, p_res[0] + (ha_bias if a_row["FTHG_roll"] > 1.8 else 0))

    st.divider()
    st.subheader("📊 Match Probability Forecast")
    colA, colB, colC = st.columns(3)
    colA.metric(f"{home} Win", f"{f_h*100:.1f}%")
    colB.metric("Draw", f"{(1-f_h-f_a)*100:.1f}%")
    colC.metric(f"{away} Win", f"{f_a*100:.1f}%")

    st.subheader("⚽ Goal Prediction")
    st.success(f"**Over 2.5 Goals Probability:** {p_goal*100:.1f}%")
    
    # CONFLICT CHECK (Vs SportyBet Market)
    if (f_a > 0.6 and h_odd < 1.7):
        st.error(f"⚠️ **MARKET TRAP DETECTED:** AI performance metrics strongly favor {away}, but market odds are skewed towards {home}. High value potential on Away/Draw.")
    
    clear_ram()
    
