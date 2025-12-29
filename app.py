import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --- RAM-SAFE CONFIG ---
st.set_page_config(page_title="AI Match Master Ultra", page_icon="🔮", layout="wide")
Path("models").mkdir(exist_ok=True)

def clear_ram():
    gc.collect()

# --- 1. DATA REPOSITORY ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
BASE_EXTRA = "https://www.football-data.co.uk/new_leagues/"

LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": f"{BASE_MAIN}E0.csv",
    "🇪🇸 La Liga": f"{BASE_MAIN}SP1.csv",
    "🇩🇪 Bundesliga": f"{BASE_MAIN}D1.csv",
    "🇮🇹 Serie A": f"{BASE_MAIN}I1.csv",
    "🇫🇷 Ligue 1": f"{BASE_MAIN}F1.csv",
    "🇳🇱 Eredivisie": f"{BASE_MAIN}N1.csv",
    "🇵🇹 Liga Portugal": f"{BASE_MAIN}P1.csv",
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Premiership": f"{BASE_MAIN}SC0.csv",
    "🇹🇷 Turkey": f"{BASE_EXTRA}TUR.csv"
}

# --- 2. THE ENGINE (Goals + Results) ---
@st.cache_data(ttl=1800)
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    # Scoring Form & Defense
    for s in ["FTHG", "FTAG"]:
        df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    # Goal Logic (Over 2.5)
    df["Over25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
    return df

def train_models(url, df):
    code = url.split("/")[-1].replace(".csv", "")
    res_path, goal_path = f"models/{code}_res.pkl", f"models/{code}_goal.pkl"
    X = df[["FTHG_roll", "FTAG_roll", "HG_conc_roll", "AG_conc_roll"]]
    
    if not os.path.exists(res_path):
        y_res = df["FTR"].map({"H": 2, "D": 1, "A": 0})
        m_res = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_res)
        with open(res_path, "wb") as f: pickle.dump(m_res, f)
        
    if not os.path.exists(goal_path):
        y_goal = df["Over25"]
        m_goal = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_goal)
        with open(goal_path, "wb") as f: pickle.dump(m_goal, f)
    clear_ram()

@st.cache_resource
def get_model(url, m_type):
    code = url.split("/")[-1].replace(".csv", "")
    with open(f"models/{code}_{m_type}.pkl", "rb") as f: return pickle.load(f)

# --- 3. UI ---
sel_league = st.sidebar.selectbox("Competition", list(LEAGUES.keys()))
df = load_data(LEAGUES[sel_league])
train_models(LEAGUES[sel_league], df)
m_res = get_model(LEAGUES[sel_league], "res")
m_goal = get_model(LEAGUES[sel_league], "goal")

teams = sorted(df["HomeTeam"].unique())
c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("🏠 Home", teams)
    h_odd = st.number_input(f"Sporty Odd: {h_team}", 1.01, 50.0, 1.80)
with c2:
    a_team = st.selectbox("🚩 Away", teams, index=1)
    a_odd = st.number_input(f"Sporty Odd: {a_team}", 1.01, 50.0, 3.50)

if st.button("🚀 DEEP ANALYSIS"):
    h_data = df[df["HomeTeam"] == h_team].iloc[-1]
    a_data = df[df["AwayTeam"] == a_team].iloc[-1]
    X_input = [[h_data["FTHG_roll"], a_data["FTAG_roll"], h_data["HG_conc_roll"], a_data["AG_conc_roll"]]]
    
    # Predict Results & Goals
    p_res = m_res.predict_proba(X_input)[0]
    p_goal = m_goal.predict_proba(X_input)[0][1] # Probability of Over 2.5
    
    st.divider()
    # 1. RESULTS SECTION
    st.subheader("📊 Match Outcome")
    cols = st.columns(3)
    cols[0].metric(h_team, f"{p_res[2]*100:.1f}%")
    cols[1].metric("Draw", f"{p_res[1]*100:.1f}%")
    cols[2].metric(a_team, f"{p_res[0]*100:.1f}%")

    # 2. GOALS SECTION (NEW & VISIBLE)
    st.subheader("⚽ Goal Forecast")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.write(f"**Over 2.5 Goals:** {p_goal*100:.1f}%")
        st.progress(p_goal)
    with g_col2:
        st.write(f"**Under 2.5 Goals:** {(1-p_goal)*100:.1f}%")
        st.progress(1-p_goal)

    # 3. SAFETY & TRAPS
    with st.expander("🛡️ Strategic Insights & Market Traps"):
        # Double Chance
        st.write(f"**Double Chance 1X:** {(p_res[2]+p_res[1])*100:.1f}%")
        st.write(f"**Double Chance X2:** {(p_res[0]+p_res[1])*100:.1f}%")
        
        # Odd Effect
        imp_h = 1 / h_odd
        if p_res[2] - imp_h > 0.20: st.success("💎 VALUE: Bookie odds are high for a strong Home team.")
        elif imp_h - p_res[2] > 0.25: st.error("🚨 SPORTY TRAP: Home team is overpriced. Avoid!")

    clear_ram()
    
