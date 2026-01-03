import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import time

# --- 1. CONFIG & STABILITY ---
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
Path("models").mkdir(exist_ok=True)

def clear_ram():
    gc.collect()

# --- 2. DATA REPOSITORY & ADVANCED PROCESSING ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
BASE_EXTRA = "https://www.football-data.co.uk/new_leagues/"

LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": f"{BASE_MAIN}E0.csv",
    "🇪🇸 La Liga": f"{BASE_MAIN}SP1.csv",
    "🇩🇪 Bundesliga": f"{BASE_MAIN}D1.csv",
    "🇮🇹 Serie A": f"{BASE_MAIN}I1.csv",
    "🇫🇷 Ligue 1": f"{BASE_MAIN}F1.csv",
    "🇵🇹 Liga Portugal": f"{BASE_MAIN}P1.csv",      # Added
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Premiership": f"{BASE_MAIN}SC0.csv",    # Added
    "🇳🇱 Eredivisie": f"{BASE_MAIN}N1.csv"
}

def calculate_elo(df):
    """Memory-efficient Elo calculation."""
    unique_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo = {team: 1500 for team in unique_teams}
    K = 30
    # Vectorized check for performance
    for _, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        r_h, r_a = elo[h], elo[a]
        e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0)
        elo[h] += K * (s_h - e_h)
        elo[a] += K * ((1 - s_h) - (1 - e_h))
    return elo

@st.cache_data(ttl=3600)
def load_and_enhance_data(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        
        # Shots on Target for 'Threat' Index
        df['HST'] = df.get('HST', 0).fillna(0)
        df['AST'] = df.get('AST', 0).fillna(0)
        
        # Rolling Metrics (Last 5)
        df['H_Threat'] = df.groupby('HomeTeam')['HST'].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
        df['A_Threat'] = df.groupby('AwayTeam')['AST'].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
        
        df["Over25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        elo_map = calculate_elo(df)
        return df, elo_map
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame(), {}

def train_professional_models(url, df, elo_map):
    if df.empty: return
    code = url.split("/")[-1].replace(".csv", "")
    paths = {"res": f"models/{code}_res.pkl", "goal": f"models/{code}_goal.pkl", "btts": f"models/{code}_btts.pkl"}
    
    # Train if local model is missing or >24h old
    if not os.path.exists(paths["res"]) or (time.time() - os.path.getmtime(paths["res"]) > 86400):
        df['Elo_Diff'] = df['HomeTeam'].map(elo_map) - df['AwayTeam'].map(elo_map)
        X = df[["H_Threat", "A_Threat", "Elo_Diff"]].fillna(0)
        
        # Light estimators to keep RAM < 1GB
        params = {"n_estimators": 50, "max_depth": 5}
        pickle.dump(RandomForestClassifier(**params).fit(X, df["FTR"].map({"H":2,"D":1,"A":0})), open(paths["res"], "wb"))
        pickle.dump(RandomForestClassifier(**params).fit(X, df["Over25"]), open(paths["goal"], "wb"))
        pickle.dump(RandomForestClassifier(**params).fit(X, df["BTTS"]), open(paths["btts"], "wb"))
    clear_ram()

@st.cache_resource(max_entries=3) # Limit cache size to prevent memory leaks
def get_model(path):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None

# --- 3. UI ---
sel_league = st.sidebar.selectbox("Elite Competition", list(LEAGUES.keys()))
df, elo_map = load_and_enhance_data(LEAGUES[sel_league])

if not df.empty:
    train_professional_models(LEAGUES[sel_league], df, elo_map)
    teams = sorted(df["HomeTeam"].unique())

    c1, c2 = st.columns(2)
    with c1:
        h_team = st.selectbox("🏠 Home Team", teams)
        h_fatigue = st.select_slider(f"Fatigue ({h_team})", ["Fresh", "Normal", "Tired"], "Normal")
        h_missing = st.multiselect(f"Missing ({h_team})", ["Strikers", "Midfield", "Defense"], key="h_m")
        is_fortress = st.toggle("🏟️ Home Fortress Effect")
        
    with c2:
        a_team = st.selectbox("🚩 Away Team", teams, index=1 if len(teams)>1 else 0)
        a_fatigue = st.select_slider(f"Fatigue ({a_team})", ["Fresh", "Normal", "Tired"], "Normal")
        a_missing = st.multiselect(f"Missing ({a_team})", ["Strikers", "Midfield", "Defense"], key="a_m")

    if st.button("🚀 EXECUTE TOP-NOTCH ANALYSIS"):
        code = LEAGUES[sel_league].split("/")[-1].replace(".csv", "")
        m_res = get_model(f"models/{code}_res.pkl")
        m_goal = get_model(f"models/{code}_goal.pkl")
        m_btts = get_model(f"models/{code}_btts.pkl")

        if m_res:
            h_elo, a_elo = elo_map[h_team], elo_map[a_team]
            h_threat = df[df["HomeTeam"] == h_team]["H_Threat"].iloc[-1]
            a_threat = df[df["AwayTeam"] == a_team]["A_Threat"].iloc[-1]
            
            X_input = [[h_threat, a_threat, h_elo - a_elo]]
            
            p_res = m_res.predict_proba(X_input)[0]
            f_h, f_d, f_a = p_res[2], p_res[1], p_res[0]

            # Situational Adjustments
            fat_mod = {"Fresh": 0.04, "Normal": 0.0, "Tired": -0.08}
            f_h += fat_mod[h_fatigue] - (len(h_missing)*0.04) + (0.05 if is_fortress else 0)
            f_a += fat_mod[a_fatigue] - (len(a_missing)*0.04)

            probs = np.clip([f_h, f_d, f_a], 0.01, 0.98)
            f_h, f_d, f_a = probs / probs.sum()

            st.divider()
            st.subheader(f"📊 {h_team} vs {a_team}")
            
            # Prediction Metrics
            res_cols = st.columns(3)
            res_cols[0].metric(h_team, f"{f_h*100:.1f}%")
            res_cols[1].metric("Draw", f"{f_d*100:.1f}%")
            res_cols[2].metric(a_team, f"{f_a*100:.1f}%")

            p_goal = m_goal.predict_proba(X_input)[0][1]
            p_btts = m_btts.predict_proba(X_input)[0][1]
            
            g1, g2 = st.columns(2)
            with g1:
                st.write("**Goal Line (Over 2.5)**")
                st.progress(float(p_goal))
                st.caption(f"Probability: {p_goal*100:.1f}%")
            with g2:
                st.write("**BTTS (GG Probability)**")
                st.progress(float(p_btts))
                st.caption(f"Probability: {p_btts*100:.1f}%")

            clear_ram()
            
