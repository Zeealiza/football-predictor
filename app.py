import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import time

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
    "🇹🇷 Turkey": f"{BASE_EXTRA}TUR.csv",
    "🇬🇷 Greece": f"{BASE_EXTRA}GREECE.csv",
    "🇩🇰 Denmark": f"{BASE_EXTRA}DNK.csv"
}

# --- 2. ENGINE (Auto-Updating & RAM-Safe) ---
@st.cache_data(ttl=3600)
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    for s in ["FTHG", "FTAG"]:
        df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["Over25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
    return df

def train_models(url, df):
    code = url.split("/")[-1].replace(".csv", "")
    res_path, goal_path = f"models/{code}_res.pkl", f"models/{code}_goal.pkl"
    needs_train = not os.path.exists(res_path) or (time.time() - os.path.getmtime(res_path) > 86400)
    
    if needs_train:
        X = df[["FTHG_roll", "FTAG_roll", "HG_conc_roll", "AG_conc_roll"]]
        y_res = df["FTR"].map({"H": 2, "D": 1, "A": 0})
        with open(res_path, "wb") as f:
            pickle.dump(RandomForestClassifier(n_estimators=50, max_depth=5).fit(X, y_res), f)
        y_goal = df["Over25"]
        with open(goal_path, "wb") as f:
            pickle.dump(RandomForestClassifier(n_estimators=50, max_depth=5).fit(X, y_goal), f)
        get_model.clear()
    clear_ram()

@st.cache_resource(max_entries=2)
def get_model(url, m_type):
    code = url.split("/")[-1].replace(".csv", "")
    with open(f"models/{code}_{m_type}.pkl", "rb") as f:
        return pickle.load(f)

# --- 3. UI ---
sel_league = st.sidebar.selectbox("Competition", list(LEAGUES.keys()))
df = load_data(LEAGUES[sel_league])
train_models(LEAGUES[sel_league], df)
m_res, m_goal = get_model(LEAGUES[sel_league], "res"), get_model(LEAGUES[sel_league], "goal")

teams = sorted(df["HomeTeam"].unique())

def get_form_str(team):
    results = df[(df['HomeTeam']==team) | (df['AwayTeam']==team)].tail(5)
    f = []
    for _, row in results.iterrows():
        if row['FTR'] == 'D': f.append("➖")
        elif (row['HomeTeam']==team and row['FTR']=='H') or (row['AwayTeam']==team and row['FTR']=='A'): f.append("✅")
        else: f.append("❌")
    return "".join(f)

with st.sidebar:
    st.divider()
    # Visibility Fix: Added Freshness Info and Refresh Button
    last_date = df['Date'].iloc[-1]
    st.info(f"📅 **Data up to:** {last_date}")
    if st.button("🔄 Force Refresh Data"):
        st.cache_data.clear()
        get_model.clear()
        st.rerun()
    
    st.write("**Recent Team Form**")
    for t in teams: st.caption(f"{get_form_str(t)} {t}")

c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("🏠 Home Team", teams)
    h_odd = st.number_input(f"Sporty Odd: {h_team}", 1.01, 50.0, 1.80)
    h_missing = st.multiselect(f"Missing (Home)", ["Star Striker", "Key Playmaker", "Main Defender"], key="h_miss")
with c2:
    a_team = st.selectbox("🚩 Away Team", teams, index=1)
    a_odd = st.number_input(f"Sporty Odd: {a_team}", 1.01, 50.0, 3.50)
    a_missing = st.multiselect(f"Missing (Away)", ["Star
                                                   
