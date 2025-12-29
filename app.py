
    clear_ram()
    

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --- CONFIG & RAM MANAGEMENT ---
st.set_page_config(page_title="AI Match Master Ultra", page_icon="🔮", layout="wide")
Path("models").mkdir(exist_ok=True)

def clear_ram():
    gc.collect()

# --- 1. FULL DATA REPOSITORY (From your shared code) ---
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

# --- 2. THE ULTRA ENGINE (New Features) ---
def get_league_code(url):
    return url.split("/")[-1].replace(".csv", "")

def train_and_save(url):
    code = get_league_code(url)
    res_path = f"models/{code}_ultra_res.pkl"
    
    if not os.path.exists(res_path):
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
            
            # --- FEATURE ENGINEERING ---
            # 1. Attacking Power (Goals Scored)
            for s in ["FTHG", "FTAG"]:
                df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
            
            # 2. Defensive Wall (Goals Conceded)
            df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
            df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
            
            # 3. Half-Time "Stability" (HT Results)
            df["HT_Points"] = df["HTR"].map({"H": 3, "D": 1, "A": 0})
            df["HT_form"] = df.groupby("HomeTeam")["HT_Points"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))

            y_res = df["FTR"].map({"H": 2, "D": 1, "A": 0})
            X = df[["FTHG_roll", "FTAG_roll", "HG_conc_roll", "AG_conc_roll", "HT_form"]]

            # Efficient Model setup
            model = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_res)

            with open(res_path, "wb") as f: pickle.dump(model, f)
            del df, X, y_res
            clear_ram()
        except Exception: pass

@st.cache_resource
def load_tools(url):
    train_and_save(url)
    code = get_league_code(url)
    with open(f"models/{code}_ultra_res.pkl", "rb") as f: res = pickle.load(f)
    return res

@st.cache_data(ttl=1800)
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    # Apply same feature engineering to the live data
    for s in ["FTHG", "FTAG"]:
        df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HT_Points"] = df["HTR"].map({"H": 3, "D": 1, "A": 0})
    df["HT_form"] = df.groupby("HomeTeam")["HT_Points"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    return df

# --- 3. UI ---
sel_league = st.sidebar.selectbox("Select Competition", list(LEAGUES.keys()))
target_url = LEAGUES[sel_league]

data = load_data(target_url)
rf_ultra = load_tools(target_url)

teams = sorted(data["HomeTeam"].unique())
c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("🏠 Home Team", teams)
    h_odd = st.number_input(f"SportyBet {home} Odd", 1.01, 50.0, 2.0)
with c2:
    away = st.selectbox("🚩 Away Team", teams, index=1)
    a_odd = st.number_input(f"SportyBet {away} Odd", 1.01, 50.0, 2.0)

if st.button("🚀 RUN ULTRA ANALYSIS"):
    h_row = data[data["HomeTeam"] == home].iloc[-1]
    a_row = data[data["AwayTeam"] == away].iloc[-1]
    
    # Feature Input: Attack, Defense, and HT Stability
    X_input = [[h_row["FTHG_roll"], a_row["FTAG_roll"], h_row["HG_conc_roll"], a_row["AG_conc_roll"], h_row["HT_form"]]]
    p_res = rf_ultra.predict_proba(X_input)[0]

    # Weighted Calculation
    ha_bias = 0.10 if a_row["FTAG_roll"] < 1.7 else 0.02
    f_h, f_d, f_a = max(0, p_res[2]-ha_bias), p_res[1], min(1, p_res[0]+ha_bias)

    st.divider()
    st.subheader("📊 Match Probability (Ultra Logic)")
    colA, colB, colC = st.columns(3)
    colA.metric(f"{home} Win", f"{f_h*100:.1f}%")
    colB.metric("Draw", f"{f_d*100:.1f}%")
    colC.metric(f"{away} Win", f"{f_a*100:.1f}%")

    # DOUBLE CHANCE Logic
    st.subheader("🛡️ Strategic Picks")
    st.info(f"💡 **1X (Home or Draw):** {(f_h+f_d)*100:.1f}% | **X2 (Away or Draw):** {(f_a+f_d)*100:.1f}%")
    
    # TRAP WARNING
    if (f_a > 0.58 and h_odd < 1.75):
        st.error(f"🚨 **MARKET TRAP:** Bookies favor {home}, but stats show {away} has a superior defensive wall and scoring consistency.")

    clear_ram()
    
