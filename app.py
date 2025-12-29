import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --- CONFIG & SYSTEM SAFETY ---
st.set_page_config(page_title="AI Match Master Ultra", page_icon="🔮", layout="wide")
Path("models").mkdir(exist_ok=True)

def clear_ram():
    gc.collect()

# --- 1. DATA REPOSITORY ---
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

# --- 2. THE ENGINE ---
def get_league_code(url):
    return url.split("/")[-1].replace(".csv", "")

def train_and_save(url):
    code = get_league_code(url)
    res_path = f"models/{code}_v3_ultra.pkl"
    
    if not os.path.exists(res_path):
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
            
            # ATTACK & DEFENSE FEATURES
            for s in ["FTHG", "FTAG"]:
                df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
            
            df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
            df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
            
            # HALF-TIME STABILITY
            df["HT_form"] = df["HTR"].map({"H": 3, "D": 1, "A": 0})
            df["HT_roll"] = df.groupby("HomeTeam")["HT_form"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))

            y_res = df["FTR"].map({"H": 2, "D": 1, "A": 0})
            X = df[["FTHG_roll", "FTAG_roll", "HG_conc_roll", "AG_conc_roll", "HT_roll"]]

            model = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_res)
            with open(res_path, "wb") as f: pickle.dump(model, f)
            del df, X, y_res
            clear_ram()
        except Exception: pass

@st.cache_resource
def load_tools(url):
    train_and_save(url)
    code = get_league_code(url)
    with open(f"models/{code}_v3_ultra.pkl", "rb") as f: return pickle.load(f)

@st.cache_data(ttl=1800)
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    for s in ["FTHG", "FTAG"]:
        df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HT_roll"] = df["HTR"].map({"H": 3, "D": 1, "A": 0}).groupby(df["HomeTeam"]).transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    return df

# --- 3. UI & PREDICTION ---
sel_league = st.sidebar.selectbox("Competition", list(LEAGUES.keys()))
data = load_data(LEAGUES[sel_league])
rf_model = load_tools(LEAGUES[sel_league])

teams = sorted(data["HomeTeam"].unique())
c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("🏠 Home Team", teams)
    h_odd = st.number_input(f"SportyBet {home} Odd", 1.01, 50.0, 2.0)
with c2:
    away = st.selectbox("🚩 Away Team", teams, index=1)
    a_odd = st.number_input(f"SportyBet {away} Odd", 1.01, 50.0, 2.0)

if st.button("🚀 FULL PERFORMANCE ANALYSIS"):
    h_row = data[data["HomeTeam"] == home].iloc[-1]
    a_row = data[data["AwayTeam"] == away].iloc[-1]
    
    # AI INPUT
    X_input = [[h_row["FTHG_roll"], a_row["FTAG_roll"], h_row["HG_conc_roll"], a_row["AG_conc_roll"], h_row["HT_roll"]]]
    p_res = rf_model.predict_proba(X_input)[0]

    # PROBABILITY LOGIC
    ha_bias = 0.10 if a_row["FTAG_roll"] < 1.7 else 0.02
    f_h, f_d, f_a = max(0, p_res[2]-ha_bias), p_res[1], min(1, p_res[0]+ha_bias)

    # 1. VISUALIZE WIN CHANCE
    st.divider()
    st.subheader("📊 Match Forecast")
    colA, colB, colC = st.columns(3)
    colA.metric(f"{home} Win", f"{f_h*100:.1f}%")
    colB.metric("Draw", f"{f_d*100:.1f}%")
    colC.metric(f"{away} Win", f"{f_a*100:.1f}%")

    # 2. DOUBLE CHANCE & SAFETY (VISIBLE NOW)
    st.subheader("🛡️ Double Chance (Safety Picks)")
    s1, s2 = st.columns(2)
    s1.info(f"**1X (Home or Draw):** {(f_h+f_d)*100:.1f}%")
    s2.info(f"**X2 (Away or Draw):** {(f_a+f_d)*100:.1f}%")

    # 3. ENHANCED TRAP DETECTION
    st.subheader("🕵️ Market Trap Analysis")
    market_h_prob = 1 / h_odd
    market_a_prob = 1 / a_odd

    # The AI checks if the Odd is "lying" compared to performance
    if (f_h > market_h_prob + 0.18):
        st.warning(f"⚠️ VALUE DETECTED: AI says {home} is much stronger than SportyBet's Odd {h_odd} suggest.")
    elif (market_h_prob > f_h + 0.22):
        st.error(f"🚨 SPORTY TRAP: Odds for {home} are suspiciously low. Stats don't support it. Potential upset.")
    elif (f_a > market_a_prob + 0.18):
        st.warning(f"⚠️ VALUE DETECTED: AI favors {away} significantly more than the market.")
    else:
        st.success("✅ Market Alignment: Odds accurately reflect current team performance.")

    clear_ram()
    
