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
st.set_page_config(page_title="AI Match Master Ultra", page_icon="🔮", layout="wide")
Path("models").mkdir(exist_ok=True)

def clear_ram():
    gc.collect()

# --- 2. DATA REPOSITORY (All Leagues Intact) ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
BASE_EXTRA = "https://www.football-data.co.uk/new_leagues/"

LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": f"{BASE_MAIN}E0.csv",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship": f"{BASE_MAIN}E1.csv",
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

@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean())).fillna(0)
        df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean())).fillna(0)
        df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean())).fillna(0)
        df["Over25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

def train_models(url, df):
    if df.empty: return
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

# --- 3. UI & INPUTS ---
sel_league = st.sidebar.selectbox("Competition", list(LEAGUES.keys()))
df = load_data(LEAGUES[sel_league])

if not df.empty:
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
        st.info(f"📅 Last Update: {df['Date'].iloc[-1]}")
        st.write("**Recent Team Form**")
        for t in teams: st.caption(f"{get_form_str(t)} {t}")

    c1, c2 = st.columns(2)
    with c1:
        h_team = st.selectbox("🏠 Home Team", teams)
        h_odd = st.number_input(f"Odd: {h_team}", 1.01, 50.0, 1.80)
        h_fatigue = st.select_slider(f"Fatigue ({h_team})", ["Fresh", "Normal", "Tired"], "Normal")
        h_missing = st.multiselect(f"Missing ({h_team})", ["Strikers", "Midfield", "Defense"])
        is_fortress = st.toggle("🏟️ Home Fortress Effect")
        
    with c2:
        a_team = st.selectbox("🚩 Away Team", teams, index=1 if len(teams) > 1 else 0)
        a_odd = st.number_input(f"Odd: {a_team}", 1.01, 50.0, 3.50)
        a_fatigue = st.select_slider(f"Fatigue ({a_team})", ["Fresh", "Normal", "Tired"], "Normal")
        a_missing = st.multiselect(f"Missing ({a_team})", ["Strikers", "Midfield", "Defense"])

    # --- 4. ANALYSIS ---
    if st.button("🚀 RUN SMART ANALYSIS"):
        h_row, a_row = df[df["HomeTeam"] == h_team].iloc[-1], df[df["AwayTeam"] == a_team].iloc[-1]
        X_input = [[h_row["FTHG_roll"], a_row["FTAG_roll"], h_row["HG_conc_roll"], a_row["AG_conc_roll"]]]
        
        p_now = m_res.predict_proba(X_input)[0]
        f_h, f_d, f_a = p_now[2], p_now[1], p_now[0]

        # Adjustments
        fat_map = {"Fresh": 0.03, "Normal": 0.0, "Tired": -0.07}
        f_h += (fat_map[h_fatigue] - len(h_missing)*0.05 + (0.05 if is_fortress else 0))
        f_a += (fat_map[a_fatigue] - len(a_missing)*0.05)
        
        f_h, f_a = max(0.01, f_h), max(0.01, f_a)
        f_d = max(0.01, 1 - f_h - f_a)

        st.divider()
        st.subheader("📊 Adjusted Win Probabilities")
        cols = st.columns(3)
        cols[0].metric(h_team, f"{f_h*100:.1f}%")
        cols[1].metric("Draw", f"{f_d*100:.1f}%")
        cols[2].metric(a_team, f"{f_a*100:.1f}%")

        # Goal Forecast
        p_goal = m_goal.predict_proba(X_input)[0][1]
        st.subheader("⚽ Goal Forecast")
        g1, g2 = st.columns(2)
        g1.write(f"**Over 2.5:** {p_goal*100:.1f}%")
        g1.progress(p_goal)
        g2.write(f"**Under 2.5:** {(1-p_goal)*100:.1f}%")
        g2.progress(1-p_goal)

        # Strategic Insights & Recommendations
        st.subheader("🎯 AI Recommendations")
        rec1, rec2 = st.columns(2)
        with rec1:
            if (f_h + f_d) > 0.82: st.success(f"🛡️ SAFE: 1X ({h_team}/Draw)")
            elif (f_a + f_d) > 0.82: st.success(f"🛡️ SAFE: X2 ({a_team}/Draw)")
            else: st.warning("⚖️ BALANCE: High risk for Double Chance.")
        
        with rec2:
            imp_h = 1 / h_odd
            if f_h - imp_h > 0.18: st.info("💎 VALUE: Home win potential is underrated.")
            elif h_fatigue == "Tired": st.error("🚨 ALERT: Home team is heavily fatigued.")

        clear_ram()
else:
    st.warning("⚠️ Could not load data.")
    
