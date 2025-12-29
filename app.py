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
    "🇹🇷 Turkey": f"{BASE_EXTRA}TUR.csv",
    "🇬🇷 Greece": f"{BASE_EXTRA}GREECE.csv",
    "🇩🇰 Denmark": f"{BASE_EXTRA}DNK.csv"
}

# --- 2. THE ENGINE ---
@st.cache_data(ttl=1800)
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
    X = df[["FTHG_roll", "FTAG_roll", "HG_conc_roll", "AG_conc_roll"]]
    if not os.path.exists(res_path):
        y_res = df["FTR"].map({"H": 2, "D": 1, "A": 0})
        pickle.dump(RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_res), open(res_path, "wb"))
    if not os.path.exists(goal_path):
        y_goal = df["Over25"]
        pickle.dump(RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y_goal), open(goal_path, "wb"))
    clear_ram()

@st.cache_resource
def get_model(url, m_type):
    code = url.split("/")[-1].replace(".csv", "")
    return pickle.load(open(f"models/{code}_{m_type}.pkl", "rb"))

# --- 3. UI & FORM VISUALIZER ---
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
    st.write("**Recent Team Form**")
    for t in teams: st.caption(f"{get_form_str(t)} {t}")

c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("🏠 Home Team", teams)
    h_odd = st.number_input(f"Sporty Odd: {h_team}", 1.01, 50.0, 1.80)
with c2:
    a_team = st.selectbox("🚩 Away Team", teams, index=1)
    a_odd = st.number_input(f"Sporty Odd: {a_team}", 1.01, 50.0, 3.50)

# --- 4. ANALYSIS LOGIC ---
if st.button("🚀 RUN FULL ANALYSIS"):
    h_row = df[df["HomeTeam"] == h_team].iloc[-1]
    a_row = df[df["AwayTeam"] == a_team].iloc[-1]
    X_input = [[h_row["FTHG_roll"], a_row["FTAG_roll"], h_row["HG_conc_roll"], a_row["AG_conc_roll"]]]
    
    p_now = m_res.predict_proba(X_input)[0]
    p_goal = m_goal.predict_proba(X_input)[0][1]

    # H2H JUSTICE FILTER
    h2h = df[((df['HomeTeam']==h_team) & (df['AwayTeam']==a_team)) | ((df['HomeTeam']==a_team) & (df['AwayTeam']==h_team))]
    if not h2h.empty:
        h_h2h = len(h2h[h2h['FTR'] == ('H' if h2h['HomeTeam'].iloc[0] == h_team else 'A')]) / len(h2h)
        a_h2h = len(h2h[h2h['FTR'] == ('A' if h2h['HomeTeam'].iloc[0] == h_team else 'H')]) / len(h2h)
        # Blend: 85% Current Form + 15% History
        f_h, f_a = (p_now[2]*0.85 + h_h2h*0.15), (p_now[0]*0.85 + a_h2h*0.15)
        f_d = 1 - f_h - f_a
    else:
        f_h, f_d, f_a = p_now[2], p_now[1], p_now[0]

    # --- OUTPUTS ---
    st.divider()
    st.subheader("📊 Balanced Outcome (Form + H2H History)")
    cols = st.columns(3)
    cols[0].metric(h_team, f"{f_h*100:.1f}%")
    cols[1].metric("Draw", f"{f_d*100:.1f}%")
    cols[2].metric(a_team, f"{f_a*100:.1f}%")

    st.subheader("⚽ Goal Forecast (Over/Under 2.5)")
    g1, g2 = st.columns(2)
    g1.write(f"**Over 2.5:** {p_goal*100:.1f}%")
    g1.progress(p_goal)
    g2.write(f"**Under 2.5:** {(1-p_goal)*100:.1f}%")
    g2.progress(1-p_goal)

    with st.expander("🛡️ Strategic Insights, Traps & H2H"):
        st.info(f"💡 **Double Chance 1X:** {(f_h+f_d)*100:.1f}% | **Double Chance X2:** {(f_a+f_d)*100:.1f}%")
        
        # ODD EFFECT (The Trap Detector)
        imp_h = 1 / h_odd
        if f_h - imp_h > 0.18: st.success("💎 VALUE: AI sees high potential for Home Win compared to Odd.")
        elif imp_h - f_h > 0.22: st.error("🚨 SPORTY TRAP: Odds are low, but AI stats say Home team is risky.")
        
        # FIXED H2H TABLE VIEW (No cutting off)
        if not h2h.empty:
            st.write("**Full Recent Head-to-Head Scores:**")
            st.table(h2h.tail(5)[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam']])
    
    clear_ram()
    
