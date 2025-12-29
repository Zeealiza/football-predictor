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

# --- 1. LEAGUE REPOSITORY ---
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
@st.cache_data(ttl=1800)
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    # Feature Engineering
    for s in ["FTHG", "FTAG"]:
        df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HG_conc_roll"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["AG_conc_roll"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    df["HT_roll"] = df["HTR"].map({"H": 3, "D": 1, "A": 0}).groupby(df["HomeTeam"]).transform(lambda x: x.rolling(5, closed='left').mean().fillna(x.mean()))
    return df

def train_and_save(url, df):
    code = url.split("/")[-1].replace(".csv", "")
    res_path = f"models/{code}_v4.pkl"
    if not os.path.exists(res_path):
        y = df["FTR"].map({"H": 2, "D": 1, "A": 0})
        X = df[["FTHG_roll", "FTAG_roll", "HG_conc_roll", "AG_conc_roll", "HT_roll"]]
        model = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y)
        with open(res_path, "wb") as f: pickle.dump(model, f)
        clear_ram()

@st.cache_resource
def get_model(url):
    code = url.split("/")[-1].replace(".csv", "")
    with open(f"models/{code}_v4.pkl", "rb") as f: return pickle.load(f)

# --- 3. UI & FORM VISUALIZER ---
sel_league = st.sidebar.selectbox("Competition", list(LEAGUES.keys()))
df = load_data(LEAGUES[sel_league])
train_and_save(LEAGUES[sel_league], df)
rf_model = get_model(LEAGUES[sel_league])

teams = sorted(df["HomeTeam"].unique())

# Sidebar Form Guide
def get_form(team):
    results = df[(df['HomeTeam']==team) | (df['AwayTeam']==team)].tail(5)
    form = []
    for _, row in results.iterrows():
        if row['FTR'] == 'D': form.append("➖")
        elif (row['HomeTeam']==team and row['FTR']=='H') or (row['AwayTeam']==team and row['FTR']=='A'): form.append("✅")
        else: form.append("❌")
    return "".join(form)

with st.sidebar:
    st.divider()
    st.write("**Recent Team Form**")
    for t in teams: st.caption(f"{get_form(t)} {t}")

# Main Layout
c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("🏠 Home", teams)
    h_odd = st.number_input(f"Odd: {h_team}", 1.01, 50.0, 1.50)
with c2:
    a_team = st.selectbox("🚩 Away", teams, index=1)
    a_odd = st.number_input(f"Odd: {a_team}", 1.01, 50.0, 2.50)

if st.button("🚀 ANALYZE MATCH"):
    h_data = df[df["HomeTeam"] == h_team].iloc[-1]
    a_data = df[df["AwayTeam"] == a_team].iloc[-1]
    
    X_input = [[h_data["FTHG_roll"], a_data["FTAG_roll"], h_data["HG_conc_roll"], a_data["AG_conc_roll"], h_data["HT_roll"]]]
    probs = rf_model.predict_proba(X_input)[0]
    
    # Calculate Bias
    ha_bias = 0.08 if a_data["FTAG_roll"] < 1.5 else 0.02
    p_h, p_d, p_a = max(0, probs[2]-ha_bias), probs[1], min(1, probs[0]+ha_bias)

    # OUTPUT
    st.divider()
    st.subheader("📊 Win Probability")
    cols = st.columns(3)
    cols[0].metric(h_team, f"{p_h*100:.1f}%")
    cols[1].metric("Draw", f"{p_d*100:.1f}%")
    cols[2].metric(a_team, f"{p_a*100:.1f}%")

    st.subheader("🛡️ Safety (Double Chance)")
    st.info(f"💡 **1X:** {(p_h+p_d)*100:.1f}% | **X2:** {(p_a+p_d)*100:.1f}%")

    # --- ODD EFFECT & TRAP DETECTION ---
    st.subheader("🕵️ Market Sentiment")
    imp_h = 1 / h_odd
    gap = p_h - imp_h

    if gap > 0.20:
        st.success(f"💎 VALUE: {h_team} is stronger than Odd {h_odd} suggests. AI sees hidden strength.")
    elif gap < -0.25:
        st.error(f"🚨 SPORTY TRAP: {h_team} is overpriced (Low Odd {h_odd}). AI stats say they are weak. Beware!")
    else:
        st.write("✅ Odds are fair and match team performance.")

    clear_ram()
    
