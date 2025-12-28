import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from duckduckgo_search import DDGS

# --- 1. CONFIG & LIVE INTEL ---
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Match Master Pro – XGBoost Cloud Edition")

def get_intel(team):
    try:
        with DDGS() as ddgs:
            q = f"{team} injuries lineup today"
            r = list(ddgs.text(q, max_results=2))
            return r if r else [{"body": "No recent updates"}]
    except:
        return [{"body": "Intel unavailable"}]

# --- 2. FEATURE ENGINEERING (No lines removed) ---
def build_table(df):
    table = {}
    teams = pd.concat([df.HomeTeam, df.AwayTeam]).unique()
    for t in teams:
        home = df[df.HomeTeam == t]
        away = df[df.AwayTeam == t]
        points = ((home.FTR == "H").sum()*3 + (home.FTR == "D").sum() +
                  (away.FTR == "A").sum()*3 + (away.FTR == "D").sum())
        played = len(home) + len(away)
        table[t] = points / played if played else 0
    return pd.Series(table, name="ppg")

def add_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    for g in ["FTHG", "FTAG"]:
        df[f"{g}_roll"] = df.groupby("HomeTeam")[g].transform(lambda x: x.rolling(5, closed="left").mean().fillna(x.mean()))
    pts = {"H":3,"D":1,"A":0}
    df["home_form"] = df.groupby("HomeTeam")["FTR"].transform(lambda x: x.map(pts).rolling(5).mean())
    df["away_form"] = df.groupby("AwayTeam")["FTR"].transform(lambda x: x.map(pts).rolling(5).mean())
    df["form_diff"] = df["home_form"] - df["away_form"]
    df["home_rest"] = df.groupby("HomeTeam")["Date"].diff().dt.days.fillna(7)
    df["away_rest"] = df.groupby("AwayTeam")["Date"].diff().dt.days.fillna(7)
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    ppg = build_table(df)
    df["home_ppg"] = df["HomeTeam"].map(ppg)
    df["away_ppg"] = df["AwayTeam"].map(ppg)
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]
    df.fillna(0, inplace=True)
    return df

# --- 3. OPTIMIZED CLOUD TRAINING ---
@st.cache_data(ttl=86400) # Cache for 24 hours to save time
def train_models(url):
    df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["FTR","HomeTeam","AwayTeam"])
    df = add_features(df)
    df["res_target"] = df.FTR.map({"H":2,"D":1,"A":0})
    df["o25"] = ((df.FTHG + df.FTAG) > 2.5).astype(int)
    df["btts"] = ((df.FTHG > 0) & (df.FTAG > 0)).astype(int)

    features = ["FTHG_roll","FTAG_roll","ppg_diff","form_diff","rest_diff"]

    # Optimizing estimators from 500 down to 100 for Cloud Speed
    m_res = XGBClassifier(n_estimators=100, max_depth=3, tree_method="hist", objective="multi:softprob", num_class=3)
    m_o25 = XGBClassifier(n_estimators=100, max_depth=3, tree_method="hist")
    m_gg = XGBClassifier(n_estimators=100, max_depth=3, tree_method="hist")

    X = df[features]
    m_res.fit(X, df["res_target"])
    m_o25.fit(X, df["o25"])
    m_gg.fit(X, df["btts"])
    return df, m_res, m_o25, m_gg, features

# --- 4. UI & EXECUTION ---
BASE = "https://www.football-data.co.uk/mmz4281/2526/"
LEAGUES = {"Premier League": f"{BASE}E0.csv", "La Liga": f"{BASE}SP1.csv", "Bundesliga": f"{BASE}D1.csv", "Serie A": f"{BASE}I1.csv"}

league = st.sidebar.selectbox("League", LEAGUES.keys())
df, m_res, m_o25, m_gg, feats = train_models(LEAGUES[league])

teams = sorted(df.HomeTeam.unique())
c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("🏠 Home Team", teams)
    h_abs = st.slider("Home absences", 0, 5, 0)
with c2:
    away = st.selectbox("🚩 Away Team", teams)
    a_abs = st.slider("Away absences", 0, 5, 0)

if st.button("🚀 RUN AI PREDICTION"):
    row = df[df.HomeTeam == home].iloc[-1]
    Xp = row[feats].values.reshape(1,-1)
    p_res = m_res.predict_proba(Xp)[0]
    p_o25 = m_o25.predict_proba(Xp)[0][1]
    p_gg = m_gg.predict_proba(Xp)[0][1]

    adj = 0.08
    home_p = max(0, min(1, p_res[2] - h_abs*adj + a_abs*adj))
    away_p = max(0, min(1, p_res[0] - a_abs*adj + h_abs*adj))
    draw_p = max(0, 1 - home_p - away_p)

    st.subheader("🏆 Match Result")
    a,b,c = st.columns(3)
    a.metric(home, f"{home_p*100:.1f}%")
    b.metric("Draw", f"{draw_p*100:.1f}%")
    c.metric(away, f"{away_p*100:.1f}%")

    st.subheader("⚽ Goals")
    st.success(f"Over 2.5 Goals: {p_o25*100:.1f}%")
    st.info(f"BTTS: {p_gg*100:.1f}%")

    st.divider()
    outcome = home if home_p > max(away_p, draw_p) else away if away_p > draw_p else "Draw"
    confidence = max(home_p, away_p, draw_p)
    st.warning(f"🧠 AI VERDICT: **{outcome}** ({confidence*100:.1f}% confidence)")

    st.subheader("🗞️ Team News")
    for n in get_intel(home): st.caption(f"{home}: {n['body'][:120]}...")
    for n in get_intel(away): st.caption(f"{away}: {n['body'][:120]}...")
        
