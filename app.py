import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from duckduckgo_search import DDGS

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Match Master Pro – Tactical Edition")

# --------------------------------------------------
# LIVE INTEL
# --------------------------------------------------
def get_intel(team):
    try:
        with DDGS() as ddgs:
            q = f"{team} injuries lineup today"
            r = list(ddgs.text(q, max_results=2))
            return r if r else [{"body":"No recent updates"}]
    except:
        return [{"body":"Intel unavailable"}]

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def build_table(df):
    table = {}
    teams = pd.concat([df.HomeTeam, df.AwayTeam]).unique()

    for t in teams:
        home = df[df.HomeTeam == t]
        away = df[df.AwayTeam == t]

        points = (
            (home.FTR == "H").sum()*3 +
            (home.FTR == "D").sum() +
            (away.FTR == "A").sum()*3 +
            (away.FTR == "D").sum()
        )
        played = len(home) + len(away)
        gd = (home.FTHG.sum() + away.FTAG.sum()) - (home.FTAG.sum() + away.FTHG.sum())

        table[t] = {
            "ppg": points / played if played else 0,
            "gd": gd
        }
    return pd.DataFrame(table).T

def add_features(df):
    df["Date"] = pd.to_datetime(df["Date"])

    # Rolling goals
    for g in ["FTHG", "FTAG"]:
        df[f"{g}_roll"] = df.groupby("HomeTeam")[g]\
            .transform(lambda x: x.rolling(5, closed="left").mean().fillna(x.mean()))

    # Form
    points_map = {"H":3, "D":1, "A":0}
    df["home_form"] = df.groupby("HomeTeam")["FTR"]\
        .transform(lambda x: x.map(points_map).rolling(5).mean())
    df["away_form"] = df.groupby("AwayTeam")["FTR"]\
        .transform(lambda x: x.map(points_map).rolling(5).mean())
    df["form_diff"] = df["home_form"] - df["away_form"]

    # Fatigue
    df["home_rest"] = df.groupby("HomeTeam")["Date"].diff().dt.days.fillna(7)
    df["away_rest"] = df.groupby("AwayTeam")["Date"].diff().dt.days.fillna(7)
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # Table strength
    table = build_table(df)
    df["home_ppg"] = df["HomeTeam"].map(table["ppg"])
    df["away_ppg"] = df["AwayTeam"].map(table["ppg"])
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

    df.fillna(0, inplace=True)
    return df

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------
@st.cache_data(ttl=3600)
def train_models(url):
    df = pd.read_csv(url, storage_options={'User-Agent':'Mozilla/5.0'})
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["FTR","HomeTeam","AwayTeam"])
    df = add_features(df)

    df["res_target"] = df.FTR.map({"H":2,"D":1,"A":0})
    df["o25"] = ((df.FTHG + df.FTAG) > 2.5).astype(int)
    df["btts"] = ((df.FTHG>0) & (df.FTAG>0)).astype(int)

    feats = [
        "FTHG_roll","FTAG_roll",
        "ppg_diff","form_diff",
        "rest_diff"
    ]

    base = RandomForestClassifier(n_estimators=300, max_depth=8)
    res = CalibratedClassifierCV(base).fit(df[feats], df["res_target"])
    g25 = CalibratedClassifierCV(base).fit(df[feats], df["o25"])
    gg = CalibratedClassifierCV(base).fit(df[feats], df["btts"])

    return df, res, g25, gg, feats

# --------------------------------------------------
# DATA
# --------------------------------------------------
BASE = "https://www.football-data.co.uk/mmz4281/2526/"
LEAGUES = {
    "Premier League": f"{BASE}E0.csv",
    "La Liga": f"{BASE}SP1.csv",
    "Bundesliga": f"{BASE}D1.csv",
    "Serie A": f"{BASE}I1.csv",
}

league = st.sidebar.selectbox("League", LEAGUES.keys())
df, m_res, m_o25, m_gg, feats = train_models(LEAGUES[league])

# --------------------------------------------------
# UI
# --------------------------------------------------
teams = sorted(df.HomeTeam.unique())
c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("🏠 Home", teams)
    h_abs = st.slider("Home absences", 0, 5, 0)
with c2:
    away = st.selectbox("🚩 Away", teams)
    a_abs = st.slider("Away absences", 0, 5, 0)

if st.button("🚀 RUN AI ANALYSIS"):
    row = df[df.HomeTeam == home].iloc[-1]
    X = row[feats].values.reshape(1,-1)

    p_res = m_res.predict_proba(X)[0]
    p_o25 = m_o25.predict_proba(X)[0][1]
    p_gg = m_gg.predict_proba(X)[0][1]

    # Absence adjustment
    adj = 0.08
    home_p = max(0, min(1, p_res[2] - h_abs*adj + a_abs*adj))
    away_p = max(0, min(1, p_res[0] - a_abs*adj + h_abs*adj))
    draw_p = max(0, 1 - home_p - away_p)

    # DISPLAY
    st.subheader("🏆 Match Result")
    a,b,c = st.columns(3)
    a.metric(f"{home}", f"{home_p*100:.1f}%")
    b.metric("Draw", f"{draw_p*100:.1f}%")
    c.metric(f"{away}", f"{away_p*100:.1f}%")

    st.subheader("⚽ Goals")
    st.success(f"Over 2.5: {p_o25*100:.1f}%")
    st.info(f"BTTS: {p_gg*100:.1f}%")

    # FINAL VERDICT
    st.divider()
    outcome = home if home_p>away_p and home_p>draw_p else away if away_p>draw_p else "Draw"
    confidence = max(home_p,away_p,draw_p)
    st.warning(f"🧠 **AI PICK:** {outcome} ({confidence*100:.1f}%)")

    # INTEL
    st.subheader("🗞️ Live Intel")
    for n in get_intel(home):
        st.caption(f"{home}: {n['body'][:120]}")
    for n in get_intel(away):
        st.caption(f"{away}: {n['body'][:120]}")
