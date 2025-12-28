import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from duckduckgo_search import DDGS

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Match Master Pro – XGBoost Edition")

# --------------------------------------------------
# LIVE INTEL
# --------------------------------------------------
def get_intel(team):
    try:
        with DDGS() as ddgs:
            q = f"{team} injuries lineup today"
            r = list(ddgs.text(q, max_results=2))
            return r if r else [{"body": "No recent updates"}]
    except:
        return [{"body": "Intel unavailable"}]

# --------------------------------------------------
# CONFIDENCE GRADING
# --------------------------------------------------
def grade_confidence(prob):
    if prob >= 0.70:
        return "A+"
    elif prob >= 0.63:
        return "A"
    elif prob >= 0.57:
        return "B"
    else:
        return "C"

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
        table[t] = points / played if played else 0

    return pd.Series(table, name="ppg")

def add_features(df):
    df["Date"] = pd.to_datetime(df["Date"])

    # Rolling goals
    for g in ["FTHG", "FTAG"]:
        df[f"{g}_roll"] = df.groupby("HomeTeam")[g] \
            .transform(lambda x: x.rolling(5, closed="left").mean().fillna(x.mean()))

    # Form
    pts = {"H":3, "D":1, "A":0}
    df["home_form"] = df.groupby("HomeTeam")["FTR"] \
        .transform(lambda x: x.map(pts).rolling(5).mean())
    df["away_form"] = df.groupby("AwayTeam")["FTR"] \
        .transform(lambda x: x.map(pts).rolling(5).mean())
    df["form_diff"] = df["home_form"] - df["away_form"]

    # Fatigue
    df["home_rest"] = df.groupby("HomeTeam")["Date"].diff().dt.days.fillna(7)
    df["away_rest"] = df.groupby("AwayTeam")["Date"].diff().dt.days.fillna(7)
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # Table strength
    ppg = build_table(df)
    df["home_ppg"] = df["HomeTeam"].map(ppg)
    df["away_ppg"] = df["AwayTeam"].map(ppg)
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

    df.fillna(0, inplace=True)
    return df

# --------------------------------------------------
# TRAIN MODELS (XGBOOST)
# --------------------------------------------------
@st.cache_data(ttl=3600)
def train_models(url):
    df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
    df = add_features(df)

    df["res_target"] = df.FTR.map({"H":2, "D":1, "A":0})
    df["o25"] = ((df.FTHG + df.FTAG) > 2.5).astype(int)
    df["btts"] = ((df.FTHG > 0) & (df.FTAG > 0)).astype(int)

    features = [
        "FTHG_roll", "FTAG_roll",
        "ppg_diff", "form_diff", "rest_diff"
    ]

    model_res = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )

    model_o25 = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.04,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    model_gg = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.04,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    X = df[features]
    model_res.fit(X, df["res_target"])
    model_o25.fit(X, df["o25"])
    model_gg.fit(X, df["btts"])

    return df, model_res, model_o25, model_gg, features

# --------------------------------------------------
# ACCUMULATOR ENGINE
# --------------------------------------------------
def extract_pick(home, away, probs, p_o25, p_gg):
    home_p, draw_p, away_p = probs[2], probs[1], probs[0]
    best_prob = max(home_p, draw_p, away_p)
    grade = grade_confidence(best_prob)

    if best_prob == home_p:
        market = f"{home} Win"
    elif best_prob == away_p:
        market = f"{away} Win"
    else:
        market = "Draw"

    return {
        "match": f"{home} vs {away}",
        "market": market,
        "prob": best_prob,
        "grade": grade
    }

def generate_accumulators(df, model_res, feats):
    picks = []

    for _, row in df.iterrows():
        X = row[feats].values.reshape(1, -1)
        probs = model_res.predict_proba(X)[0]

        pick = extract_pick(
            row["HomeTeam"],
            row["AwayTeam"],
            probs,
            None,
            None
        )

        if pick["grade"] != "C":
            picks.append(pick)

    picks = sorted(picks, key=lambda x: x["prob"], reverse=True)

    return {
        "safe": picks[:3],
        "balanced": picks[:5],
        "aggressive": picks[:8]
    }

# --------------------------------------------------
# DATA SOURCES
# --------------------------------------------------
BASE = "https://www.football-data.co.uk/mmz4281/2526/"
LEAGUES = {
    "Premier League": f"{BASE}E0.csv",
    "La Liga": f"{BASE}SP1.csv",
    "Bundesliga": f"{BASE}D1.csv",
    "Serie A": f"{BASE}I1.csv"
}

league = st.sidebar.selectbox("League", LEAGUES.keys())
df, m_res, m_o25, m_gg, feats = train_models(LEAGUES[league])

# --------------------------------------------------
# SINGLE MATCH UI
# --------------------------------------------------
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
    Xp = row[feats].values.reshape(1, -1)

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

# --------------------------------------------------
# ACCUMULATOR UI
# --------------------------------------------------
st.divider()
st.header("🎯 AI Accumulator Generator")

if st.button("📊 Generate Accumulators"):
    accas = generate_accumulators(df.tail(20), m_res, feats)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🟢 SAFE ACCA")
        for p in accas["safe"]:
            st.write(f"✔ {p['match']}")
            st.caption(f"{p['market']} — {p['prob']*100:.1f}% ({p['grade']})")

    with col2:
        st.warning("🟡 BALANCED ACCA")
        for p in accas["balanced"]:
            st.write(f"✔ {p['match']}")
            st.caption(f"{p['market']} — {p['prob']*100:.1f}% ({p['grade']})")

    with col3:
        st.error("🔴 AGGRESSIVE ACCA")
        for p in accas["aggressive"]:
            st.write(f"✔ {p['match']}")
            st.caption(f"{p['market']} — {p['prob']*100:.1f}% ({p['grade']})")
