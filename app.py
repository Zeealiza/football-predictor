import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor: Win/Loss & Goals")

# --- 1. DEFINE FUNCTIONS (Must be at the top) ---

def get_intel(team, mode="news"):
    """Fetches real-time team news or starting lineups"""
    try:
        with DDGS() as ddgs:
            # Query varies based on the mode requested
            query = f"{team} starting lineup today" if mode == "lineup" else f"{team} team news injuries"
            results = list(ddgs.text(query, max_results=2))
            return results if results else [{"body": "No recent live updates found."}]
    except Exception:
        return [{"body": "Live intel service temporarily unavailable."}]

@st.cache_data(ttl=3600)
def load_and_train(url):
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
        
        # Targets
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(4, closed='left').mean().fillna(x.mean()))
        
        feats = ["FTHG_roll", "FTAG_roll"]
        m_res = RandomForestClassifier(n_estimators=100).fit(df[feats], df["target_res"])
        m_goal = RandomForestClassifier(n_estimators=100).fit(df[feats], df["target_o25"])
        m_gg = RandomForestClassifier(n_estimators=100).fit(df[feats], df["target_gg"])
        return df, m_res, m_goal, m_gg, feats
    except:
        return None, None, None, None, None

# --- 2. DATA REPOSITORY ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
BASE_EXTRA = "https://www.football-data.co.uk/new_leagues/"

league_urls = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England: Premier League": f"{BASE_MAIN}E0.csv",
    "🇪🇸 Spain: La Liga": f"{BASE_MAIN}SP1.csv",
    "🇩🇪 Germany: Bundesliga": f"{BASE_MAIN}D1.csv",
    "🇮🇹 Italy: Serie A": f"{BASE_MAIN}I1.csv",
    "🇫🇷 France: Ligue 1": f"{BASE_MAIN}F1.csv",
    "🇳🇱 Netherlands: Eredivisie": f"{BASE_MAIN}N1.csv",
    "🇧🇪 Belgium: Pro League": f"{BASE_MAIN}B1.csv",
    "🇵🇹 Portugal: Liga Portugal": f"{BASE_MAIN}P1.csv",
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland: Premiership": f"{BASE_MAIN}SC0.csv",
    "🇦🇹 Austria: Bundesliga": f"{BASE_EXTRA}AUT.csv",
    "🇨🇭 Switzerland: Super League": f"{BASE_EXTRA}SWZ.csv",
    "🇬🇷 Greece: Super League": f"{BASE_EXTRA}GREECE.csv",
    "🇹🇷 Turkey: Süper Lig": f"{BASE_EXTRA}TUR.csv",
    "🇩🇰 Denmark: Superliga": f"{BASE_EXTRA}DNK.csv"
}

# --- 3. UI LAYOUT ---
sel_league = st.sidebar.selectbox("Competition", list(league_urls.keys()))
data, rf_res, rf_goal, rf_gg, predictors = load_and_train(league_urls[sel_league])

if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
        h_abs = st.slider(f"Absences ({home})", 0, 5, 0)
    with c2:
        away = st.selectbox("🚩 Away Team", teams, index=1 if len(teams)>1 else 0)
        a_abs = st.slider(f"Absences ({away})", 0, 5, 0)

    if st.button("🚀 RUN FULL PREDICTION"):
        # 1. AI Logic
        h_row = data[data["HomeTeam"] == home].iloc[-1]
        p_res = rf_res.predict_proba([[h_row[p] for p in predictors]])[0]
        p_o25 = rf_goal.predict_proba([[h_row[p] for p in predictors]])[0][1]
        p_gg = rf_gg.predict_proba([[h_row[p] for p in predictors]])[0][1]

        # 2. Live Intel (Now defined and called inside button)
        with st.spinner("Analyzing news and lineups..."):
            h_news = get_intel(home)
            a_news = get_intel(away)

        # 3. Adjust for Absences
        f_h = max(0, min(1, p_res[2] - (h_abs * 0.10) + (a_abs * 0.10)))
        f_a = max(0, min(1, p_res[0] - (a_abs * 0.10) + (h_abs * 0.10)))
        f_d = max(0, 1 - f_h - f_a)

        # --- TABLET DISPLAY ---
        st.divider()
        st.subheader("🏆 Match Winner AI")
        res_cols = st.columns(3)
        res_cols[0].metric(f"{home} Win", f"{f_h*100:.1f}%")
        res_cols[1].metric("Draw", f"{f_d*100:.1f}%")
        res_cols[2].metric(f"{away} Win", f"{f_a*100:.1f}%")

        st.subheader("⚽ Goals & BTTS")
        g_col1, g_col2 = st.columns(2)
        g_col1.success(f"**Over 2.5 Goals:** {p_o25*100:.1f}%")
        g_col2.info(f"**BTTS (Both Teams to Score):** {p_gg*100:.1f}%")

        # --- FINAL AI VERDICT BLOCK (NEW) ---
        st.divider()
        st.header("🧠 AI Final Verdict")
        
        # Determination of result and confidence
        top_prob = max(f_h, f_a, f_d)
        if top_prob == f_h:
            outcome = home
            conf_level = "High" if f_h > 0.60 else "Moderate"
            advice = f"Strong statistical bias for a {home} victory."
        elif top_prob == f_a:
            outcome = away
            conf_level = "High" if f_a > 0.60 else "Moderate"
            advice = f"Statistical advantage lies with {away}."
        else:
            outcome = "Draw"
            conf_level = "Moderate"
            advice = "High parity between teams suggests a stalemate."

        # Goal Context
        goal_advice = "Expect a high-scoring encounter." if p_o25 > 0.65 else "Anticipate a tight, low-scoring game."
        
        # Display Verdict
        st.warning(f"**Main Prediction:** {outcome} ({conf_level} Confidence)")
        st.markdown(f"**Strategic Insight:** {advice} {goal_advice}")

        st.subheader("🗞️ Real-Time Intelligence")
        n_col1, n_col2 = st.columns(2)
        with n_col1:
            st.write(f"**{home} News:**")
            for n in h_news: st.caption(f"• {n['body'][:120]}...")
        with n_col2:
            st.write(f"**{away} News:**")
            for n in a_news: st.caption(f"• {n['body'][:120]}...")
        
