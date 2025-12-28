import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS

# --- 1. CORE ENGINE & CACHING ---
st.set_page_config(page_title="Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor: Win/Loss & Goals")

def get_intel(team, mode="news"):
    """Fetches real-time team news or starting lineups"""
    try:
        with DDGS() as ddgs:
            query = f"{team} starting lineup today" if mode == "lineup" else f"{team} team news injuries"
            results = list(ddgs.text(query, max_results=2))
            return results if results else [{"body": "No live updates found."}]
    except:
        return [{"body": "Intel service offline."}]

@st.cache_data(ttl=3600)
def load_and_train(url):
    """Downloads data and trains the AI models with safe parameters for cloud deployment"""
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
        
        # Binary targets for Goals and Results
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        # Rolling averages (The 'Brain' of the model)
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(
                lambda x: x.rolling(4, closed='left').mean().fillna(x.mean())
            )
        
        feats = ["FTHG_roll", "FTAG_roll"]
        # Optimized for 1GB RAM limits
        m_res = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_res"])
        m_goal = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_o25"])
        m_gg = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_gg"])
        
        return df, m_res, m_goal, m_gg, feats
    except:
        return None, None, None, None, None

# --- 2. DATA SOURCE REPOSITORY ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": f"{BASE_MAIN}E0.csv",
    "🇪🇸 La Liga": f"{BASE_MAIN}SP1.csv",
    "🇩🇪 Bundesliga": f"{BASE_MAIN}D1.csv",
    "🇮🇹 Serie A": f"{BASE_MAIN}I1.csv"
}

sel_league = st.sidebar.selectbox("Select Competition", list(LEAGUES.keys()))
data, rf_res, rf_goal, rf_gg, predictors = load_and_train(LEAGUES[sel_league])

# --- 3. UI DASHBOARD ---
if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
        h_abs = st.slider(f"Absences ({home})", 0, 5, 0)
    with c2:
        away = st.selectbox("🚩 Away Team", teams, index=min(1, len(teams)-1))
        a_abs = st.slider(f"Absences ({away})", 0, 5, 0)

    if st.button("🚀 GENERATE PREDICTION REPORT"):
        # Model Logic
        h_row = data[data["HomeTeam"] == home].iloc[-1]
        p_res = rf_res.predict_proba([[h_row[p] for p in predictors]])[0]
        p_o25 = rf_goal.predict_proba([[h_row[p] for p in predictors]])[0][1]
        p_gg = rf_gg.predict_proba([[h_row[p] for p in predictors]])[0][1]

        # Absence Adjustment (0.08 shift per player)
        f_h = max(0, min(1, p_res[2] - (h_abs * 0.08) + (a_abs * 0.08)))
        f_a = max(0, min(1, p_res[0] - (a_abs * 0.08) + (h_abs * 0.08)))
        f_d = max(0, 1 - f_h - f_a)

        # RESULTS DISPLAY
        st.divider()
        st.subheader("📊 Win Probability")
        r1, r2, r3 = st.columns(3)
        r1.metric(home, f"{f_h*100:.1f}%")
        r2.metric("Draw", f"{f_d*100:.1f}%")
        r3.metric(away, f"{f_a*100:.1f}%")

        st.subheader("⚽ Goal Intensity")
        g1, g2 = st.columns(2)
        g1.success(f"Over 2.5 Goals: {p_o25*100:.1f}%")
        g2.info(f"Both Teams to Score: {p_gg*100:.1f}%")

        # --- FINAL AI VERDICT ---
        st.divider()
        st.header("🧠 AI Final Verdict")
        top_prob = max(f_h, f_a, f_d)
        outcome = home if top_prob == f_h else away if top_prob == f_a else "Draw"
        conf = "High" if top_prob > 0.6 else "Moderate"
        
        st.warning(f"**Pick:** {outcome} ({conf} Confidence)")
        st.markdown(f"**Strategic Insight:** Expect {'high' if p_o25 > 0.6 else 'low'} goal volume.")

        # LIVE INTEL
        st.subheader("🗞️ Real-Time Intelligence")
        n1, n2 = st.columns(2)
        with n1:
            for n in get_intel(home): st.caption(f"**{home}:** {n['body'][:140]}...")
        with n2:
            for n in get_intel(away): st.caption(f"**{away}:** {n['body'][:140]}...")
else:
    st.error("Data could not be loaded. Please check your internet connection.")
    
