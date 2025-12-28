import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Match Master Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor: Full League Access")

# --- 1. OPTIMIZED ENGINE ---

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
        
        # Rolling averages (Last 5 games)
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(
                lambda x: x.rolling(5, closed='left').mean().fillna(x.mean())
            )
        
        feats = ["FTHG_roll", "FTAG_roll"]
        m_res = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_res"])
        m_goal = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_o25"])
        m_gg = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_gg"])
        return df, m_res, m_goal, m_gg, feats
    except:
        return None, None, None, None, None

# --- 2. COMPLETE DATA REPOSITORY ---
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


# --- 3. UI DASHBOARD ---
sel_league = st.sidebar.selectbox("Select Competition", list(league_urls.keys()))
data, rf_res, rf_goal, rf_gg, predictors = load_and_train(league_urls[sel_league])

if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
    with c2:
        away = st.selectbox("🚩 Away Team", teams, index=min(1, len(teams)-1))

    if st.button("🚀 RUN AI PREDICTION"):
        h_row = data[data["HomeTeam"] == home].iloc[-1]
        a_row = data[data["HomeTeam"] == away].iloc[-1]
        
        p_res = rf_res.predict_proba([[h_row[p] for p in predictors]])[0]
        p_o25 = rf_goal.predict_proba([[h_row[p] for p in predictors]])[0][1]
        p_gg = rf_gg.predict_proba([[h_row[p] for p in predictors]])[0][1]

        # --- CORRECTED FORM-BASED HOME ADVANTAGE ---
        a_form = a_row["FTHG_roll"]  # Scored goals in last 5
        
        # If Away Team in Top Form (>1.8 goals/avg), nullify standard home weight
        ha_weight = 0.12 if a_form < 1.8 else 0.02
        
        f_h = max(0, min(1, p_res[2] - ha_weight))
        f_a = max(0, min(1, p_res[0] + (ha_weight if a_form > 1.8 else 0)))
        f_d = max(0, 1 - f_h - f_a)

        # RESULTS DISPLAY
        st.divider()
        if a_form > 1.8:
            st.info(f"🔥 **Elite Form Detected for {away}:** Home advantage has been neutralized.")
            
        st.subheader("📊 Match Probabilities")
        r1, r2, r3 = st.columns(3)
        r1.metric(f"{home} Win", f"{f_h*100:.1f}%")
        r2.metric("Draw", f"{f_d*100:.1f}%")
        r3.metric(f"{away} Win", f"{f_a*100:.1f}%")

        st.subheader("⚽ Goal Expectations")
        g1, g2 = st.columns(2)
        g1.success(f"Over 2.5 Goals: {p_o25*100:.1f}%")
        g2.info(f"Both Teams to Score: {p_gg*100:.1f}%")

        # FINAL AI VERDICT
        st.divider()
        st.header("🧠 AI Final Verdict")
        top_prob = max(f_h, f_a, f_d)
        outcome = home if top_prob == f_h else away if top_prob == f_a else "Draw"
        st.warning(f"**Main Selection:** {outcome} ({'High' if top_prob > 0.6 else 'Moderate'} Confidence)")
else:
    st.error("Select a league from the sidebar to begin.")
    
