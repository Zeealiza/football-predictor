import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. CORE AI ENGINE ---
@st.cache_data(ttl=3600)
def load_and_train(url):
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
        
        # Binary Targets
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        
        # Rolling averages (The AI's Performance Source of Truth)
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(
                lambda x: x.rolling(5, closed='left').mean().fillna(x.mean())
            )
        
        feats = ["FTHG_roll", "FTAG_roll"]
        m_res = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_res"])
        m_goal = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df[feats], df["target_o25"])
        return df, m_res, m_goal, feats
    except:
        return None, None, None, None

# --- 2. DATA REPOSITORY ---
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
league_urls = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": f"{BASE_MAIN}E0.csv",
    "🇪🇸 La Liga": f"{BASE_MAIN}SP1.csv",
    "🇩🇪 Bundesliga": f"{BASE_MAIN}D1.csv",
    "🇮🇹 Serie A": f"{BASE_MAIN}I1.csv"
}

sel_league = st.sidebar.selectbox("Competition", list(league_urls.keys()))
data, rf_res, rf_goal, predictors = load_and_train(league_urls[sel_league])

# --- 3. UI & MANUAL INPUTS ---
if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
        h_odd = st.number_input(f"SportyBet Odd ({home})", min_value=1.01, value=2.00)
    with c2:
        away = st.selectbox("🚩 Away Team", teams, index=min(1, len(teams)-1))
        a_odd = st.number_input(f"SportyBet Odd ({away})", min_value=1.01, value=2.00)

    if st.button("🚀 ANALYZE PERFORMANCE VS MARKET"):
        # AI Performance Calculation
        h_row = data[data["HomeTeam"] == home].iloc[-1]
        a_row = data[data["HomeTeam"] == away].iloc[-1]
        p_res = rf_res.predict_proba([[h_row[p] for p in predictors]])[0]
        
        # Form Neutralizer
        a_form = a_row["FTHG_roll"]
        ha_weight = 0.12 if a_form < 1.8 else 0.02
        f_h = max(0, min(1, p_res[2] - ha_weight))
        f_a = max(0, min(1, p_res[0] + (ha_weight if a_form > 1.8 else 0)))
        f_d = 1 - f_h - f_a

        # Market Implied Probabilities
        market_h_prob = 1 / h_odd
        market_a_prob = 1 / a_odd

        # --- CONFLICT DETECTION LOGIC ---
        # Detect if SportyBet odds strongly contradict AI performance data
        conflict_detected = False
        verdict_color = "warning"
        if (f_a > 0.6 and h_odd < 1.70): # AI says Away wins, but Market favors Home
            conflict_detected = True
            verdict_msg = "⚠️ MARKET TRAP: SportyBet odds favor Home, but Performance data suggests an Away dominant force."
        elif (f_h > 0.6 and a_odd < 1.70): # AI says Home wins, but Market favors Away
            conflict_detected = True
            verdict_msg = "⚠️ MARKET TRAP: SportyBet odds favor Away, but Performance data shows Home is superior."
        else:
            verdict_msg = "✅ Market and AI are in alignment."

        # DISPLAY
        st.divider()
        st.subheader("📊 Performance Logic vs. Market Odds")
        r1, r2 = st.columns(2)
        r1.metric("AI Score (Form)", f"{max(f_h, f_a)*100:.1f}%")
        r2.metric("Market Sentiment", "Strong" if min(h_odd, a_odd) < 1.5 else "Neutral")

        # FINAL VERDICT
        st.divider()
        st.header("🧠 Informed AI Verdict")
        if conflict_detected:
            st.error(verdict_msg)
            st.info("Recommendation: Trust performance over odds. The bookmaker may be overvaluing reputation over current form.")
        else:
            st.success(verdict_msg)
            
        outcome = home if f_h > f_a else away
        st.warning(f"**Performance Pick:** {outcome}")

else:
    st.error("Select a league to load performance data.")

