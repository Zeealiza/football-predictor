import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Top League Predictor", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor (Top Leagues Only)")

# --- UTILITY FUNCTIONS ---
def get_team_intel(team_name, search_type="news"):
    """Fetches news or lineup info using DDGS."""
    try:
        with DDGS() as ddgs:
            if search_type == "lineup":
                query = f"{team_name} confirmed starting lineup today"
            else:
                query = f"{team_name} team news injuries suspensions"
            results = ddgs.text(query, max_results=3)
            return list(results)
    except:
        return []

def calculate_h2h_boost(df, home, away):
    """Detects 'Bogey Team' patterns from the last 3 meetings."""
    h2h = df[((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) | 
             ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))].tail(3)
    if len(h2h) < 1: return 0
    home_wins = len(h2h[((h2h['HomeTeam'] == home) & (h2h['FTR'] == 'H')) | 
                       ((h2h['AwayTeam'] == home) & (h2h['FTR'] == 'A'))])
    # Returns 10% swing if one team dominates the fixture
    if home_wins >= 2: return 0.10
    if (len(h2h) - home_wins) >= 2: return -0.10
    return 0

# --- TOP LEAGUES REPOSITORY (2025/2026 Season) ---
league_urls = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England: Premier League": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇪🇸 Spain: La Liga": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇩🇪 Germany: Bundesliga": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "🇮🇹 Italy: Serie A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇫🇷 France: Ligue 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "🇳🇱 Netherlands: Eredivisie": "https://www.football-data.co.uk/mmz4281/2526/N1.csv",
    "🇧🇪 Belgium: Pro League": "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
    "🇵🇹 Portugal: Liga Portugal": "https://www.football-data.co.uk/mmz4281/2526/P1.csv",
    "🇹🇷 Turkey: Süper Lig": "https://www.football-data.co.uk/mmz4281/2526/T1.csv",
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland: Premiership": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    "🇬🇷 Greece: Super League": "https://www.football-data.co.uk/mmz4281/2526/G1.csv",
    "🇦🇹 Austria: Bundesliga": "https://www.football-data.co.uk/mmz4281/2526/AUT.csv",
    "🇨🇭 Switzerland: Super League": "https://www.football-data.co.uk/mmz4281/2526/SWZ.csv",
    "🇩🇰 Denmark: Superliga": "https://www.football-data.co.uk/mmz4281/2526/DNK.csv"
}

league_choice = st.sidebar.selectbox("Select Top League", list(league_urls.keys()))

@st.cache_data(ttl=3600)
def load_and_train(url):
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"]).sort_values("Date")
        
        # Binary Targets
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        # Stats & Rolling Averages
        stats = ["FTHG", "FTAG", "HST", "AST", "HC", "AC"]
        avail = [s for s in stats if s in df.columns]
        for s in avail:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(4, closed='left').mean())
            df[f"{s}_roll"] = df[f"{s}_roll"].fillna(df[s].mean())
        
        features = [f"{s}_roll" for s in avail]
        m_res = RandomForestClassifier(n_estimators=100).fit(df[features], df["target_res"])
        m_goal = RandomForestClassifier(n_estimators=100).fit(df[features], df["target_o25"])
        m_gg = RandomForestClassifier(n_estimators=100).fit(df[features], df["target_gg"])
        return df, m_res, m_goal, m_gg, features
    except Exception as e:
        st.error(f"Error loading league: {e}"); return None, None, None, None, None

data, rf_res, rf_goals, rf_gg, predictors = load_and_train(league_urls[league_choice])

if data is not None:
    st.sidebar.info(f"Database contains {len(data)} results.")
    st.sidebar.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
    
    teams = sorted(data["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
        h_news = get_team_intel(home)
        for n in h_news: st.caption(f"📰 {n['body'][:90]}...")
        h_inj = st.slider(f"Key Absences ({home})", 0, 5, 0, key="h_inj_final")
        
    with c2:
        away = st.selectbox("🚩 Away Team", teams, index=1 if len(teams)>1 else 0)
        a_news = get_team_intel(away)
        for n in a_news: st.caption(f"📰 {n['body'][:90]}...")
        a_inj = st.slider(f"Key Absences ({away})", 0, 5, 0, key="a_inj_final")

    # --- LINEUP CHECKER ---
    st.divider()
    if st.checkbox("🔍 Check for Confirmed Lineups"):
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            h_lineup = get_team_intel(home, "lineup")
            if h_lineup: st.info(f"**{home} Lineup Info:** {h_lineup[0]['body'][:150]}...")
            else: st.warning(f"No lineup confirmed for {home} yet.")
        with l_col2:
            a_lineup = get_team_intel(away, "lineup")
            if a_lineup: st.info(f"**{away} Lineup Info:** {a_lineup[0]['body'][:150]}...")
            else: st.warning(f"No lineup confirmed for {away} yet.")

    # --- PREDICTION ENGINE ---
    if st.button("🚀 PREDICT NOW") and home != away:
        h_dat = data[data["HomeTeam"] == home].iloc[-1]
        feats = [[h_dat[p] for p in predictors]]
        
        # Probabilities
        raw_res = rf_res.predict_proba(feats)[0]
        p_o25 = rf_goals.predict_proba(feats)[0][1]
        p_gg = rf_gg.predict_proba(feats)[0][1]

        # 10% Injury Impact & H2H Swing
        h2h = calculate_h2h_boost(data, home, away)
        h_win = max(0, min(1, raw_res[2] - (h_inj * 0.10) + (a_inj * 0.10) + h2h))
        a_win = max(0, min(1, raw_res[0] - (a_inj * 0.10) + (h_inj * 0.10) - h2h))
        draw = max(0, 1 - h_win - a_win)

        st.divider()
        res1, res2, res3 = st.columns(3)
        res1.metric(f"🏠 {home} Win", f"{h_win*100:.1f}%")
        res1.write(f"🤝 Draw: {draw*100:.1f}% | 🚩 {away}: {a_win*100:.1f}%")
        res2.metric("Over 2.5 Goals", f"{p_o25*100:.1f}%")
        res3.metric("Both Teams Score (GG)", f"{p_gg*100:.1f}%")
        
        st.divider()
        if h_win > 0.60: st.success(f"AI VERDICT: Back {home} to Win")
        elif a_win > 0.60: st.success(f"AI VERDICT: Back {away} to Win")
        elif p_gg > 0.65: st.success("AI VERDICT: Strong Value in GG (BTTS)")
        elif p_o25 > 0.65: st.success("AI VERDICT: High Chance of Over 2.5 Goals")
        else: st.info("AI VERDICT: Market is uncertain. Suggest Double Chance.")

