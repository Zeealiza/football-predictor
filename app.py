import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Pro Football Predictor", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor (H2H + Injury Weighted)")

# --- UTILITY FUNCTIONS ---
def get_team_news(team_name):
    try:
        with DDGS() as ddgs:
            query = f"{team_name} football team injury news suspensions"
            results = [r for r in ddgs.text(query, max_results=3)]
            return results
    except: return []

def calculate_h2h_boost(df, home, away):
    h2h = df[((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) | 
             ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))].tail(3)
    if len(h2h) < 1: return 0
    home_wins = len(h2h[((h2h['HomeTeam'] == home) & (h2h['FTR'] == 'H')) | 
                       ((h2h['AwayTeam'] == home) & (h2h['FTR'] == 'A'))])
    return 0.10 if home_wins >= 2 else (-0.10 if len(h2h) - home_wins >= 2 else 0)

def get_defensive_leak(df, team):
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(4)
    leaks = len(recent[((recent['HomeTeam'] == team) & (recent['FTAG'] > 0)) | 
                       ((recent['AwayTeam'] == team) & (recent['FTHG'] > 0))])
    return leaks * 0.05 

# --- DATA FEED ---
league_files = {
    "English Premier League": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "Spanish La Liga": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "German Bundesliga": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "Italian Serie A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "France Ligue 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "Belgium Pro League": "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
    "Scottish League": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv"
}

league_choice = st.sidebar.selectbox("Choose League", list(league_files.keys()))

@st.cache_data
def load_and_train(url):
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"]).sort_values("Date")
        df["target_res"] = df.apply(lambda r: 2 if r['FTR']=='H' else (1 if r['FTR']=='D' else 0), axis=1)
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        essential = ["FTHG", "FTAG", "HST", "AST", "HC", "AC"]
        available = [c for c in essential if c in df.columns]
        for c in available:
            df[f"{c}_roll"] = df.groupby("HomeTeam")[c].transform(lambda x: x.rolling(4, closed='left').mean())
            df[f"{c}_roll"] = df[f"{c}_roll"].fillna(df[c].mean())
        preds = [f"{c}_roll" for c in available]
        m_res = RandomForestClassifier(n_estimators=100).fit(df[preds], df["target_res"])
        m_goals = RandomForestClassifier(n_estimators=100).fit(df[preds], df["target_o25"])
        m_gg = RandomForestClassifier(n_estimators=100).fit(df[preds], df["target_gg"])
        return df, m_res, m_goals, m_gg, preds
    except Exception as e:
        st.error(f"Error: {e}"); return None, None, None, None, None

data, rf_res, rf_goals, rf_gg, predictors = load_and_train(league_files[league_choice])

if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    col1, col2 = st.columns(2)
    
    with col1:
        home = st.selectbox("🏠 Home Team", teams, index=0)
        # Unique news fetch
        for n in get_team_news(home): st.caption(f"📰 {n['title'][:80]}...")
        # FIX: Added unique 'key' to avoid Duplicate Element ID
        h_inj = st.slider(f"Key Absences ({home})", 0, 5, 0, key="home_slider")
        
    with col2:
        # Default Away to second team to avoid crash on start
        away = st.selectbox("🚩 Away Team", teams, index=1 if len(teams) > 1 else 0)
        for n in get_team_news(away): st.caption(f"📰 {n['title'][:80]}...")
        # FIX: Added unique 'key' to avoid Duplicate Element ID
        a_inj = st.slider(f"Key Absences ({away})", 0, 5, 0, key="away_slider")

    if home == away:
        st.warning("⚠️ Please select two different teams for a valid prediction.")

    if st.button("🚀 GENERATE PREDICTION") and home != away:
        h_latest = data[data["HomeTeam"] == home].iloc[-1]
        a_latest = data[data["AwayTeam"] == away].iloc[-1]
        input_feats = [[h_latest[p] for p in predictors]]
        
        p_res = rf_res.predict_proba(input_feats)[0]
        p_o25 = rf_goals.predict_proba(input_feats)[0][1]
        p_gg = rf_gg.predict_proba(input_feats)[0][1]

        h2h_swing = calculate_h2h_boost(data, home, away)
        h_leak = get_defensive_leak(data, home)
        a_leak = get_defensive_leak(data, away)

        # 10% penalty per player
        h_win_adj = max(0, min(1, p_res[2] - (h_inj * 0.10) + (a_inj * 0.10) + h2h_swing))
        a_win_adj = max(0, min(1, p_res[0] - (a_inj * 0.10) + (h_inj * 0.10) - h2h_swing))
        draw_adj = max(0, 1 - h_win_adj - a_win_adj)
        
        gg_final = max(0, min(1, p_gg + h_leak + a_leak))

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric(f"🏠 {home} Win", f"{h_win_adj*100:.1f}%")
        r1.write(f"🚩 {away} Win: {a_win_adj*100:.1f}%")
        r2.metric("⚽ Over 2.5 Goals", f"{p_o25*100:.1f}%")
        r3.metric("🔥 GG (BTTS)", f"{gg_final*100:.1f}%")

        st.divider()
        if h_win_adj > 0.60: st.success(f"TIP: {home} Win")
        elif a_win_adj > 0.60: st.success(f"TIP: {away} Win")
        elif gg_final > 0.70: st.success("TIP: Both Teams to Score (GG)")
        else: st.info("TIP: No clear value found.")

