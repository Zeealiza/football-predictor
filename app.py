import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from duckduckgo_search import DDGS
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Top League Predictor", page_icon="⚽", layout="wide")
st.title("⚽ AI Pro Match Predictor (Top European Leagues)")

# --- UTILITY FUNCTIONS ---
def get_team_intel(team_name, search_type="news"):
    """Live News & Lineup Search"""
    try:
        with DDGS() as ddgs:
            query = f"{team_name} confirmed starting lineup today" if search_type == "lineup" else f"{team_name} team news injuries suspensions"
            return list(ddgs.text(query, max_results=2))
    except: return []

def calculate_h2h_boost(df, home, away):
    """The 'Bogey Team' logic we decided on"""
    h2h = df[((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) | 
             ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))].tail(3)
    if len(h2h) < 1: return 0
    home_wins = len(h2h[((h2h['HomeTeam'] == home) & (h2h['FTR'] == 'H')) | 
                       ((h2h['AwayTeam'] == home) & (h2h['FTR'] == 'A'))])
    return 0.10 if home_wins >= 2 else (-0.10 if (len(h2h)-home_wins) >= 2 else 0)

# --- VERIFIED LINKS (Mapped to 2025/26 Folders) ---
# Note: Some 'Extra' leagues like Austria/Switzerland/Greece use a different base URL
BASE_MAIN = "https://www.football-data.co.uk/mmz4281/2526/"
BASE_EXTRA = "https://www.football-data.co.uk/new_leagues/" # For Austria, Greece, etc.

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
    "🇹🇷 Turkey: Süper Lig": f"{BASE_EXTRA}TUR.csv",
    "🇬🇷 Greece: Super League": f"{BASE_EXTRA}GREECE.csv",
    "🇦🇹 Austria: Bundesliga": f"{BASE_EXTRA}AUT.csv",
    "🇨🇭 Switzerland: Super League": f"{BASE_EXTRA}SWZ.csv",
    "🇩🇰 Denmark: Superliga": f"{BASE_EXTRA}DNK.csv"
}

league_choice = st.sidebar.selectbox("Select Top League", list(league_urls.keys()))

@st.cache_data(ttl=3600)
def load_and_train(url):
    try:
        # User-Agent prevents the website from blocking the app
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
        
        # Mapping results
        df["target_res"] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        # Universal Stats (Goals) used for all leagues to avoid errors
        for s in ["FTHG", "FTAG"]:
            df[f"{s}_roll"] = df.groupby("HomeTeam")[s].transform(lambda x: x.rolling(4, closed='left').mean().fillna(x.mean()))
        
        features = ["FTHG_roll", "FTAG_roll"]
        m_res = RandomForestClassifier(n_estimators

