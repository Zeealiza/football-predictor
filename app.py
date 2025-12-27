import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Global AI Football Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Global AI Match Predictor")
st.markdown("Live AI-backed predictions for Results, Goals, and GG market.")

# --- AUTOMATIC DATA FEED (2025/2026 Season) ---
league_files = {
    "English Premier League": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "Spanish La Liga": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "German Bundesliga": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "Italian Serie A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "France Ligue 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "Belgium Pro League": "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
    "Netherlands Eredivisie": "https://www.football-data.co.uk/mmz4281/2526/N1.csv",
    "Portuguese League": "https://www.football-data.co.uk/mmz4281/2526/P1.csv",
    "Scottish League": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    "Turkish Super Lig": "https://www.football-data.co.uk/mmz4281/2526/T1.csv"
}

league_choice = st.sidebar.selectbox("Choose League", list(league_files.keys()))

@st.cache_data
def load_and_train_league(url):
    try:
        # Load from URL with browser-like header to prevent blocking
        df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        df.columns = df.columns.str.strip()
        
        # 1. Clean Names & Dates
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"]).sort_values("Date")
        
        # 2. Target Creation (Result, Over 2.5, and GG)
        df["target_res"] = df.apply(lambda r: 2 if r['FTR']=='H' else (1 if r['FTR']=='D' else 0), axis=1)
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        # 3. Protector: Identify available stats
        essential_stats = ["FTHG", "FTAG", "HST", "AST", "HC", "AC"]
        available_stats = [col for col in essential_stats if col in df.columns]
        
        # 4. Feature Engineering (Rolling averages)
        for col in available_stats:
            df[f"{col}_roll"] = df.groupby("HomeTeam")[col].transform(lambda x: x.rolling(4, closed='left').mean())
            # Fill missing (new teams) with league average
            df[f"{col}_roll"] = df[f"{col}_roll"].fillna(df[col].mean())
        
        predictors = [f"{col}_roll" for col in available_stats]
        
        # 5. Train Models
        model_res = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[predictors], df["target_res"])
        model_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[predictors], df["target_o25"])
        model_gg = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[predictors], df["target_gg"])
        
        return df, model_res, model_goals, model_gg, predictors
    except Exception as e:
        st.error(f"⚠️ Data Load Error: {e}")
        return None, None, None, None, None

# --- LOAD ACTIVE LEAGUE ---
data, rf_res, rf_goals, rf_gg, predictors = load_and_train_league(league_files[league_choice])

if data is not None:
    # Sidebar Live Feed Preview
    st.sidebar.divider()
    st.sidebar.subheader("📅 Recent Database Results")
    st.sidebar.table(data[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].tail(3))
    
    # Team Selection
    teams = sorted(data["HomeTeam"].unique())
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Home Team", teams)
        h_inj = st.slider(f"Key Home Absences ({home})", 0, 5, 0)
    with col2:
        away = st.selectbox("🚩 Away Team", teams)
        a_inj = st.slider(f"Key Away Absences ({away})", 0, 5, 0)

    if st.button("🚀 GENERATE PREDICTION"):
        try:
            h_latest = data[data["HomeTeam"] == home].iloc[-1]
            a_latest = data[data["AwayTeam"] == away].iloc[-1]
            
            # Prepare Input Data
            input_feats = [[h_latest[p] for p in predictors]]
            
            # Get Probabilities
            p_res = rf_res.predict_proba(input_feats)[0]
            p_o25 = rf_goals.predict_proba(input_feats)[0][1]
            p_gg = rf_gg.predict_proba(input_feats)[0][1]

            # Adjust Home Win based on Injuries
            h_win_final = max(0, min(1, p_res[2] - (h_inj * 0.05) + (a_inj * 0.05)))
            
            # Results UI
            st.divider()
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.subheader("Match Outcome")
                st.write(f"🏠 **{home}:** {h_win_final*100:.1f}%")
                st.write(f"🤝 **Draw:** {p_res[1]*100:.1f}%")
                st.write(f"🚩 **{away}:** {(1 - h_win_final - p_res[1])*100:.1f}%")
                
            with res_col2:
                st.subheader("Goal Market")
                st.write(f"⚽ **Over 2.5:** {p_o25*100:.1f}%")
                st.write(f"🛡️ **Under 2.5:** {(1-p_o25)*100:.1f}%")

            with res_col3:
                st.subheader("GG Market")
                st.write(f"🔥 **GG (Yes):** {p_gg*100:.1f}%")
                st.write(f"🔒 **NG (No):** {(1-p_gg)*100:.1f}%")

            # --- FINAL VERDICT ---
            st.divider()
            if h_win_final > 0.65:
                st.success(f"**RECOMMENDATION:** 🟢 {home} WIN")
            elif p_o25 > 0.65:
                st.success(f"**RECOMMENDATION:** ⚽ OVER 2.5 GOALS")
            elif p_gg > 0.65:
                st.success(f"**RECOMMENDATION:** 🔥 BOTH TEAMS TO SCORE (GG)")
            else:
                st.info("**RECOMMENDATION:** ⚪ NO CLEAR VALUE")
        except Exception:
            st.error("Error processing this matchup. Please try different teams.")
        
