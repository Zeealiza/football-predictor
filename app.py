import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Football Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Global AI Match Predictor")
st.markdown("Select a league and input match details to get AI-backed predictions for Results and Goals.")

# --- DATA SELECTION (Aligned with your tablet file names) ---
league_files = {
    "English Premier League": "epl.csv",
    "Belgium Pro League": "BL.csv",
    "Spanish La Liga": "SL.csv",
    "Italian Serie A": "seriea.csv",
    "France Ligue 1": "FL.csv",
    "Portuguese League": "PL.csv",
    "Scottish League": "ScL.csv",
    "German Bundesliga": "GL.csv",
    "Netherlands Eredivisie": "NL.csv",
    "Turkish Super Lig": "T1.csv"
}

league_choice = st.sidebar.selectbox("Choose League", list(league_files.keys()))

@st.cache_data
def load_and_train_league(file_path):
    try:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
        df = df.dropna(subset=["Date", "FTR"]).sort_values("Date")
        
        # Target 1: Match Result (2=H, 1=D, 0=A)
        df["target_res"] = df.apply(lambda r: 2 if r['FTR']=='H' else (1 if r['FTR']=='D' else 0), axis=1)
        # Target 2: Over 2.5 Goals (1=Yes, 0=No)
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        
        # Feature Engineering (Rolling averages)
        cols = ["FTHG", "FTAG", "HST", "AST", "HC", "AC"]
        for col in cols:
            df[f"{col}_roll"] = df.groupby("HomeTeam")[col].transform(lambda x: x.rolling(4, closed='left').mean())
        
        final_df = df.dropna()
        predictors = ["HST_roll", "AST_roll", "HC_roll", "AC_roll"]
        
        # Train Models
        model_res = RandomForestClassifier(n_estimators=100, random_state=42).fit(final_df[predictors], final_df["target_res"])
        model_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(final_df[predictors], final_df["target_o25"])
        
        return final_df, model_res, model_goals
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None, None, None

# --- LOAD ACTIVE LEAGUE ---
data, rf_res, rf_goals = load_and_train_league(league_files[league_choice])

if data is not None:
    teams = sorted(data["HomeTeam"].unique())
    
    # --- INTERFACE ---
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Home Team", teams)
        h_inj = st.slider(f"Key Home Absences ({home})", 0, 5, 0)
    with col2:
        away = st.selectbox("🚩 Away Team", teams)
        a_inj = st.slider(f"Key Away Absences ({away})", 0, 5, 0)

    if st.button("🚀 GENERATE PREDICTION"):
        h_latest = data[data["HomeTeam"] == home].iloc[-1]
        a_latest = data[data["HomeTeam"] == away].iloc[-1] # Simple proxy for away form
        
        # Inputs for AI
        input_data = [[h_latest["HST_roll"], a_latest["AST_roll"], h_latest["HC_roll"], a_latest["AC_roll"]]]
        
        # Predictions
        p_res = rf_res.predict_proba(input_data)[0]
        p_o25 = rf_goals.predict_proba(input_data)[0][1]

        # Results Display
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader("Match Outcome")
            st.write(f"🏠 **{home} Win:** {p_res[2]*100:.1f}%")
            st.write(f"🤝 **Draw:** {p_res[1]*100:.1f}%")
            st.write(f"🚩 **{away} Win:** {p_res[0]*100:.1f}%")
            
        with res_col2:
            st.subheader("Goal Market")
            st.write(f"⚽ **Over 2.5 Goals:** {p_o25*100:.1f}%")
            st.write(f"🛡️ **Under 2.5 Goals:** {(1-p_o25)*100:.1f}%")

        # --- FINAL VERDICT ---
        st.info(f"**AI Recommendation:** " + 
                ("PLAY HOME WIN" if p_res[2] > 0.65 else "PLAY OVER 2.5" if p_o25 > 0.65 else "AVOID - NO CLEAR VALUE"))

                
