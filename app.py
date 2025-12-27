import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Global AI Football Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Global AI Match Predictor")
st.markdown("Select a league and input match details for AI-backed Result & Goal predictions.")

# --- DATA SELECTION (Aligned with your tablet file names) ---
league_files = {
    "English Premier League": "epl.csv",
    "Belgium Pro League": "BL.csv",
    "Spanish La Liga": "SL.csv",
    "France Ligue 1": "FL.csv",
    "Portuguese League": "PL.csv",
    "Scottish League": "ScL.csv",
    "German Bundesliga": "GL.csv",
    "Netherlands Eredivisie": "NL.csv",
    "Turkish Super Lig": "T1.csv"
}

league_choice = st.sidebar.selectbox("Choose League", list(league_files.keys()))
st.sidebar.write(f"Looking for: {league_files[league_choice]}")

@st.cache_data
def load_and_train_league(file_path):
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # 1. Clean Column Names (Removes hidden spaces)
        df.columns = df.columns.str.strip()
        
        # 2. Date and Empty Row Cleanup
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
        df = df.dropna(subset=["Date", "FTR", "HomeTeam", "AwayTeam"]).sort_values("Date")
        
        if df.empty:
            st.error(f"No valid data found in {file_path}. Check for scores in the file.")
            return None, None, None

        # 3. Target Creation
        df["target_res"] = df.apply(lambda r: 2 if r['FTR']=='H' else (1 if r['FTR']=='D' else 0), axis=1)
        df["target_o25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        
        # 4. Feature Engineering (Rolling averages of last 4 games)
        cols = ["FTHG", "FTAG", "HST", "AST", "HC", "AC"]
        for col in cols:
            # Check if column exists before rolling (prevents crashes)
            if col in df.columns:
                df[f"{col}_roll"] = df.groupby("HomeTeam")[col].transform(lambda x: x.rolling(4, closed='left').mean())
            else:
                df[f"{col}_roll"] = 0 # Default to 0 if data is missing
        
        final_df = df.dropna(subset=["HST_roll", "AST_roll"])
        predictors = ["HST_roll", "AST_roll", "HC_roll", "AC_roll"]
        
        # 5. Train Models
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
        h_inj = st.slider(f"Key Home Players OUT ({home})", 0, 5, 0)
    with col2:
        away = st.selectbox("🚩 Away Team", teams)
        a_inj = st.slider(f"Key Away Players OUT ({away})", 0, 5, 0)

    if st.button("🚀 GENERATE PREDICTION"):
        try:
            h_latest = data[data["HomeTeam"] == home].iloc[-1]
            a_latest = data[data["AwayTeam"] == away].iloc[-1]
            
            # Injury Adjustment Logic (-20 Elo effect per player)
            # (In this simple version, it reduces the probability of a win)
            
            # Inputs for AI
            input_data = [[h_latest["HST_roll"], a_latest["AST_roll"], h_latest["HC_roll"], a_latest["AC_roll"]]]
            
            # Predictions
            p_res = rf_res.predict_proba(input_data)[0]
            p_o25 = rf_goals.predict_proba(input_data)[0][1]

            # Adjust probabilities based on injuries
            h_win_final = max(0, p_res[2] - (h_inj * 0.05) + (a_inj * 0.05))
            
            # Results Display
            st.divider()
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Match Outcome")
                st.write(f"🏠 **{home} Win:** {h_win_final*100:.1f}%")
                st.write(f"🤝 **Draw:** {p_res[1]*100:.1f}%")
                st.write(f"🚩 **{away} Win:** {(1 - h_win_final - p_res[1])*100:.1f}%")
                
            with res_col2:
                st.subheader("Goal Market")
                st.write(f"⚽ **Over 2.5 Goals:** {p_o25*100:.1f}%")
                st.write(f"🛡️ **Under 2.5 Goals:** {(1-p_o25)*100:.1f}%")

            # --- FINAL VERDICT ---
            if h_win_final > 0.65:
                st.success(f"**AI VERDICT:** 🟢 PLAY {home} WIN")
            elif p_o25 > 0.65:
                st.success(f"**AI VERDICT:** ⚽ PLAY OVER 2.5 GOALS")
            elif h_win_final < 0.35 and p_res[0] > 0.40:
                st.warning(f"**AI VERDICT:** 🚩 PLAY {away} DOUBLE CHANCE")
            else:
                st.info("**AI VERDICT:** ⚪ AVOID - High Uncertainty")
        except:
            st.error("Not enough recent data for these specific teams. Try another matchup.")
            

