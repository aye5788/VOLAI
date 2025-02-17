import streamlit as st
import openai
import requests
import pandas as pd
import datetime

# Set page configuration
st.set_page_config(page_title="ORATS Options Analysis Dashboard", layout="wide")

st.title("ORATS Options Analysis Dashboard")

# ------------------------------
# 1. Configuration & Secrets
# ------------------------------
ORATS_TOKEN = st.secrets["ORATS_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# ------------------------------
# 2. User Input for Ticker
# ------------------------------
ticker = st.text_input("Enter ticker (e.g., AAPL):", value="AAPL").strip().upper()

if ticker:
    # ------------------------------
    # 3. Fetch Summaries (ORATS)
    # ------------------------------
    summaries_url = f"https://api.orats.io/datav2/summaries?token={ORATS_TOKEN}&ticker={ticker}"
    resp_summaries = requests.get(summaries_url).json()
    df_summaries = pd.DataFrame(resp_summaries.get("data", []))
    
    if df_summaries.empty:
        st.error("No summary data returned from ORATS. Check your ticker or API token.")
    else:
        if "stockPrice" in df_summaries.columns:
            current_price = float(df_summaries.loc[0, "stockPrice"])
            st.write(f"**Current {ticker} price (ORATS Summaries):** {current_price:.2f}")
        else:
            current_price = None
            st.write("No 'stockPrice' in summaries.")
        
        st.write("--- **Summaries Data** ---")
        st.dataframe(df_summaries)
    
        # ------------------------------
        # 4. Fetch Strikes (ORATS)
        # ------------------------------
        strikes_url = f"https://api.orats.io/datav2/strikes?token={ORATS_TOKEN}&ticker={ticker}"
        resp_strikes = requests.get(strikes_url).json()
        df_strikes = pd.DataFrame(resp_strikes.get("data", []))
    
        # Standardize expiration field
        if "expiration" in df_strikes.columns:
            df_strikes["expiration"] = pd.to_datetime(df_strikes["expiration"]).dt.date
        elif "expirDate" in df_strikes.columns:
            df_strikes["expiration"] = pd.to_datetime(df_strikes["expirDate"]).dt.date
        else:
            st.error("Strikes data missing 'expiration' or 'expirDate'.")
    
        today = datetime.date.today()
        two_months = today + datetime.timedelta(days=60)
        df_two_months = df_strikes[
            (df_strikes["expiration"] >= today) & (df_strikes["expiration"] <= two_months)
        ]
    
        df_atm_grouped = pd.DataFrame()
        if not df_two_months.empty and current_price:
            atm_threshold = 0.05 * current_price
            df_atm = df_two_months[
                (df_two_months["strike"] - current_price).abs() <= atm_threshold
            ]
            if df_atm.empty:
                df_atm = df_two_months.copy()
    
            def closest_atm(group):
                idx = (group["strike"] - current_price).abs().idxmin()
                return group.loc[idx]
    
            df_atm_grouped = (
                df_atm.groupby("expiration", group_keys=False)
                .apply(closest_atm)
                .reset_index(drop=True)
            )
    
        st.write("--- **ATM Options (Next 2 Months)** ---")
        if df_atm_grouped.empty:
            st.write("No ATM options found within two months.")
        else:
            st.dataframe(df_atm_grouped[["expiration", "strike", "callMidIv", "putMidIv", "delta", "gamma", "theta", "vega"]])
    
        # ------------------------------
        # 5. Fetch Core Data (ORATS)
        # ------------------------------
        fields = (
            "ticker,tradeDate,priorCls,pxAtmIv,contango,atmIvM1,atmFitIvM1,atmFcstIvM1,dtExM1,"
            "atmIvM2,atmFitIvM2,atmFcstIvM2,dtExM2,slope,deriv"
        )
        cores_url = f"https://api.orats.io/datav2/cores?token={ORATS_TOKEN}&ticker={ticker}&fields={fields}"
        resp_cores = requests.get(cores_url).json()
        df_cores = pd.DataFrame(resp_cores.get("data", []))
    
        st.write("--- **ORATS Core Data** ---")
        if df_cores.empty:
            st.write("No core data returned.")
        else:
            st.dataframe(df_cores)
    
        # ------------------------------
        # 6. Build the Prompt for ChatCompletion
        # ------------------------------
        # Convert dataframes to text.
        summaries_text = df_summaries.to_string(index=False)
        atm_text = df_atm_grouped.to_string(index=False) if not df_atm_grouped.empty else "No ATM data"
        cores_text = df_cores.to_string(index=False) if not df_cores.empty else "No Core Data"
    
        prompt_user = f"""
        You are an expert options trader. Below is specific ORATS data for {ticker}:
    
        1) Summaries Data:
        {summaries_text}
    
        2) ATM Options for the Next Two Months:
        {atm_text}
    
        3) Core Data:
        {cores_text}
    
        Based on these specific numbers, please provide a detailed interpretation. Instead of general educational remarks, analyze what these exact values indicate for {ticker}'s diagonal spread strategy. For example, discuss:
        - What does the current price of {current_price:.2f} and the fact that ATM options are at a strike of 245 imply?
        - How do the specific implied volatility numbers (callMidIv and putMidIv) and Greeks (delta, gamma, theta, vega) influence the strategy?
        - What do the core data values such as a contango of 0.2915, slope of 2.327968, and deriv of 0.0651 tell us about the term structure and skew?
        - Provide actionable insights based on these exact output values, including potential trade adjustments or strategy recommendations.
    
        Please be as specific as possible in your analysis using the provided data.
        """
    
        messages = [
            {"role": "system", "content": "You are a financial analyst specializing in options strategies."},
            {"role": "user", "content": prompt_user}
        ]
    
        # ------------------------------
        # 7. Call the ChatCompletion Endpoint
        # ------------------------------
        st.write("--- **AI Interpretation** ---")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",      # Using GPT-4
                messages=messages,
                temperature=0.7,
                max_tokens=600      # Reduced max_tokens for efficiency
            )
            ai_analysis = response["choices"][0]["message"]["content"]
            st.write(ai_analysis)
    
        except Exception as e:
            st.error(f"Error with OpenAI API: {e}")
    
        st.write("Analysis complete. Check above for AI interpretation of the ORATS data.")
