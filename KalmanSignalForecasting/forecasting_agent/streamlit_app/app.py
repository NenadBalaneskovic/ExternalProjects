import streamlit as st
import sys
import os
from agent.agent_runner import create_agent

# Add the project root to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Initialize agent once


@st.cache_resource
def load_agent():
    return create_agent()


agent = load_agent()

# Streamlit UI
st.set_page_config(page_title="Forecasting Agent", layout="centered")
st.title("📈 Hybrid Forecasting Agent")
st.markdown("Ask me to forecast any asset using natural language!")

# Prompt input
user_prompt = st.text_input(
    "Enter your forecasting prompt:",
    placeholder="e.g. Forecast Asset_2 with quantum denoising")

if st.button("Run Forecast") and user_prompt:
    with st.spinner("Running LangChain agent..."):
        response = agent.run(user_prompt)
    st.success("✅ Agent Response:")
    st.write(response)

    # Try to infer asset name from prompt
    import re
    match = re.search(r"Asset_\d+", user_prompt)
    if match:
        asset_name = match.group(0)
        plot_path = f"forecast_results/{asset_name}_forecast.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption=f"{asset_name} Forecast Plot",
                     use_column_width=True)
        else:
            st.warning(
                "Plot not found — check if the forecast ran successfully.")
