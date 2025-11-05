import streamlit as st

st.set_page_config(page_title="What If Dashboard", page_icon="🤖", layout="centered")

# Custom CSS for modern look
st.markdown(
    """
    <style>
    .main-heading {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 10px;
        color: #1e3c72;
        text-shadow: 0 2px 12px #b0c4de;
    }
    .sector-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
        border-radius: 18px;
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.10);
        padding: 18px 24px;
        margin-bottom: 18px;
        color: #222;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .confidence-bar {
        background: #e0e7ef;
        border-radius: 8px;
        height: 24px;
        margin-bottom: 12px;
        position: relative;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-heading">🤖 What if</div>', unsafe_allow_html=True)


# --- Sectors ---
sectors = ["Energy", "Healthcare", "Finance", "Retail", "Manufacturing", "Residential", "Commercial"]
sector = st.selectbox("Select Sector", sectors, index=0)
st.markdown(f'<div class="sector-box">Selected Sector: <b>{sector}</b></div>', unsafe_allow_html=True)

# --- AI Confidence Placeholder ---
st.markdown("**How confident is our AI?**")
confidence = 0.82  # Placeholder value
st.markdown(f'''
<div class="confidence-bar">
  <div class="confidence-fill" style="width: {confidence*100}%;"></div>
</div>
<div style="text-align:center;font-size:1.1rem;font-weight:600;">{int(confidence*100)}% confident</div>
''', unsafe_allow_html=True)

# --- Sliders for CapEx, OpEx, Demand Growth ---
st.markdown("---")
st.subheader("Adjust Scenario Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    capex_mult = st.slider("CapEx Multiplier", 0.5, 2.0, 1.0, 0.05)
with col2:
    opex_mult = st.slider("OpEx Multiplier", 0.5, 2.0, 1.0, 0.05)
with col3:
    demand_growth = st.slider("Demand Growth (%)", 0.0, 0.10, 0.03, 0.01)

# --- Predict Button ---
predict = st.button("� Predict")

# --- Chart Placeholders ---
st.markdown("---")
st.subheader("Forecast Results")
if predict:
    st.success("Prediction logic will go here! (Integrate with backend API)")
    st.line_chart([1, 2, 3, 4, 5])  # Placeholder chart
else:
    st.info("Adjust parameters and click Predict to see results.")
