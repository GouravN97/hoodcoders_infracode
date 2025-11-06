import streamlit as st
import json
import os

def main():

    st.set_page_config(page_title="Project Details", page_icon="📄", layout="centered")

    # =============== CSS Styling ===============
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family:'Poppins'; }
        .form-heading { font-size: 2.8rem; font-weight: 800; color: white; text-align:center; margin-bottom:20px; }
        </style>
        <div class="form-heading">📋 Project Details</div>
        """,
        unsafe_allow_html=True
    )

    # ===================== MODEL SELECTION =====================
    st.markdown("<h3 style='text-align:center;color:white;'>Select Prediction Model</h3>", unsafe_allow_html=True)

    model_map = {
        "Actual_Cost": "actual_cost_model",
        "Delay_Months": "delay_months_model",
        "ROI_Realized": "realized_ROI_model",
        "Priority_Category": "priority_category_model"
    }

    col1, col2, col3, col4 = st.columns(4)

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Actual_Cost"

    with col1:
        if st.button("Actual_Cost"): st.session_state.selected_model = "Actual_Cost"
    with col2:
        if st.button("Delay_Months"): st.session_state.selected_model = "Delay_Months"
    with col3:
        if st.button("ROI_Realized"): st.session_state.selected_model = "ROI_Realized"
    with col4:
        if st.button("Priority_Category"): st.session_state.selected_model = "Priority_Category"

    st.markdown(
        f"<p style='text-align:center;color:white;font-weight:600;'>Selected: {st.session_state.selected_model}</p>",
        unsafe_allow_html=True
    )

    # ===================== FORM =====================
    with st.form("project_form"):

        sector = st.selectbox("Sector", ["Public Work","Transportation","Education","Water","Energy"])
        region = st.selectbox("Region", ["Central","West","South","East","North"])
        owner_agency = st.multiselect("Owner Agency", ["Municipal","State","Central"])

        start_year = st.number_input("Start Year (YYYY)", min_value=1900, max_value=2100)
        end_year   = st.number_input("End Year (YYYY)", min_value=1900, max_value=2100)

        planned_budget = st.number_input("Planned Budget (₹)", min_value=0)
        funding_approved = st.number_input("Funding Approved (₹)", min_value=0)
        funding_received = st.number_input("Funding Received (₹)", min_value=0)

        inflation_rate = st.number_input("Inflation Rate", min_value=0.0, max_value=1.0, value=0.05)
        inflation_index = st.number_input("Inflation Index", min_value=1.0, value=1.05)

        feasibility = st.slider("Feasibility (%)", 0, 100, 80)
        sustainability = st.slider("Sustainability (%)", 0, 100, 75)
        public_benefit = st.slider("Public Benefit Score", 0, 100, 70)
        stakeholder_priority = st.number_input("Stakeholder Priority", min_value=1, max_value=4)

        funding_delay = st.number_input("Funding Delay (%)", min_value=0.0, max_value=100.0, value=5.0)

        submitted = st.form_submit_button("Submit")

    # ===================== BACKEND CALL =====================
    if submitted:

        # ---- Build RAW INPUT PAYLOAD ----
        input_payload = {
            "Sector": sector,
            "Region": region,
            "Owner_Agency": owner_agency[0] if owner_agency else "Unknown",
            "Start_Year": start_year,
            "End_Year": end_year,
            "Planned_Budget": planned_budget,
            "Funding_Approved": funding_approved,
            "Funding_Received": funding_received,
            "Inflation_Rate": inflation_rate,
            "Inflation_Index": inflation_index,
            "Feasibility_Score": feasibility,
            "Sustainability_Score": sustainability,
            "Public_Benefit_Score": public_benefit,
            "Stakeholder_Priority": stakeholder_priority,
            "Funding_Delay_%": funding_delay
        }

        st.json(input_payload)

        # Determine folder name
        folder = model_map[st.session_state.selected_model]

        # Auto-build file paths
        model_path = f"models/{folder}/{folder}.json"
        encoders_path = f"models/{folder}/encoders.pkl"
        columns_path = f"models/{folder}/model_columns.pkl"

        from explanation_layer import explain_record

        with st.spinner("Running model & generating explanations..."):
            try:
                result = explain_record(
                    model_path=model_path,
                    encoders_path=encoders_path,
                    columns_path=columns_path,
                    input_json=json.dumps(input_payload),
                    output_dir=f"./explanations/{folder}",
                    save_plots=True
                )
                st.success("✅ Explanation generated successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                raise

        # ---- Display results ----
        st.subheader("🔮 Prediction Result")
        st.write(result.get("prediction"))

        if "plots" in result:
            if "waterfall" in result["plots"]:
                st.image(result["plots"]["waterfall"], caption="SHAP Waterfall Plot")

            if "summary" in result["plots"]:
                st.image(result["plots"]["summary"], caption="SHAP Summary Plot")


if __name__ == "__main__":
    main()
