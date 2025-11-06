import streamlit as st
import json
import os
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():

    st.set_page_config(page_title="Project Details", page_icon="📄", layout="centered")

    # Get base64 encoded image
    bg_image = get_base64_image("background.png")

    # =============== CSS Styling ===============
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
        .stApp {{ 
            background-image: url("data:image/png;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family:'Poppins'; 
        }}
        .form-heading {{ 
            font-size: 3.5rem; 
            font-weight: 900; 
            color: white; 
            text-align:center; 
            margin-bottom:30px;
            margin-top: 20px;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
            font-family: 'Copperplate', 'Copperplate Gothic Light', serif;
            letter-spacing: 3px;
        }}
        /* Semi-transparent container for better readability */
        .main .block-container {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        /* Model button styling */
        .stButton > button {{
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
            transform: translateY(-2px);
        }}
        /* Help icon visibility */
        .stTooltipIcon {{
            color: white !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}
        /* Form labels */
        label {{
            color: white !important;
            font-weight: 600 !important;
        }}
        </style>
        <div class="form-heading">Project Prediction</div>
        """,
        unsafe_allow_html=True
    )

    # ===================== MODEL SELECTION =====================
    st.markdown("<h3 style='text-align:center;color:white;font-weight:700;'>Select Prediction Model</h3>", unsafe_allow_html=True)

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
        btn1 = st.button("Actual_Cost", key="btn1", use_container_width=True, 
                        type="primary" if st.session_state.selected_model == "Actual_Cost" else "secondary")
        if btn1: st.session_state.selected_model = "Actual_Cost"
    with col2:
        btn2 = st.button("Delay_Months", key="btn2", use_container_width=True,
                        type="primary" if st.session_state.selected_model == "Delay_Months" else "secondary")
        if btn2: st.session_state.selected_model = "Delay_Months"
    with col3:
        btn3 = st.button("ROI_Realized", key="btn3", use_container_width=True,
                        type="primary" if st.session_state.selected_model == "ROI_Realized" else "secondary")
        if btn3: st.session_state.selected_model = "ROI_Realized"
    with col4:
        btn4 = st.button("Priority_Category", key="btn4", use_container_width=True,
                        type="primary" if st.session_state.selected_model == "Priority_Category" else "secondary")
        if btn4: st.session_state.selected_model = "Priority_Category"

    st.markdown("<br>", unsafe_allow_html=True)

    # ===================== FORM =====================
    with st.form("project_form"):

        sector = st.selectbox("Sector", ["Public Work","Transportation","Education","Water","Energy"],
                              help="Select the primary sector for this project")
        region = st.selectbox("Region", ["Central","West","South","East","North"],
                              help="Geographic region where the project is located")
        owner_agency = st.multiselect("Owner Agency", ["Municipal","State","Central"],
                                      help="Select one or more agencies responsible for the project")

        start_year = st.number_input("Start Year (YYYY)", min_value=1900, max_value=2100,
                                     help="Year when the project is expected to begin")
        end_year   = st.number_input("Expected End Year (YYYY)", min_value=1900, max_value=2100,
                                     help="Year when the project is expected to be completed")

        planned_budget = st.number_input("Planned Budget (₹)", min_value=0,
                                        help="Total budget planned for the project")
        funding_approved = st.number_input("Funding Approved (₹)", min_value=0,
                                          help="Amount of funding that has been approved")
        funding_received = st.number_input("Funding Received (₹)", min_value=0,
                                          help="Actual funding received so far")

        inflation_rate = st.number_input("Inflation Rate", min_value=0.0, max_value=1.0, value=0.05,
                                        help="Expected annual inflation rate (0.05 = 5%)")
        inflation_index = st.number_input("Inflation Index", min_value=1.0, value=1.05,
                                         help="Current inflation index relative to base year")

        feasibility = st.slider("Feasibility (%)", 0, 100, 80,
                               help="Technical and practical feasibility score")
        sustainability = st.slider("Sustainability (%)", 0, 100, 75,
                                   help="Environmental and long-term sustainability score")
        public_benefit = st.slider("Public Benefit Score", 0, 100, 70,
                                   help="Expected benefit to the public")
        stakeholder_priority = st.number_input("Stakeholder Priority", min_value=1, max_value=4,
                                              help="Priority level assigned by stakeholders (1=highest, 4=lowest)")

        funding_delay = st.number_input("Funding Delay (%)", min_value=0.0, max_value=100.0, value=5.0,
                                       help="Percentage of funding that has been delayed")

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
