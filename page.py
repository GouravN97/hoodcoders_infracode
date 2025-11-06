import streamlit as st
import json
import os
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():

    st.set_page_config(page_title="Project Details", page_icon="📄", layout="centered")

    # Check if going back - show only animation
    is_going_back = st.session_state.get("going_back", False)

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
            font-size: 1rem;
        }}
        .stButton > button:hover {{
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
            transform: translateY(-2px);
        }}
        /* Primary button (selected) styling - subtle neon highlight */
        div[data-testid="stButton"] button[kind="primary"] {{
            background: rgba(255, 255, 255, 0.15) !important;
            border: 2px solid rgba(255, 255, 255, 0.4) !important;
            box-shadow: 0 0 20px rgba(102, 200, 255, 0.8), 0 0 40px rgba(102, 200, 255, 0.4) !important;
            font-weight: 700 !important;
        }}
        /* Help icon visibility */
        .stTooltipIcon {{
            color: white !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}
        /* Form labels - bigger and bolder */
        label {{
            color: white !important;
            font-weight: 900 !important;
            font-size: 1.5rem !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            font-family: 'Verdana', sans-serif !important;
            margin-bottom: 0.5rem !important;
        }}
        /* Help text styling - always visible */
        .stMarkdown p {{
            color: rgba(255, 255, 255, 0.85) !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
            font-family: 'Verdana', sans-serif !important;
            margin-top: 0.3rem !important;
            margin-bottom: 0.8rem !important;
        }}
        /* Input fields - much larger */
        .stSelectbox > div > div, .stNumberInput > div > div > input {{
            font-size: 1.2rem !important;
            padding: 0.75rem !important;
            font-weight: 600 !important;
            font-family: 'Verdana', sans-serif !important;
            color: white !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
        }}
        input, select, textarea {{
            font-size: 1.2rem !important;
            padding: 0.75rem !important;
            font-weight: 600 !important;
            font-family: 'Verdana', sans-serif !important;
            color: white !important;
        }}
        /* Radio buttons styling - white text visible */
        .stRadio > label {{
            color: white !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            font-family: 'Verdana', sans-serif !important;
        }}
        .stRadio > div {{
            gap: 1rem !important;
        }}
        .stRadio > div > label {{
            color: white !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            font-family: 'Verdana', sans-serif !important;
            background: rgba(255, 255, 255, 0.1) !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            transition: all 0.2s ease !important;
        }}
        .stRadio > div > label:hover {{
            background: rgba(255, 255, 255, 0.2) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
        }}
        .stRadio > div > label > div {{
            color: white !important;
        }}
        /* Radio button circle */
        .stRadio input[type="radio"] {{
            accent-color: #66C8FF !important;
        }}
        /* Number input values */
        .stNumberInput input {{
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            font-family: 'Verdana', sans-serif !important;
            color: white !important;
        }}
        /* Slider text */
        .stSlider p {{
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            font-family: 'Verdana', sans-serif !important;
        }}
        /* Slider styling - black/dark */
        .stSlider [role="slider"] {{
            background-color: #333333 !important;
            border: 2px solid white !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div {{
            background-color: rgba(100, 100, 100, 0.5) !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div > div {{
            background-color: #222222 !important;
        }}
        /* Back button styling */
        .back-btn {{
            margin-top: 30px;
            text-align: center;
        }}
        .back-btn button {{
            background: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 10px !important;
            padding: 10px 30px !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }}
        .back-btn button:hover {{
            background: rgba(255, 255, 255, 0.3) !important;
            transform: translateY(-2px) !important;
        }}
        /* Input section spacing */
        .input-section {{
            margin-bottom: 2rem !important;
        }}
        /* Section headings */
        h3 {{
            font-size: 2rem !important;
            font-weight: 900 !important;
        }}
        /* Hide JSON output */
        .stJson {{
            display: none !important;
        }}
        /* Animation overlay for back button - zoom out bigger */
        .back-animation-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url("data:image/png;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            z-index: 9999;
            animation: zoomOutRotate 1.5s ease-out forwards;
        }}
        @keyframes zoomOutRotate {{
            0% {{
                transform: scale(1) rotate(0deg);
                opacity: 1;
            }}
            100% {{
                transform: scale(3) rotate(-360deg);
                opacity: 0.7;
            }}
        }}
        </style>
        <div class="form-heading">Project Prediction</div>
        """,
        unsafe_allow_html=True
    )

    # Show animation overlay when going back and return early
    if is_going_back:
        st.markdown(
            f"""
            <style>
            /* Hide everything except animation overlay when going back */
            .stApp > header,
            .main .block-container {{
                display: none !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="back-animation-overlay"></div>', unsafe_allow_html=True)
        import time
        time.sleep(1.5)
        st.session_state.page = None
        st.session_state.going_back = False
        st.rerun()
        return

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
        if st.button("Actual_Cost", key="btn1", use_container_width=True, 
                    type="primary" if st.session_state.selected_model == "Actual_Cost" else "secondary"):
            st.session_state.selected_model = "Actual_Cost"
            st.rerun()
    with col2:
        if st.button("Delay_Months", key="btn2", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Delay_Months" else "secondary"):
            st.session_state.selected_model = "Delay_Months"
            st.rerun()
    with col3:
        if st.button("ROI_Realized", key="btn3", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "ROI_Realized" else "secondary"):
            st.session_state.selected_model = "ROI_Realized"
            st.rerun()
    with col4:
        if st.button("Priority_Category", key="btn4", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Priority_Category" else "secondary"):
            st.session_state.selected_model = "Priority_Category"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ===================== FORM =====================
    with st.form("project_form"):

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Sector</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Select the primary sector for this project</p>", unsafe_allow_html=True)
        sector = st.radio("Sector", ["Public Work","Transportation","Education","Water","Energy"], label_visibility="collapsed", key="sector_select", horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Region</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Geographic region where the project is located</p>", unsafe_allow_html=True)
        region = st.radio("Region", ["Central","West","South","East","North"], label_visibility="collapsed", key="region_select", horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Owner Agency</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Agency responsible for the project</p>", unsafe_allow_html=True)
        owner_agency = st.radio("Owner Agency", ["Municipal","State","Central"], label_visibility="collapsed", key="owner_select", horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Start Year (YYYY)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Year when the project is expected to begin</p>", unsafe_allow_html=True)
        start_year = st.number_input("Start Year (YYYY)", min_value=1900, max_value=2100, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Expected End Year (YYYY)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Year when the project is expected to be completed</p>", unsafe_allow_html=True)
        end_year = st.number_input("Expected End Year (YYYY)", min_value=1900, max_value=2100, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Planned Budget (₹)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Total budget planned for the project</p>", unsafe_allow_html=True)
        planned_budget = st.number_input("Planned Budget (₹)", min_value=0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Funding Approved (₹)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Amount of funding that has been approved</p>", unsafe_allow_html=True)
        funding_approved = st.number_input("Funding Approved (₹)", min_value=0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Funding Received (₹)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Actual funding received so far</p>", unsafe_allow_html=True)
        funding_received = st.number_input("Funding Received (₹)", min_value=0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Inflation Rate</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Expected annual inflation rate (0.05 = 5%)</p>", unsafe_allow_html=True)
        inflation_rate = st.number_input("Inflation Rate", min_value=0.0, max_value=1.0, value=0.05, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Inflation Index</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Current inflation index relative to base year</p>", unsafe_allow_html=True)
        inflation_index = st.number_input("Inflation Index", min_value=1.0, value=1.05, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Feasibility (%)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Technical and practical feasibility score</p>", unsafe_allow_html=True)
        feasibility = st.slider("Feasibility (%)", 0, 100, 80, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Sustainability (%)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Environmental and long-term sustainability score</p>", unsafe_allow_html=True)
        sustainability = st.slider("Sustainability (%)", 0, 100, 75, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Public Benefit Score</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Expected benefit to the public</p>", unsafe_allow_html=True)
        public_benefit = st.slider("Public Benefit Score", 0, 100, 70, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Stakeholder Priority</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Priority level assigned by stakeholders (1=highest, 4=lowest)</p>", unsafe_allow_html=True)
        stakeholder_priority = st.number_input("Stakeholder Priority", min_value=1, max_value=4, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem;color:white;font-weight:900;font-family:Verdana;margin-bottom:0.5rem;'>Funding Delay (%)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.85);margin-top:0.3rem;margin-bottom:0.8rem;'>Percentage of funding that has been delayed</p>", unsafe_allow_html=True)
        funding_delay = st.number_input("Funding Delay (%)", min_value=0.0, max_value=100.0, value=5.0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Submit")
    
    # Back button at the bottom
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_btn"):
        st.session_state.going_back = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ===================== BACKEND CALL =====================
    if submitted:

        # ---- Build RAW INPUT PAYLOAD ----
        input_payload = {
            "Sector": sector,
            "Region": region,
            "Owner_Agency": owner_agency if owner_agency else "Unknown",
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

        # JSON output hidden by CSS

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
