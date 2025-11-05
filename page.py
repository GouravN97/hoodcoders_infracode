import streamlit as st

def main():
    st.set_page_config(page_title="Project Details", page_icon="📄", layout="centered")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        .form-heading {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: 2px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
            color: #ffffff;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            animation: fadeInDown 0.8s ease;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .stForm {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(10px);
            padding: 40px 48px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            animation: fadeInUp 0.8s ease;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .stSelectbox label, .stMultiSelect label, .stTextInput label, .stNumberInput label, .stSlider label {
            color: #1e3c72 !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
        }
        
        div[data-baseweb="select"] > div {
            background-color: #f8fafc;
            border-radius: 12px;
            border: 2px solid #e0e7ef;
            transition: all 0.3s ease;
        }
        
        div[data-baseweb="select"] > div:hover {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .stTextInput input, .stNumberInput input {
            background-color: #f8fafc;
            border-radius: 12px;
            border: 2px solid #e0e7ef;
            padding: 12px 16px;
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .stSlider {
            padding: 10px 0;
        }
        
        button[kind="formSubmit"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 14px 48px !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            width: 100% !important;
            margin-top: 20px !important;
        }
        
        button[kind="formSubmit"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }
        
        .stSuccess {
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid #28a745;
            padding: 16px;
            border-radius: 8px;
        }
        </style>
        <div class="form-heading">📋 Project Details</div>
        """,
        unsafe_allow_html=True
    )
    with st.form("project_form"):
        sector = st.selectbox(
            "Sector",
            ["Public Work", "Transportation", "Education", "Water", "Energy"],
            help="The domain or industry category of the project, such as transportation, education, water, energy, or public works. It determines the project’s nature, regulatory framework, and potential social impact."
        )
        region = st.selectbox(
            "Region",
            ["Central", "West", "South", "East", "North"],
            help="The geographical area where the project is implemented. Regional context affects environmental regulations and cost factors."
        )
        owner_agency = st.multiselect(
            "Owner Agency",
            ["Municipal", "State", "Central"],
            help="The administrative level that owns, sponsors, or manages the project: Central, State, or Municipal. Central: Projects owned or sponsored by national-level agencies or ministries. State: Projects owned by state governments or state departments. Municipal: Projects owned by city/municipal corporations or local bodies. This field indicates the governance tier responsible for approvals, budget allocation, procurement rules, and escalation paths — all of which materially influence funding speed, compliance requirements, and project oversight."
        )
        start_year = st.text_input(
            "Start Year (YYYY)",
            help="Used for calculating project duration, inflation impact, and scheduling performance."
        )
        planned_budget = st.number_input(
            "Planned Budget (in ₹)", min_value=0, step=10000,
            help="The total financial allocation estimated during project planning. Represents the baseline against which actual expenditure, overruns, and ROI are compared."
        )
        funding_approved = st.number_input(
            "Funding Approved (in ₹)", min_value=0, step=10000,
            help="The amount of funding officially sanctioned by the financing authority or agency. This may differ from the planned budget due to policy constraints, prioritization, or resource availability."
        )
        funding_received = st.number_input(
            "Funding Received (in ₹)", min_value=0, step=10000,
            help="The actual funds disbursed and made available for project execution. It directly affects work progress and overall schedule adherence."
        )
        feasibility = st.slider(
            "Feasibility (%)", 0, 100, 80,
            help="Indicates how confident the owner agency is about a project’s success based on their internal evaluation of technical, financial, and policy factors. It helps gauge how realistic and well-prepared the project plan is before execution."
        )
        sustainability = st.slider(
            "Sustainability (%)", 0, 100, 75,
            help="Reflects how well the project can maintain its benefits over time: environmentally, socially, and financially. It shows whether the project’s design and execution are built for long-term impact rather than short-term completion."
        )
        stakeholder_priority = st.number_input(
            "Stakeholder Priority (integer)", min_value=0, step=1,
            help="Priority assigned by stakeholders, used for ranking or resource allocation."
        )
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.success("Project details submitted!")

if __name__ == "__main__":
    main()
