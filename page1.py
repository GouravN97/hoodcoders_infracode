import streamlit as st
import json
import base64


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    st.set_page_config(page_title="Project Dashboard", page_icon="🚀", layout="centered")
    
    # Check session state for page navigation
    current_page = st.session_state.get("page", None)
    
    if current_page == "second":
        import page
        page.main()
    else:
        # Landing page - show only if NOT going to page 2
        # Get base64 encoded image from project directory
        bg_image = get_base64_image("background.png")
        
        # Check if animating
        is_animating = st.session_state.get("animating", False)
        
        st.markdown(
            f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800;900&display=swap');
            
            .stApp {{
                background-image: url("data:image/png;base64,{bg_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                font-family: 'Poppins', sans-serif;
            }}
            
            /* Hide default Streamlit elements */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}
            
            /* Center container with glass effect */
            .main .block-container {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(15px);
                border-radius: 30px;
                padding: 4rem 3rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.2);
                max-width: 800px;
                margin: auto;
                margin-top: 10vh;
            }}
            
            .main-heading {{
                font-size: 7rem;
                font-weight: 900;
                letter-spacing: 12px;
                text-align: center;
                margin-top: 20px;
                margin-bottom: 60px;
                color: #ffffff;
                text-shadow: 0 6px 30px rgba(0, 0, 0, 0.9), 0 0 40px rgba(102, 200, 255, 0.5);
                font-family: 'Copperplate', 'Copperplate Gothic Light', serif;
                animation: glow 3s ease-in-out infinite;
            }}
            
            @keyframes glow {{
                0%, 100% {{ 
                    text-shadow: 0 6px 30px rgba(0, 0, 0, 0.9), 0 0 40px rgba(102, 200, 255, 0.5);
                }}
                50% {{ 
                    text-shadow: 0 6px 30px rgba(0, 0, 0, 0.9), 0 0 60px rgba(102, 200, 255, 0.8);
                }}
            }}
            
            .subtitle {{
                font-size: 1.8rem;
                font-weight: 500;
                text-align: center;
                color: rgba(255, 255, 255, 0.95);
                margin-bottom: 80px;
                letter-spacing: 3px;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.7);
            }}
            
            /* Button container and styling */
            .stButton {{
                display: flex;
                justify-content: center;
                margin-top: 50px;
            }}
            
            .stButton > button {{
                background: rgba(255, 255, 255, 0.15) !important;
                color: white !important;
                border: 2px solid rgba(255, 255, 255, 0.4) !important;
                border-radius: 15px !important;
                padding: 28px 0px !important;
                font-size: 2.2rem !important;
                font-weight: 900 !important;
                letter-spacing: 6px !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 0 25px rgba(102, 200, 255, 0.8), 0 0 50px rgba(102, 200, 255, 0.5) !important;
                text-transform: uppercase !important;
                width: 100% !important;
                min-height: 90px !important;
            }}
            
            .stButton > button:hover {{
                transform: translateY(-5px) scale(1.03) !important;
                box-shadow: 0 0 35px rgba(102, 200, 255, 1), 0 0 70px rgba(102, 200, 255, 0.7) !important;
                background: rgba(255, 255, 255, 0.2) !important;
                border-color: rgba(255, 255, 255, 0.6) !important;
            }}
            
            .stButton > button:active {{
                transform: translateY(-2px) scale(1.01) !important;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ 
                    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.8);
                }}
                50% {{ 
                    box-shadow: 0 12px 50px rgba(102, 126, 234, 1), 0 0 30px rgba(118, 75, 162, 0.8);
                }}
            }}
            
            @keyframes zoomSpin {{
                0% {{
                    transform: scale(1) rotate(0deg);
                    filter: brightness(1);
                }}
                100% {{
                    transform: scale(2) rotate(360deg);
                    filter: brightness(1.5);
                }}
            }}
            
            .animation-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-image: url("data:image/png;base64,{bg_image}");
                background-size: cover;
                background-position: center;
                z-index: 9999;
                animation: zoomSpin 2s ease-in-out forwards;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Show animation overlay when animating
        if is_animating:
            st.markdown('<div class="animation-overlay"></div>', unsafe_allow_html=True)
            import time
            time.sleep(2)
            st.session_state.page = "second"
            st.session_state.animating = False
            st.rerun()
        else:
            st.markdown('<h1 class="main-heading">What If?</h1>', unsafe_allow_html=True)
            st.markdown('<p class="subtitle">Explore Infrastructure Predictions</p>', unsafe_allow_html=True)
            
            if st.button("Start Now", key="letsgo", use_container_width=True):
                # Set animating state
                st.session_state.animating = True
                st.rerun()

if __name__ == "__main__":
    main()