import streamlit as st
import json
import base64


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    st.set_page_config(page_title="Project Dashboard", page_icon="🚀", layout="centered")
    if st.session_state.get("page", None) == "second":
        import page
        page.main()
    else:
        # Get base64 encoded image from project directory
        bg_image = get_base64_image("background.png")
        
        # Check if animating
        is_animating = st.session_state.get("animating", False)
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bg_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            .main-heading {{
                font-size: 2.8rem;
                font-weight: 800;
                letter-spacing: 2px;
                text-align: center;
                margin-top: 80px;
                margin-bottom: 30px;
                color: #ffffff;
                text-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
            }}
            
            .letsgo-btn {{
                display: flex;
                justify-content: center;
                margin-top: 60px;
            }}
            
            .letsgo-btn button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 14px 48px;
                font-size: 1.1rem;
                font-weight: 700;
                letter-spacing: 1px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }}
            
            .letsgo-btn button:hover {{
                transform: translateY(-4px) scale(1.05);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.7);
                animation: pulse 1.5s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ box-shadow: 0 8px 25px rgba(102, 126, 234, 0.7); }}
                50% {{ box-shadow: 0 8px 35px rgba(102, 126, 234, 1); }}
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
            
            st.markdown('<div class="letsgo-btn">', unsafe_allow_html=True)
            if st.button("Let's Go", key="letsgo"):
                # Set animating state
                st.session_state.animating = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()