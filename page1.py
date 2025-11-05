import streamlit as st

def main():
    st.set_page_config(page_title="Project Dashboard", page_icon="🚀", layout="centered")
    if st.session_state.get("page", None) == "second":
        import page
        page.main()
    else:
        st.markdown(
            """
            <style>
            .main-heading {
                font-size: 2.8rem;
                font-weight: 800;
                letter-spacing: 2px;
                text-align: center;
                margin-top: 80px;
                margin-bottom: 30px;
                color: #1e3c72;
                text-shadow: 0 2px 12px #b0c4de;
            }
            .letsgo-btn {
                display: flex;
                justify-content: center;
                margin-top: 60px;
            }
            </style>
        <div class="main-heading">🚀 What If</div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="letsgo-btn">', unsafe_allow_html=True)
        if st.button("Let's Go", key="letsgo"):
            st.session_state.page = "second"
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
