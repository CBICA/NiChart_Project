import streamlit as st

def top_nav(pages, current_page):
    """
    pages: list of page names in order (e.g., ["Home", "Upload", "Process", "Results"])
    current_page: the title of the current page (string)
    """
    st.markdown("""
        <style>
        .nav-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            padding: 0.5rem;
            background-color: #1e3a8a;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .nav-button {
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 0.4rem 1rem;
            border-radius: 0.3rem;
            font-weight: 500;
            cursor: pointer;
        }
        .nav-button:disabled {
            opacity: 0.5;
            cursor: default;
        }
        </style>
    """, unsafe_allow_html=True)

    idx = pages.index(current_page)
    prev_page = pages[idx - 1] if idx > 0 else None
    next_page = pages[idx + 1] if idx < len(pages) - 1 else None

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if prev_page:
            if st.button("⬅️ Prev", key=f"prev_{current_page}"):
                st.session_state["page"] = prev_page
                st.experimental_rerun()
        else:
            st.button("⬅️ Prev", disabled=True, key=f"prev_{current_page}")
    with col2:
        if st.button("🏠 Home", key=f"home_{current_page}"):
            st.session_state["page"] = "Home"
            st.experimental_rerun()
    with col3:
        if next_page:
            if st.button("Next ➡️", key=f"next_{current_page}"):
                st.session_state["page"] = next_page
                st.experimental_rerun()
        else:
            st.button("Next ➡️", disabled=True, key=f"next_{current_page}")
