import streamlit as st

def inject_global_css():
    """
    Injects all necessary global CSS styles, including card layouts and
    centered content, into the Streamlit application.
    """
    st.markdown("""
<style>
/* ------------------------------------------------ */
/* GLOBAL RESET AND CORE STYLES */
/* ------------------------------------------------ */

/* Center the main content block for a cleaner look */
.main-content {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    padding-top: 50px;
}

/* Ensure that content inside columns is centered by default */
.st-emotion-cache-12fmw35 { /* Target the Streamlit column wrapper for centered content */
    display: flex;
    justify-content: center;
    gap: 20px; /* Space between columns */
}

/* ------------------------------------------------ */
/* CARD STYLES FOR LANDING PAGE */
/* ------------------------------------------------ */

.selection-card {
    background-color: #f0f2f6; /* Light gray background */
    padding: 20px;
    margin: 15px 10px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease-in-out;
    cursor: pointer;
    text-align: center;
    min-height: 200px; /* Increased height for better visual balance */
    display: flex;
    flex-direction: column;
    justify-content: space-around; /* Distribute content nicely */
}

.selection-card:hover {
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
    background-color: #e0e2e6; /* Slightly darker on hover */
    transform: translateY(-2px);
}

.card-title {
    font-size: 1.25em;
    font-weight: 700;
    color: #1f77b4; /* Streamlit blue */
    margin-bottom: 5px;
}

.card-icon {
    font-size: 2.8em; /* Slightly larger icon */
    margin-bottom: 10px;
}

.card-description {
    color: #555;
    font-size: 0.9em;
}
.centered-text {
    text-align: center;
    font-size: 80px;
    color: #53AB23;
}

/* Hide the mandatory Streamlit buttons used to trigger the HTML click event */
div[data-testid="stColumn"] > div > .stButton {
    visibility: hidden;
    height: 0px;
    margin: -10px; /* Negative margin to truly hide it without displacing content */
}
</style>
""", unsafe_allow_html=True)
