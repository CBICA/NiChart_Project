import streamlit as st

st.set_page_config(page_title="NiChart", layout="centered")

# --- App title ---
st.title("🧠 NiChart")
st.write("Simplified neuroimaging workflow manager")

# --- Session state to track progress ---
if "page" not in st.session_state:
    st.session_state.page = "start"
if "choice" not in st.session_state:
    st.session_state.choice = None

# --- Navigation function ---
def go_to(page):
    st.session_state.page = page

# --- Main navigation logic ---
if st.session_state.page == "start":
    st.subheader("What type of MRI data do you have?")
    choice = st.radio(
        "Select one:",
        ["I have MRI data for a Single Subject",
         "I have MRI data for a Dataset",
         "I have no MRI data"]
    )

    if st.button("Next"):
        st.session_state.choice = choice
        go_to("workflow")

elif st.session_state.page == "workflow":
    st.button("← Back", on_click=lambda: go_to("start"))

    if st.session_state.choice == "I have MRI data for a Single Subject":
        st.header("Single Subject Workflow")
        step = st.radio("Choose a step:", ["Upload Data", "Select Pipeline", "View / Download Results"])

        if step == "Upload Data":
            uploaded_file = st.file_uploader("Upload MRI file (e.g., NIfTI format)")
            if uploaded_file:
                st.success("File uploaded successfully!")

        elif step == "Select Pipeline":
            pipeline = st.selectbox("Select analysis pipeline:", ["fMRI preprocessing", "DTI analysis", "Cortical thickness"])
            st.info(f"Selected pipeline: {pipeline}")

        elif step == "View / Download Results":
            st.write("Results will appear here once processing is complete.")
            st.download_button("Download Results", data="fake_results.csv", file_name="results.csv")

    elif st.session_state.choice == "I have MRI data for a Dataset":
        st.header("Dataset Workflow")
        st.write("Batch upload and processing coming soon!")

    elif st.session_state.choice == "I have no MRI data":
        st.header("Explore Demo Mode")
        st.write("Try demo datasets or learn about available pipelines.")
