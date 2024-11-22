import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_st as utilst
import utils.utils_session as utilss
from PIL import Image

# Page config should be called for each page
utilss.config_page()

def set_pipeline():
    # Callback function to save the pipeline selection to Session State
    st.session_state.pipeline = st.session_state._pipeline

# Initialize session state
utilss.init_session_state()

st.write("# Welcome to NiChart Project!")

st.markdown(
    """
    NiChart is an open-source framework built specifically for
    deriving Machine Learning based indices from MRI data.
    """
)

with st.container(border=True):
    # Pipeline selection
    st.markdown(
        """
        :point_down: **Please select a pipeline!**
        """
    )
    st.selectbox(
        "Select pipeline:",
        st.session_state.pipelines,
        index=0,
        key="_pipeline",
        on_change=set_pipeline,
        label_visibility="collapsed"
    )
    utilmenu.menu()
    st.markdown(
        """
        :point_left: **And select a task from the sidebar to process, analyze and visualize your data!**
        """
    )

with st.expander('Want to learn more?', expanded=False):
    st.markdown(
        """
        - Check out [NiChart Web page](https://neuroimagingchart.com)
        - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
        - Jump into our [documentation](https://github.com/CBICA/NiChart_Project)
        - Ask a question in our [community
            forums](https://github.com/CBICA/NiChart_Project)
            """
    )

    st.markdown(
        """
        You can try NiChart manually via our github
        ```bash
        git clone https://github.com/CBICA/NiChart_Project
        git submodule update --init --recursive --remote
        pip install -r requirements.txt
        ```

        And to run the workflows, just run:
        ```bash
        python3 run.py --dir_input input folder --dir_output output_folder --studies 1 --version my_version --cores 4 --conda 0
        ```

        You can always find more options at our documentation
        """
)

st.sidebar.image("../resources/nichart1.png")
st.sidebar.info(
    """
    Note: This website is based on materials from the [NiChart Project](https://neuroimagingchart.com/).
    The content and the logo of NiChart are intellectual property of [CBICA](https://www.med.upenn.edu/cbica/).
    Make sure that you read the [licence](https://github.com/CBICA/NiChart_Project/blob/main/LICENSE).
    """
)

with st.sidebar.expander("Acknowledgments"):
    st.markdown(
        """
        The CBICA Dev team
        """
    )

# FIXME: For DEBUG
utilst.add_debug_panel()

