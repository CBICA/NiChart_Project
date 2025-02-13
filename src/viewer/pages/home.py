import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss
import utils.utils_st as utilst
from streamlit_extras.stylable_container import stylable_container
from streamlitextras.webutils import stxs_javascript

# Page config should be called for each page
utilss.config_page()


def set_pipeline() -> None:
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
        label_visibility="collapsed",
    )
    utilmenu.menu()
    st.markdown(
        """
        :point_left: **And select a task from the sidebar to process, analyze and visualize your data!**
        """
    )

with st.expander("Want to learn more?", expanded=False):
    st.markdown(
        """
        - Check out [NiChart Web page](https://neuroimagingchart.com)
        - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
        - Jump into [our documentation](https://cbica.github.io/NiChart_Project)
        - Ask a question in [our community discussions](https://github.com/CBICA/NiChart_Project/discussions)
            """
    )

    st.markdown(
        """
        You can install NiChart Project desktop
        ```
        pip install NiChart_Project
        ```

        and run the application
        ```
        cd src/viewer
        streamlit run NiChartProject.py
        ```

        Alternatively, the cloud app can be launched at
        ```
        https://cloud.neuroimagingchart.com
        ```
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

# st.markdown("""
# <style>
# .st-bb {
# background-color: #4CAF50; /* Green */
# color: white;
# }
# </style>
# """, unsafe_allow_html=True)

with st.container(border=True):

    st.markdown("### NiChart Surveys")
    st.markdown(
        "😊 Your opinion matters! Kindly take a moment to complete these two brief surveys!"
    )

    with stylable_container(
        key="my_button_container",
        css_styles="""
            button {
                background-color: #FF7944;
                color: white;
                border-radius: 20px;
            }
            """,
    ):
        if st.button(
            "📝 NiChart User Experience",
        ):
            # This code only works locally, not on a container or server.
            # webbrowser.open_new_tab('https://forms.office.com/r/mM1kx1XsgS')
            stxs_javascript(
                """window.open('https://forms.office.com/r/mM1kx1XsgS', '_blank').focus()"""
            )

        if st.button(
            "📝 Shaping the Future of NiChart",
        ):
            # This code only works locally, not on a container or server.
            # webbrowser.open_new_tab('https://forms.office.com/r/acwgn2WCc4')
            stxs_javascript(
                """window.open('https://forms.office.com/r/acwgn2WCc4', '_blank').focus()"""
            )

    # Bg color on link_button was not supported in styllable container
    # st.link_button(
    # '📝 NiChart User Experience',
    # 'https://forms.office.com/r/mM1kx1XsgS',
    # )

    # st.link_button(
    # '📝 Shaping the Future of NiChart',
    # 'https://forms.office.com/r/acwgn2WCc4',
    # )

# FIXME: For DEBUG
utilst.add_debug_panel()
