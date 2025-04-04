import streamlit as st

###################################
# Hard-coded menu items for NiChart

dict_main_menu = {
    "Home": "pages/Home.py",
    "Config": "pages/Config.py",
    "Pipelines": "pages/Menu.py",
    "Debug": "pages/Debug.py",
}

dict_pipelines = {
    "Interactive Brain Anatomy": "pages/p_brain_anatomy.py",
    "sMRI Biomarkers": "pages/Menu.py",
    "White Matter Lesion Segmentation": "pages/Menu.py",
    "DTI Biomarkers": "pages/Menu.py",
    "fMRI Biomarkers": "pages/Menu.py",
}

dict_pipeline_steps = {
    "Interactive Brain Anatomy": {
        "Overview": "pages/p_dlmusebio_overview.py",
        "Upload Data": "pages/p_dlmusebio_indata.py",
        "DLMUSE": "pages/process_dlmuse.py",
        "ML Biomarkers": "pages/p_dlmusebio_mlscores.py",
        "Plotting": "pages/p_dlmusebio_plot.py",
    },
    "sMRI Biomarkers": {
        "Overview": "pages/p_dlmusebio_overview.py",
        "Upload Data": "pages/p_dlmusebio_indata.py",
        "DLMUSE": "pages/process_dlmuse.py",
        "ML Biomarkers": "pages/p_dlmusebio_mlscores.py",
        "Plotting": "pages/p_dlmusebio_plot.py",
    },
    "White Matter Lesion Segmentation": {
        "Overview": "pages/p_dlwmls_overview.py",
        "Upload Data": "pages/p_dlwmls_indata.py",
        "DLWMLS": "pages/process_dlwmls.py",
        "Plotting": "pages/plot_sMRI_vars_study.py",
    },
}


def select_main_menu() -> None:
    """
    Select main menu page from a list and switch to it
    """
    with st.sidebar:
        sel_main_menu = st.pills(
            "Select Main Menu",
            dict_main_menu.keys(),
            selection_mode="single",
            default=st.session_state.sel_main_menu,
            label_visibility="collapsed",
        )

        if sel_main_menu is None:
            return

        if sel_main_menu == st.session_state.sel_main_menu:
            return

        # Reset selection in next steps
        st.session_state.sel_pipeline = None
        st.session_state.sel_pipeline_step = None

        st.session_state.sel_main_menu = sel_main_menu
        sel_page = dict_main_menu[sel_main_menu]
        st.switch_page(sel_page)


def select_pipeline() -> None:
    """
    Select pipeline from a list and switch to pipeline page
    """
    if st.session_state.sel_main_menu != "Pipelines":
        return

    with st.sidebar:
        # st.markdown('##### ')
        st.markdown("### Pipeline:")
        sel_pipeline = st.pills(
            "Pipelines",
            dict_pipelines.keys(),
            selection_mode="single",
            default=st.session_state.sel_pipeline,
            label_visibility="collapsed",
        )
        if sel_pipeline is None:
            return
        if sel_pipeline == st.session_state.sel_pipeline:
            return

        # Reset selection in next steps
        st.session_state.sel_pipeline_step = None

        st.session_state.sel_pipeline = sel_pipeline
        sel_page = dict_pipelines[sel_pipeline]
        st.switch_page(sel_page)


def select_pipeline_step() -> None:
    """
    Select pipeline step from a list and switch page
    """
    if st.session_state.sel_pipeline is None:
        return

    sel_dict = dict_pipeline_steps[st.session_state.sel_pipeline]
    with st.sidebar:
        st.markdown("### Pipeline step:")
        sel_step = st.pills(
            "Pipeline steps",
            sel_dict.keys(),
            selection_mode="single",
            default=st.session_state.sel_pipeline_step,
            label_visibility="collapsed",
        )
        if sel_step is None:
            return
        if sel_step == st.session_state.sel_pipeline_step:
            return
        st.session_state.sel_pipeline_step = sel_step
        sel_page = sel_dict[sel_step]
        st.switch_page(sel_page)


def config_page() -> None:
    st.set_page_config(
        page_title="NiChart",
        page_icon=st.session_state.nicon,
        layout="wide",
        # layout="centered",
        menu_items={
            "Get help": "https://neuroimagingchart.com/",
            "Report a bug": "https://github.com/CBICA/NiChart_Project/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=%5BBUG%5D+",
            "About": "https://neuroimagingchart.com/",
        },
    )


def show_menu() -> None:
    select_main_menu()
    select_pipeline()
    select_pipeline_step()
