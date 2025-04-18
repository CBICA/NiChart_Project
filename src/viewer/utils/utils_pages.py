import streamlit as st

###################################
# Hard-coded menu items for NiChart
dict_main_menu = {
    "Home": "pages/home.py",
    "Config": "pages/config.py",
    "Workflow": "pages/menu.py",
    "Debug": "pages/debug.py",
}

dict_workflow = {
    "Data": "pages/select_input.py",
    "Pipeline": "pages/run_pipeline.py",
    "Results": "pages/menu.py",
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
            default=st.session_state.navig['main_menu'],
            label_visibility="collapsed",
            key='_sel_main_menu'
        )
        if sel_main_menu is None:
            if '_sel_workflow' in st.session_state:
                del st.session_state['_sel_workflow']            
            st.session_state.navig['main_menu'] == None
            return

        if sel_main_menu == st.session_state.navig['main_menu']:
            return

        # Set menu selection
        st.session_state.navig['main_menu'] = sel_main_menu

        # Reset selection in next steps
        st.session_state.navig['workflow'] = None
        st.session_state.navig['pipeline_step'] = None
        
        # Navigate to selected page
        sel_page = dict_main_menu[sel_main_menu]
        st.switch_page(sel_page)

def select_workflow() -> None:
    """
    Select pipeline from a list and switch to pipeline page
    """
    if st.session_state.navig['main_menu'] != "Workflow":
        return

    with st.sidebar:
        # st.markdown('##### ')
        st.markdown("### Workflow Steps:")
        sel_workflow = st.pills(
            "Workflow",
            dict_workflow.keys(),
            selection_mode="single",
            default=st.session_state.navig['workflow'],
            label_visibility="collapsed",
            key='_sel_workflow'            
        )
        if sel_workflow is None:
            return
        if sel_workflow == st.session_state.navig['workflow']:
            return

        st.session_state.navig['workflow'] = sel_workflow
        sel_page = dict_workflow[sel_workflow]
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
    select_workflow()
