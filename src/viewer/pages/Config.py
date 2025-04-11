import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_panels as utilpn
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilss

# Page config should be called for each page
utilpg.config_page()
utilpg.select_main_menu()

def panel_models_path():
    """
    Panel for selecting models
    """
    with st.container(border=True):
        st.write('Work in progress!')

def panel_resources_path():
    """
    Panel for selecting resource directories
    """
    with st.container(border=True):
        sel_res = st.pills(
            "Select resource type",
            ["process definitions", "roi lists"],
            selection_mode="single",
            default=None,
            label_visibility="collapsed",
        )

        if sel_res is None:
            return
        
        if sel_res == "process definitions":
            out_dir = st.session_state.paths["proc_def"]

            # Browse output folder
            if st.button('Browse path'):
                sel_dir = utilio.browse_folder(out_dir)
                utilss.update_out_dir(sel_dir)
                st.rerun()

            # Enter output folder
            sel_dir = st.text_input(
                'Enter path',
                value=out_dir,
                # label_visibility='collapsed',
            )
            if sel_dir != out_dir:
                utilss.update_out_dir(sel_dir)
                st.rerun()

            if st.session_state.flags["out_dir"]:
                st.success(
                    f"Output directory: {st.session_state.paths['out_dir']}",
                    icon=":material/thumb_up:",
                )

            utildoc.util_help_dialog(utildoc.title_out, utildoc.def_out)

def panel_out_dir():
    """
    Panel for selecting output dir
    """
    with st.container(border=True):
        out_dir = st.session_state.paths["out_dir"]

        # Browse output folder
        if st.button('Browse path'):
            sel_dir = utilio.browse_folder(out_dir)
            utilss.update_out_dir(sel_dir)
            st.rerun()

        # Enter output folder
        sel_dir = st.text_input(
            'Enter path',
            value=out_dir,
            # label_visibility='collapsed',
        )
        if sel_dir != out_dir:
            utilss.update_out_dir(sel_dir)
            st.rerun()

        if st.session_state.flags["out_dir"]:
            st.success(
                f"Output directory: {st.session_state.paths['out_dir']}",
                icon=":material/thumb_up:",
            )

        utildoc.util_help_dialog(utildoc.title_out, utildoc.def_out)

def panel_task() -> None:
    """
    Panel for selecting task name
    """
    with st.container(border=True):

        out_dir = st.session_state.paths["out_dir"]
        curr_task = st.session_state.navig['task']

        # Select from existing
        list_tasks = utilio.get_subfolders(out_dir)
        if len(list_tasks) > 0:
            st.write("Select Existing Task")
            sel_task = st.pills(
                "Options:",
                options = list_tasks,
                default = curr_task,
                label_visibility="collapsed",
            )
            if sel_task != curr_task:
                utilss.update_task(sel_task)
                st.rerun()

        # Enter new
        st.write("Enter New Task Name")
        sel_task = st.text_input(
            "Task name:",
            None,
            label_visibility="collapsed",
            placeholder="My_new_study"
        )
        if sel_task is not None and sel_task != curr_task:
            utilss.update_task(sel_task)
            st.rerun()

        if st.session_state.flags["task"]:
            st.success(
                f"Task name: {st.session_state.navig['task']}",
                icon=":material/thumb_up:",
            )

        utildoc.util_help_dialog(utildoc.title_exp, utildoc.def_exp)


st.markdown(
    """
    ### Configuration Options
    - Select configuration options here.
    """
)

sel_config_cat = st.pills(
    "Select Config Category",
    ["Basic", "Advanced"],
    selection_mode="single",
    default=None,
    label_visibility="collapsed",
)

if sel_config_cat == "Basic":
    sel_config = st.pills(
        "Select Basic Config",
        ["Output Dir", "Task Name"],
        selection_mode="single",
        default=None,
        label_visibility="collapsed",
    )

    if sel_config == "Output Dir":
        panel_out_dir()

    if sel_config == "Task Name":
        panel_task()

elif sel_config_cat == "Advanced":
    sel_config = st.pills(
        "Select Advanced Config",
        ["Resources", "Models"],
        selection_mode="single",
        default=None,
        label_visibility="collapsed",
    )

    if sel_config == "Resources":
        panel_resources_path()

    if sel_config == "Models":
        panel_models_path()
