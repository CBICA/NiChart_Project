import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses


def panel_interactive():
    with st.container(border=True):
        st.markdown(
            """
            ### Interactive Pipeline Builder
            """
        )
        proc = st.session_state.processes

        # --- Input Selection ---
        if not proc['sel_inputs']:
            st.markdown("### Step 1. Select Available Input Files")
            sel_inputs = st.multiselect(
                "Choose files you already have",
                proc['in_files']
            )
            if st.button("Confirm Inputs") and sel_inputs:
                proc['sel_inputs'] = sel_inputs
                proc['avail_files'] = set(sel_inputs)
                st.rerun()

        # --- Step-by-Step Build ---
        else:
            st.markdown("### Step 2. Add Pipeline Step")
            current_pipeline = proc['sel_steps']
            avail_files = proc['avail_files']
            runnable = utilprc.get_runnable_steps(
                proc['data'],
                current_pipeline,
                avail_files
            )
            st.markdown(f"**Available Files**: `{sorted(list(avail_files))}`")
            st.markdown(f"**Steps in Pipeline**: `{current_pipeline}`")

            if runnable:
                sel_step = st.selectbox("Select next step to add", runnable)
                if st.button("Add Step"):
                    proc['sel_steps'].append(sel_step)
                    step_outputs = proc['data'][sel_step].get("output", [])
                    proc['avail_files'].update(step_outputs)
                    st.rerun()
            else:
                st.info("No more steps can be added with current files.")

            # --- Graph + Command Output ---
            st.markdown("### Current Pipeline Graph")
            graph = utilprc.build_graphviz_pipeline(
                proc['data'],
                proc['sel_steps'],
                proc['sel_inputs'],
            )
            st.graphviz_chart(graph.source)

            if proc['sel_steps']:
                st.markdown("### Pipeline Command")
                st.code(
                    utilprc.generate_pipeline_command(
                        proc['sel_steps'], proc['data']
                    ),
                    language="bash",
                )

            if st.button("🔄 Reset Pipeline"):
                utilses.update_proc_def(st.session_state.paths['proc_def'])
                st.rerun()


# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    ### Select Pipeline
    """
)

st.markdown("##### Select Task")
list_tasks = ["By Input", "By Output", "Interactive"]
sel_task = st.pills(
    "Select Workflow Task", list_tasks, selection_mode="single", label_visibility="collapsed"
)
if sel_task == "By Input":
    st.write('Wait...')
elif sel_task == "By Output":
    st.write('Wait...')
elif sel_task == "Interactive":
    panel_interactive()

