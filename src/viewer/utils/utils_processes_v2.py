import streamlit as st
import json
import os
from graphviz import Digraph
import yaml

def load_steps_from_folder(folder_path):
    steps = {}
    for file in os.listdir(folder_path):
        if file.endswith(".yaml") or file.endswith(".yml"):
            with open(os.path.join(folder_path, file), "r") as f:
                step_data = yaml.safe_load(f)
                steps[step_data["pname"]] = step_data
    return steps

# --- Build Graph ---
def build_graphviz_pipeline(steps):
    dot = Digraph()
    dot.attr(rankdir='TB')

    for pname, step in steps.items():
        dot.node(pname, label=pname, shape="box", style="filled", fillcolor="lightblue")
        for i in step.get("input", []):
            dot.node(i, shape="ellipse", style="filled", fillcolor="lightgreen")
            dot.edge(i, pname)
        for o in step.get("output", []):
            dot.node(o, shape="ellipse", style="filled", fillcolor="lightyellow")
            dot.edge(pname, o)

    return dot

# --- UI Starts Here ---
st.title("🔧 Dynamic Filtered Process Explorer")

steps = load_steps_from_folder("/home/guraylab/GitHub/gurayerus/NiChart_Project/resources/process_definitions")
step_list = list(steps.values())

# Get all unique items
all_inputs = sorted({i for step in step_list for i in step.get("input", [])})
all_outputs = sorted({o for step in step_list for o in step.get("output", [])})
all_pnames = sorted({step["pname"] for step in step_list})

# Create session state to hold selections
if "selected_inputs" not in st.session_state:
    st.session_state.selected_inputs = []
if "selected_outputs" not in st.session_state:
    st.session_state.selected_outputs = []
if "selected_pnames" not in st.session_state:
    st.session_state.selected_pnames = []

# --- Filter helper ---
def apply_partial_filter(steps, inputs=None, outputs=None, pnames=None):
    return [
        step for step in steps
        if (not inputs or any(i in step.get("input", []) for i in inputs)) and
           (not outputs or any(o in step.get("output", []) for o in outputs)) and
           (not pnames or step["pname"] in pnames)
    ]

# --- Update multiselect options dynamically ---
filtered_for_inputs = apply_partial_filter(step_list, outputs=st.session_state.selected_outputs, pnames=st.session_state.selected_pnames)
filtered_for_outputs = apply_partial_filter(step_list, inputs=st.session_state.selected_inputs, pnames=st.session_state.selected_pnames)
filtered_for_pnames = apply_partial_filter(step_list, inputs=st.session_state.selected_inputs, outputs=st.session_state.selected_outputs)

available_inputs = sorted({i for step in filtered_for_inputs for i in step.get("input", [])})
available_outputs = sorted({o for step in filtered_for_outputs for o in step.get("output", [])})
available_pnames = sorted({step["pname"] for step in filtered_for_pnames})

# --- Select boxes with dynamic options ---
st.session_state.selected_inputs = st.multiselect("Filter by Input(s)", available_inputs, default=st.session_state.selected_inputs)
st.session_state.selected_outputs = st.multiselect("Filter by Output(s)", available_outputs, default=st.session_state.selected_outputs)
st.session_state.selected_pnames = st.multiselect("Filter by Process Name(s)", available_pnames, default=st.session_state.selected_pnames)

# --- Final filter based on current selections ---
final_filtered = apply_partial_filter(
    step_list,
    inputs=st.session_state.selected_inputs,
    outputs=st.session_state.selected_outputs,
    pnames=st.session_state.selected_pnames
)
final_step_dict = {step["pname"]: step for step in final_filtered}

# --- Display ---
if final_step_dict:
    st.graphviz_chart(build_graphviz_pipeline(final_step_dict))
else:
    st.warning("⚠️ No steps match the current filters.")
