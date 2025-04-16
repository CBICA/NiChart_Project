import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict

# Load YAML step definitions
def load_steps(folder):
    steps = {}
    for file in Path(folder).glob("*.yaml"):
        with open(file) as f:
            data = yaml.safe_load(f)
            steps[data['pname']] = data
    return steps

# Determine roles of files
def get_file_roles(steps):
    roles = defaultdict(lambda: {"input": False, "output": False})
    for step in steps.values():
        for f in step.get("input", []):
            roles[f]["input"] = True
        for f in step.get("output", []):
            roles[f]["output"] = True
    return roles

# Choose color by file role
def get_file_color(role):
    if role["input"] and role["output"]:
        return "khaki"
    elif role["input"]:
        return "lightgreen"
    elif role["output"]:
        return "salmon"
    else:
        return "white"

# Build the visual pipeline
def build_graphviz_pipeline(steps, selected_steps, starting_inputs):
    dot = Digraph()
    dot.attr(rankdir='TB')
    file_roles = get_file_roles(steps)
    used_files = set(starting_inputs)

    for step_name in selected_steps:
        step = steps[step_name]
        dot.node(step_name, shape='box', style='filled', fillcolor='lightblue')

        for f in step.get("input", []):
            color = get_file_color(file_roles[f])
            dot.node(f, shape='ellipse', style='filled', fillcolor=color)
            dot.edge(f, step_name)

        for f in step.get("output", []):
            color = get_file_color(file_roles[f])
            dot.node(f, shape='ellipse', style='filled', fillcolor=color)
            dot.edge(step_name, f)
            used_files.add(f)

    for f in starting_inputs:
        if f not in used_files:
            dot.node(f, shape='ellipse', style='filled', fillcolor='lightgreen')

    return dot

# Get next runnable steps
def get_runnable_steps(steps, completed_steps, available_files):
    runnable = []
    for name, step in steps.items():
        if name in completed_steps:
            continue
        if all(f in available_files for f in step.get("input", [])):
            runnable.append(name)
    return runnable

# Generate CLI pipeline command
def generate_pipeline_command(step_list, steps):
    cmds = []
    for name in step_list:
        step = steps[name]
        params = " ".join(f"--{k} {v}" for k, v in step.get("parameters", {}).items())
        cmd = f"{name}.py {params}"
        cmds.append(cmd)
    return " | \\\n  ".join(cmds)

# --- Streamlit App Starts Here ---
st.set_page_config(page_title="Interactive Pipeline Builder", layout="wide")
st.title("🧬 Interactive Pipeline Builder")

# --- Load steps ---
folder = st.text_input("Path to YAML step files", "./steps")
if Path(folder).exists():
    steps_data = load_steps(folder)
    file_roles = get_file_roles(steps_data)

    #all_files = sorted(file_roles.keys())
    # Exclude files that are outputs of any step (only allow true source files)
    output_files = {f for step in steps_data.values() for f in step.get("output", [])}
    input_candidates = sorted(set(file_roles.keys()) - output_files)

    # --- Session State Initialization ---
    if "selected_inputs" not in st.session_state:
        st.session_state.selected_inputs = []
    if "selected_steps" not in st.session_state:
        st.session_state.selected_steps = []
    if "available_files" not in st.session_state:
        st.session_state.available_files = set()

    # --- Input Selection ---
    if not st.session_state.selected_inputs:
        st.subheader("1️⃣ Select Available Input Files")
        selected_inputs = st.multiselect("Choose files you already have", input_candidates)
        if st.button("Confirm Inputs") and selected_inputs:
            st.session_state.selected_inputs = selected_inputs
            st.session_state.available_files = set(selected_inputs)
            st.rerun()

    # --- Step-by-Step Build ---
    else:
        st.subheader("2️⃣ Build Pipeline Step by Step")
        current_pipeline = st.session_state.selected_steps
        available_files = st.session_state.available_files

        runnable = get_runnable_steps(steps_data, current_pipeline, available_files)
        st.markdown(f"**Available Files**: `{sorted(list(available_files))}`")
        st.markdown(f"**Steps in Pipeline**: `{current_pipeline}`")

        if runnable:
            selected_step = st.selectbox("Select next step to add", runnable)
            if st.button("Add Step"):
                st.session_state.selected_steps.append(selected_step)
                step_outputs = steps_data[selected_step].get("output", [])
                st.session_state.available_files.update(step_outputs)
                st.rerun()
        else:
            st.info("No more steps can be added with current files.")

        # --- Graph + Command Output ---
        st.markdown("### 📊 Current Pipeline Graph")
        graph = build_graphviz_pipeline(
            steps_data,
            st.session_state.selected_steps,
            st.session_state.selected_inputs,
        )
        st.graphviz_chart(graph.source)

        if st.session_state.selected_steps:
            st.markdown("### 💻 Generated Command")
            st.code(
                generate_pipeline_command(
                    st.session_state.selected_steps, steps_data
                ),
                language="bash",
            )

        if st.button("🔄 Reset Pipeline"):
            st.session_state.clear()
            st.rerun()
else:
    st.error("YAML folder not found.")
