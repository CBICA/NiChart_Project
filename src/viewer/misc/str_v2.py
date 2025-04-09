import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import deque, defaultdict

# Load YAML step definitions
def load_steps(folder):
    steps = {}
    for file in Path(folder).glob("*.yaml"):
        with open(file) as f:
            data = yaml.safe_load(f)
            steps[data['pname']] = data
    return steps

# Get all files and their roles
def get_file_roles(steps):
    roles = defaultdict(lambda: {"input": False, "output": False})
    for step in steps.values():
        for f in step.get("input", []):
            roles[f]["input"] = True
        for f in step.get("output", []):
            roles[f]["output"] = True
    return roles

# Color by role
def get_file_color(role):
    if role["input"] and role["output"]:
        return "khaki"
    elif role["input"]:
        return "lightgreen"
    elif role["output"]:
        return "salmon"
    else:
        return "white"

# Build a Graphviz subgraph from selected inputs
def build_subgraph_from_inputs(steps, selected_inputs):
    available_files = set(selected_inputs)
    runnable_steps = []
    generated_files = set()

    while True:
        progress = False
        for name, step in steps.items():
            if name in runnable_steps:
                continue
            if all(f in available_files for f in step.get("input", [])):
                runnable_steps.append(name)
                generated_files.update(step.get("output", []))
                available_files.update(step.get("output", []))
                progress = True
        if not progress:
            break

    return build_graphviz_subgraph(steps, runnable_steps), runnable_steps

# Build the styled Graphviz graph (subgraph)
def build_graphviz_subgraph(steps, included_steps):
    dot = Digraph()
    dot.attr(rankdir='TB')
    file_roles = get_file_roles(steps)

    used_files = set()
    for step_name in included_steps:
        step = steps[step_name]
        dot.node(step_name, shape='box', style='filled', fillcolor='lightblue')

        for input_file in step.get("input", []):
            used_files.add(input_file)
            color = get_file_color(file_roles[input_file])
            dot.node(input_file, shape='ellipse', style='filled', fillcolor=color)
            dot.edge(input_file, step_name)

        for output_file in step.get("output", []):
            used_files.add(output_file)
            color = get_file_color(file_roles[output_file])
            dot.node(output_file, shape='ellipse', style='filled', fillcolor=color)
            dot.edge(step_name, output_file)

    return dot

# Generate command line from steps
def generate_pipeline_command(steps_order, steps_data):
    cmds = []
    for name in steps_order:
        step = steps_data[name]
        params = " ".join(f"--{k} {v}" for k, v in step.get("parameters", {}).items())
        cmd = f"{name}.py {params}"
        cmds.append(cmd)
    return " | \\\n  ".join(cmds)

# --- Streamlit UI ---
st.title("Dynamic Pipeline Builder")

folder = st.text_input("Path to YAML step files", "./steps")
if Path(folder).exists():
    steps_data = load_steps(folder)
    file_roles = get_file_roles(steps_data)

    all_files = sorted(file_roles.keys())
    st.subheader("Select Available Input Files")
    selected_inputs = st.multiselect("Available files you already have", all_files)

    if selected_inputs:
        dot, runnable_steps = build_subgraph_from_inputs(steps_data, selected_inputs)

        st.markdown("### Runnable Steps Based on Inputs")
        st.write(runnable_steps)

        st.markdown("### Dependency Tree")
        st.graphviz_chart(dot.source)

        if runnable_steps:
            pipeline_cmd = generate_pipeline_command(runnable_steps, steps_data)
            st.markdown("### Generated Pipeline Command")
            st.code(pipeline_cmd, language='bash')
else:
    st.error("YAML folder not found.")
