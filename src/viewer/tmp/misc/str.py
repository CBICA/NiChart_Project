import streamlit as st
import os
import yaml
import networkx as nx
from pathlib import Path
from collections import deque

from graphviz import Digraph

def get_file_color(role):
    if role["input"] and role["output"]:
        return "khaki"  # intermediate file
    elif role["input"]:
        return "lightgreen"
    elif role["output"]:
        return "salmon"
    else:
        return "white"

def build_graphviz_graph(steps):
    dot = Digraph(format='png')
    dot.attr(rankdir='TB')  # Top to bottom

    # Step 1: Determine all file roles
    file_roles = {}  # file_name: {'input': bool, 'output': bool}
    for step in steps.values():
        for f in step.get("input", []):
            file_roles.setdefault(f, {"input": False, "output": False})["input"] = True
        for f in step.get("output", []):
            file_roles.setdefault(f, {"input": False, "output": False})["output"] = True

    # Step 2: Add process and file nodes
    for step_name, step in steps.items():
        # Process step node
        dot.node(step_name, shape='box', style='filled', fillcolor='lightblue')

        for input_file in step.get("input", []):
            fill = get_file_color(file_roles[input_file])
            dot.node(input_file, shape='ellipse', style='filled', fillcolor=fill)
            dot.edge(input_file, step_name)

        for output_file in step.get("output", []):
            fill = get_file_color(file_roles[output_file])
            dot.node(output_file, shape='ellipse', style='filled', fillcolor=fill)
            dot.edge(step_name, output_file)

    return dot

#def build_graphviz_graph(steps):
    #dot = Digraph(format='png')
    ##dot.attr(rankdir='LR')  # Left to right
    #dot.attr(rankdir='TB')  # Left to right

    #for step_name, step in steps.items():
        ## Add process node
        #dot.node(step_name, shape='box', style='filled', fillcolor='lightblue')

        ## Add input data nodes and edges
        #for input_file in step.get("input", []):
            #dot.node(input_file, shape='ellipse', style='filled', fillcolor='lightgreen')
            #dot.edge(input_file, step_name)

        ## Add output data nodes and edges
        #for output_file in step.get("output", []):
            #dot.node(output_file, shape='ellipse', style='filled', fillcolor='salmon')
            #dot.edge(step_name, output_file)

    #return dot


# Load all YAML step definitions from a folder
def load_steps(folder):
    steps = {}
    for file in Path(folder).glob("*.yaml"):
        with open(file) as f:
            data = yaml.safe_load(f)
            steps[data['pname']] = data
    return steps

# Build dependency graph
def build_graph(steps):
    G = nx.DiGraph()
    for step_name, step in steps.items():
        G.add_node(step_name, **step)
        for inp in step.get("input", []):
            for other_name, other_step in steps.items():
                if inp in other_step.get("output", []):
                    G.add_edge(other_name, step_name)
    return G

def build_graph_with_data_nodes(steps):
    G = nx.DiGraph()
    for step_name, step in steps.items():
        G.add_node(step_name, type="process")

        # Add edges from input files to the step
        for input_file in step.get("input", []):
            G.add_node(input_file, type="data")
            G.add_edge(input_file, step_name)

        # Add edges from the step to output files
        for output_file in step.get("output", []):
            G.add_node(output_file, type="data")
            G.add_edge(step_name, output_file)
    return G

# Get all dependencies for selected steps
def get_dependency_closure(G, selected):
    closure = set()
    queue = deque(selected)
    while queue:
        node = queue.popleft()
        if node not in closure:
            closure.add(node)
            queue.extendleft(G.predecessors(node))
    return list(nx.topological_sort(G.subgraph(closure)))

# Generate command line
def generate_pipeline_command(steps_order, steps_data):
    cmds = []
    for name in steps_order:
        step = steps_data[name]
        params = " ".join(f"--{k} {v}" for k, v in step.get("parameters", {}).items())
        cmd = f"{name}.py {params}"
        cmds.append(cmd)
    return " | \\\n  ".join(cmds)

# --- Streamlit UI ---
st.title("Pipeline Builder from YAML Steps")

folder = st.text_input("Path to YAML step files", "./steps")
if Path(folder).exists():
    steps_data = load_steps(folder)
    #G = build_graph(steps_data)
    G = build_graph_with_data_nodes(steps_data)

    st.subheader("Pipeline Dependency Graph")

    #try:
    #st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())
    dot = build_graphviz_graph(steps_data)
    st.graphviz_chart(dot.source)
    #except Exception:
        #st.warning("Unable to render graph.")

    selected_steps = st.multiselect("Select pipeline steps to run", list(G.nodes))
    if selected_steps:
        steps_order = get_dependency_closure(G, selected_steps)
        st.markdown("### Ordered Steps to Run")
        st.write(steps_order)

        pipeline_cmd = generate_pipeline_command(steps_order, steps_data)
        st.markdown("### Generated Pipeline Command")
        st.code(pipeline_cmd, language='bash')
else:
    st.error("Folder not found.")
