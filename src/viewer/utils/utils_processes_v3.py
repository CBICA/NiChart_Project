import streamlit as st
from collections import defaultdict, deque
import yaml
from graphviz import Digraph
import os

# Load steps from YAML files
@st.cache_data
def load_steps_from_yaml(folder):
    steps = {}
    for fname in os.listdir(folder):
        if fname.endswith(".yaml") or fname.endswith(".yml"):
            with open(os.path.join(folder, fname), "r") as f:
                data = yaml.safe_load(f)
                steps[data["pname"]] = data
    return steps

# Build graph (only once)
def build_graph(steps):
    step_inputs = {}
    step_outputs = {}
    file_to_producers = defaultdict(list)
    file_to_consumers = defaultdict(list)

    for name, step in steps.items():
        ins = set(step.get("input", []))
        outs = set(step.get("output", []))
        step_inputs[name] = ins
        step_outputs[name] = outs
        for f in outs:
            file_to_producers[f].append(name)
        for f in ins:
            file_to_consumers[f].append(name)

    return {
        "step_inputs": step_inputs,
        "step_outputs": step_outputs,
        "file_to_producers": file_to_producers,
        "file_to_consumers": file_to_consumers
    }

# Reachable steps from starting ones
def get_reachable_steps(graph, starting_steps):
    step_inputs = graph["step_inputs"]
    step_outputs = graph["step_outputs"]
    file_to_consumers = graph["file_to_consumers"]

    reachable_steps = set(starting_steps)
    produced_files = set()
    queue = deque(starting_steps)

    while queue:
        step = queue.popleft()
        outputs = step_outputs.get(step, [])
        produced_files.update(outputs)
        for f in outputs:
            for consumer in file_to_consumers.get(f, []):
                if consumer in reachable_steps:
                    continue
                if step_inputs[consumer].issubset(produced_files):
                    reachable_steps.add(consumer)
                    queue.append(consumer)

    return reachable_steps

# Group reachable steps into disconnected pipelines and sort
def sort_disconnected_pipelines(graph, reachable_steps):
    step_inputs = graph["step_inputs"]
    step_outputs = graph["step_outputs"]

    # Step graph
    dep_graph = defaultdict(set)
    rev_graph = defaultdict(set)

    for step in reachable_steps:
        for other in reachable_steps:
            if step != other and step_inputs[step] & step_outputs[other]:
                dep_graph[step].add(other)
                rev_graph[other].add(step)

    # Connected components
    def find_components():
        visited = set()
        components = []

        for step in reachable_steps:
            if step not in visited:
                comp = []
                queue = deque([step])
                while queue:
                    s = queue.popleft()
                    if s not in visited:
                        visited.add(s)
                        comp.append(s)
                        queue.extend(dep_graph[s])
                        queue.extend(rev_graph[s])
                components.append(comp)
        return components

    # Topo sort
    def topo_sort(nodes):
        deps = {n: set() for n in nodes}
        for n in nodes:
            for m in dep_graph[n]:
                if m in nodes:
                    deps[n].add(m)
        res = []
        queue = deque([n for n in nodes if not deps[n]])
        while queue:
            curr = queue.popleft()
            res.append(curr)
            for other in nodes:
                if curr in deps[other]:
                    deps[other].remove(curr)
                    if not deps[other]:
                        queue.append(other)
        return res

    return [topo_sort(comp) for comp in find_components()]

# Draw Graphviz
def draw_pipeline_graph(steps, pipeline):
    dot = Digraph()
    for step in pipeline:
        dot.node(step, shape="box", style="filled", fillcolor="lightblue")
        for inp in steps[step].get("input", []):
            dot.node(inp, shape="ellipse", style="filled", fillcolor="lightgray")
            dot.edge(inp, step)
        for out in steps[step].get("output", []):
            dot.node(out, shape="ellipse", style="filled", fillcolor="lightgreen")
            dot.edge(step, out)
    return dot

# ---------- Streamlit UI ----------

st.title("🧩 Pipeline Viewer")

# Folder selection
folder = st.text_input("Path to YAML steps folder", "pipeline_steps")

# Load steps and build graph
if os.path.isdir(folder):
    steps = load_steps_from_yaml(folder)

    if "pipeline_graph" not in st.session_state:
        st.session_state.pipeline_graph = build_graph(steps)

    graph = st.session_state.pipeline_graph

    # Step selection
    all_steps = list(steps.keys())
    selected_start_steps = st.multiselect("Select starting steps", all_steps)

    if selected_start_steps:
        reachable = get_reachable_steps(graph, set(selected_start_steps))
        sorted_pipelines = sort_disconnected_pipelines(graph, reachable)

        st.subheader("📊 Reachable Pipelines")

        for i, pipe in enumerate(sorted_pipelines):
            st.markdown(f"#### Pipeline {i + 1}")
            st.graphviz_chart(draw_pipeline_graph(steps, pipe))
else:
    st.warning("Please enter a valid folder path containing .yaml step files.")
