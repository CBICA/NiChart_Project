import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
from collections import defaultdict, deque

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
    
def detect_reachable_steps(graph, list_inputs, flag_all = False):
    '''
    Creates a graph from process definitions and detect steps reachable from given input list
    '''
    #file_to_producers = defaultdict(list)
    #file_to_consumers = defaultdict(list)
    #step_inputs = {}
    #step_outputs = {}

    #for name, step in steps.items():
        #step_inputs[name] = set(step.get("input", []))
        #step_outputs[name] = set(step.get("output", []))
        #for f in step_outputs[name]:
            #file_to_producers[f].append(name)
        #for f in step_inputs[name]:
            #file_to_consumers[f].append(name)

    # Forward propagation of input origin tracking
    file_origins = {f: {f} for f in list_inputs}
    queue = deque(list_inputs)
    while queue:
        f = queue.popleft()
        for step in graph['file_to_consumers'].get(f, []):
            if graph['step_inputs'][step].issubset(file_origins):
                origins = set().union(*(file_origins[i] for i in graph['step_inputs'][step]))
                for out in graph['step_outputs'][step]:
                    prev = file_origins.get(out, set())
                    file_origins[out] = prev | origins
                    if origins - prev:
                        queue.append(out)
    if flag_all:
        # Find outputs that depend on all starting inputs
        target_outputs = {
            f for f, origins in file_origins.items() if origins.issuperset(list_inputs)
        }

    else:
        # Find outputs that depend on ANY of the starting inputs
        target_outputs = {
            f for f, origins in file_origins.items() if origins & set(list_inputs)
        }


    # Backward trace to find required steps
    needed_steps = set()
    needed_files = set(target_outputs)
    queue = deque(target_outputs)
    while queue:
        f = queue.popleft()
        for step in graph['file_to_producers'].get(f, []):
            if step not in needed_steps:
                needed_steps.add(step)
                needed_files.update(graph['step_inputs'][step])
                queue.extend(graph['step_inputs'][step])

    return needed_steps


def topological_sort(steps, sel_steps):
    from collections import defaultdict, deque

    # Build graph for sorting
    dependency_graph = defaultdict(set)
    reverse_graph = defaultdict(set)

    for name in sel_steps:
        inputs = set(steps[name].get("input", []))
        for other in sel_steps:
            if other != name:
                if inputs & set(steps[other].get("output", [])):
                    dependency_graph[name].add(other)
                    reverse_graph[other].add(name)

    # Kahn’s algorithm
    sorted_steps = []
    no_deps = deque([s for s in sel_steps if not dependency_graph[s]])

    while no_deps:
        current = no_deps.popleft()
        sorted_steps.append(current)

        for neighbor in reverse_graph[current]:
            dependency_graph[neighbor].remove(current)
            if not dependency_graph[neighbor]:
                no_deps.append(neighbor)

    if len(sorted_steps) != len(sel_steps):
        raise ValueError("Cycle detected in pipeline steps.")

    return sorted_steps

def detect_reachable_from_steps(steps, starting_steps):
    # Map: file → steps that consume it
    file_to_consumers = defaultdict(set)
    step_inputs = {}
    step_outputs = {}

    for name, step in steps.items():
        inputs = set(step.get("input", []))
        outputs = set(step.get("output", []))
        step_inputs[name] = inputs
        step_outputs[name] = outputs
        for f in inputs:
            file_to_consumers[f].add(name)

    # Initialize queue with outputs from starting steps
    reachable_steps = set(starting_steps)
    produced_files = set()
    queue = deque(starting_steps)

    while queue:
        step_name = queue.popleft()
        outputs = step_outputs.get(step_name, [])
        produced_files.update(outputs)

        # For each file, find consumer steps that now might be reachable
        for f in outputs:
            for consumer in file_to_consumers.get(f, []):
                if consumer in reachable_steps:
                    continue
                # If all inputs to this step are satisfied by reachable outputs
                if step_inputs[consumer].issubset(produced_files):
                    reachable_steps.add(consumer)
                    queue.append(consumer)

    return reachable_steps


def find_disconnected_pipelines(steps, sel_steps):
    from collections import defaultdict, deque

    # Build undirected connectivity graph (ignore direction for grouping)
    graph = defaultdict(set)
    for name in sel_steps:
        inputs = set(steps[name].get("input", []))
        outputs = set(steps[name].get("output", []))
        for other in sel_steps:
            if other == name:
                continue
            other_inputs = set(steps[other].get("input", []))
            other_outputs = set(steps[other].get("output", []))
            # Connect if they share input/output
            if inputs & other_outputs or outputs & other_inputs:
                graph[name].add(other)
                graph[other].add(name)

    # BFS to find connected components
    visited = set()
    components = []

    for step in sel_steps:
        if step not in visited:
            queue = deque([step])
            component = []
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    queue.extend(graph[node])
            components.append(component)

    sorted_pipelines = [topological_sort(steps, comp) for comp in components]

    return sorted_pipelines

def build_proc_graph(steps, sel_steps, list_inputs=None):
    '''
    Creates a graph from process definitions
    '''    
    list_inputs = set(list_inputs or [])
    all_required_steps = set()
    all_required_files = set()
    
    # Build reverse index: which step produces each file?
    file_to_step = {}
    for step_name, step in steps.items():
        for f in step.get("output", []):
            file_to_step[f] = step_name

    # BFS from selected steps to collect required steps and inputs
    queue = deque(sel_steps)
    while queue:
        step_name = queue.popleft()
        if step_name in all_required_steps:
            continue
        all_required_steps.add(step_name)

        for f in steps[step_name].get("input", []):
            all_required_files.add(f)
            if f not in list_inputs and f in file_to_step:
                producing_step = file_to_step[f]
                if producing_step not in all_required_steps:
                    queue.append(producing_step)

    # Build the graph
    dot = Digraph()
    dot.attr(rankdir='TB')
    file_roles = get_file_roles(steps)

    for step_name in all_required_steps:
        step = steps[step_name]
        color = 'lightblue' if step_name in sel_steps else 'lightgray'
        dot.node(step_name, shape='box', style='filled', fillcolor=color)

        for f in step.get("input", []):
            color = get_file_color(file_roles.get(f, "input"))
            dot.node(f, shape='ellipse', style='filled', fillcolor=color)
            dot.edge(f, step_name)

        for f in step.get("output", []):
            color = get_file_color(file_roles.get(f, "output"))
            dot.node(f, shape='ellipse', style='filled', fillcolor=color)
            dot.edge(step_name, f)

    # Add any unused starting inputs
    for f in list_inputs:
        if f not in file_to_step:
            dot.node(f, shape='ellipse', style='filled', fillcolor='lightgreen')

    return dot

# Generate CLI pipeline command
def generate_pipeline_command(steps, step_list):
    cmds = []
    for name in step_list:
        step = steps[name]
        params = " ".join(f"--{k} {v}" for k, v in step.get("parameters", {}).items())
        cmd = f"{name}.py {params}"
        cmds.append(cmd)
    return " | \\\n  ".join(cmds)

