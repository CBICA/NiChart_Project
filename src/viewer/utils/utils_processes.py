import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
from collections import defaultdict, deque

# Load YAML step definitions
def load_steps(in_dir):
    steps = {}
    for file in Path(in_dir).glob("*.yaml"):
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
    
def detect_reachable_steps(steps, list_inputs, flag_all = False):
    '''
    Creates a graph from process definitions and detect steps reachable from given input list
    '''
    file_to_producers = defaultdict(list)
    file_to_consumers = defaultdict(list)
    step_inputs = {}
    step_outputs = {}

    for name, step in steps.items():
        step_inputs[name] = set(step.get("input", []))
        step_outputs[name] = set(step.get("output", []))
        for f in step_outputs[name]:
            file_to_producers[f].append(name)
        for f in step_inputs[name]:
            file_to_consumers[f].append(name)

    # Forward propagation of input origin tracking
    file_origins = {f: {f} for f in list_inputs}
    queue = deque(list_inputs)
    while queue:
        f = queue.popleft()
        for step in file_to_consumers.get(f, []):
            if step_inputs[step].issubset(file_origins):
                origins = set().union(*(file_origins[i] for i in step_inputs[step]))
                for out in step_outputs[step]:
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
        for step in file_to_producers.get(f, []):
            if step not in needed_steps:
                needed_steps.add(step)
                needed_files.update(step_inputs[step])
                queue.extend(step_inputs[step])

    return needed_steps

def build_proc_graph(steps, selected_steps, list_inputs=None):
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
    queue = deque(selected_steps)
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
        color = 'lightblue' if step_name in selected_steps else 'lightgray'
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

