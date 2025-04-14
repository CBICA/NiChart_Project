import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg

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

