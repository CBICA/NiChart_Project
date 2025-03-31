"""
This file contains logic for submitting containerized jobs.
These jobs can run either locally (local Docker daemon installation),
or via cloud service (AWS Batch).
Once infrastructure has been set up on a cloud provider,
a corresponding branching can be added here to handle the necessary dispatch
as well as any details needed for monitoring. 
Author: Alexander Getka
"""
import boto3
import docker
import json
import pathlib
import streamlit as st
import uuid
import yaml

# Load application lookup table. Should be consistent with the one on cloud provider.
path_to_app_config = pathlib.Path(__file__).parent / "app_config.yaml"
with open(path_to_app_config, "r") as file:
    app_config = yaml.safe_load(file)

def submit_job(app_name: str, inputs: dict, outputs: dict, **kwargs) -> str:
    """
    Submits a job either locally via Docker or to AWS Batch via Lambda.
    
    Returns a job ID (Docker container ID or AWS Batch Job ID).
    """

    if app_name not in app_config:
        raise ValueError(f"Unknown app: {app_name}")

    container_tag = app_config[app_name]["container"]
    entrypoint = app_config[app_name]["entrypoint"]

    if st.session_state["has_cloud_session"]:
        return submit_aws_batch_job(container_tag, entrypoint, inputs, outputs, **kwargs)
    else:
        return run_docker_job(container_tag, entrypoint, inputs, outputs, **kwargs)
    

def run_docker_job(container_tag, entrypoint, inputs, outputs, **kwargs):
    client = docker.from_env()
    job_id = f"local-{uuid.uuid4().hex[:8]}"

    volumes = kwargs.get("volumes", {})  # {"/host/path": {"bind": "/container/path", "mode": "rw"}}

    container = client.containers.run(
        image=container_tag,
        command=[entrypoint] + kwargs.get("cmd_args", []),
        volumes=volumes,
        detach=True,
        environment=kwargs.get("env", {}),
        name=job_id,
    )
    return container.id  # or job_id if you want your own tracking

def submit_aws_batch_job(container_tag, entrypoint, inputs, outputs, **kwargs):
    payload = {
        "container_tag": container_tag,
        "entrypoint": entrypoint,
        "inputs": inputs,
        "outputs": outputs,
        "extra": kwargs
    }
    
    lambda_client = PLACEHOLDER!!!

    response = lambda_client.invoke(
        FunctionName="submit_batch_job_lambda",
        InvocationType="RequestResponse",  # or "Event" if fully async
        Payload=json.dumps(payload)
    )
    
    result = json.load(response["Payload"])
    return result["job_id"]