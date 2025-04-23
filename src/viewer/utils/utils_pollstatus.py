import streamlit as st
import boto3
import docker
import random
import json
from time import sleep

from abc import ABC, abstractmethod

class TaskHandle(ABC):
    @abstractmethod
    def status(self) -> str:
        """Returns the current status of the task."""
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Returns True if the task resource still exists."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up the task if needed (e.g. removes container or job entry)."""
        pass

    @abstractmethod
    def get_id(self) -> str:
        """Returns the underlying ID or name of the task."""
        pass

    @abstractmethod
    def get_logs(self) -> str:
        """Returns the available logs for the task."""
        pass

import docker
import boto3
import botocore
class DummyHandle(TaskHandle):
    def __init__(self, task_name: str):
        self.task_name = task_name

    def status(self) -> str:
        return "dummystatus"
    
    def exists(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass

    def get_id(self) -> str:
        return self.task_name
    
    def get_logs(self):
        return "placeholder logs"
    
class DockerContainerHandle(TaskHandle):
    def __init__(self, container_name: str):
        self.container_name = container_name
        self.client = docker.from_env()

    def status(self) -> str:
        try:
            container = self.client.containers.get(self.container_name)
            return container.status
        except docker.errors.NotFound:
            return "deleted"

    def exists(self) -> bool:
        try:
            self.client.containers.get(self.container_name)
            return True
        except docker.errors.NotFound:
            return False

    def cleanup(self) -> None:
        try:
            container = self.client.containers.get(self.container_name)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass

    def get_id(self) -> str:
        return self.container_name
    
    def get_logs(self) -> str:
        return _get_docker_logs(self.container_name)


@st.cache_data(ttl=10)
def _get_docker_logs(container_name: str) -> str:
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        return container.logs(tail=100).decode()
    except docker.errors.NotFound:
        return "[Container not found]"
    except Exception as e:
        return f"[Error fetching logs: {str(e)}]"


class BatchJobHandle(TaskHandle):
    def __init__(self, job_id: str, region: str = "us-east-1"):
        self.job_id = job_id
        self.client = boto3.client("batch", region_name=region)

    def status(self) -> str:
        response = self.client.describe_jobs(jobs=[self.job_id])
        if not response["jobs"]:
            return "deleted"
        return response["jobs"][0]["status"]

    def exists(self) -> bool:
        response = self.client.describe_jobs(jobs=[self.job_id])
        return bool(response["jobs"])

    def cleanup(self) -> None:
        # AWS Batch jobs can't be deleted, but you could cancel if running
        try:
            self.client.cancel_job(jobId=self.job_id, reason="Manual cleanup")
        except self.client.exceptions.ClientError:
            pass

    def get_id(self) -> str:
        return self.job_id
    
    def get_logs(self) -> str:
        return _get_batch_logs(self.job_id, self.region)


@st.cache_data(ttl=15)
def _get_batch_logs(job_id: str, region: str = "us-east-1") -> str:
    batch_client = boto3.client("batch", region_name=region)
    logs_client = boto3.client("logs", region_name=region)

    job_detail = batch_client.describe_jobs(jobs=[job_id])["jobs"][0]
    log_stream = job_detail.get("container", {}).get("logStreamName")
    if not log_stream:
        return "[No log stream available yet]"

    logs = logs_client.get_log_events(
        logGroupName="/aws/batch/job",
        logStreamName=log_stream,
        startFromHead=False,
        limit=100
    )
    return "\n".join(event["message"] for event in logs["events"])
    


def get_handle(mode: str, raw_id: str) -> TaskHandle:
    if mode == "docker":
        return DockerContainerHandle(raw_id)
    elif mode == "batch":
        return BatchJobHandle(raw_id)
    elif mode == 'dummy':
        return DummyHandle(raw_id)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    

def parse_lambda_response(response_str: str) -> TaskHandle:
    try:
        response = json.loads(response_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from Lambda: {response_str}")

    if "error" in response:
        raise RuntimeError(f"Lambda error: {response['error']}")

    if "mode" not in response or "id" not in response:
        raise ValueError(f"Missing fields in Lambda response: {response}")

    mode = response["mode"]
    task_id = response["id"]

    if mode == "docker":
        return DockerContainerHandle(task_id)
    elif mode == "batch":
        return BatchJobHandle(task_id)
    else:
        raise ValueError(f"Unknown task mode '{mode}' in Lambda response")

def add_job_to_session(job_handle):
    if 'active_jobs' not in st.session_state:
        st.session_state.active_jobs = {}
    st.session_state.active_jobs[job_handle.get_id()] = job_handle


# --- Setup AWS and Docker clients ---
batch_client = boto3.client("batch", region_name="us-east-1")  # adjust region
try:
    docker_client = docker.from_env()
except:
    print("DEBUG: Could not initialize local Docker client.")

# --- Map status to colors ---
STATUS_COLOR_MAP = {
    "NOT READY": "gray",
    "SUBMITTED": "yellow",
    "PENDING": "yellow",
    "RUNNING": "blue",
    "SUCCEEDED": "green",
    "FAILED": "red",
}

# --- Render colored status badge ---
def render_status_indicator(status: str):
    color = STATUS_COLOR_MAP.get(status.upper(), "black")
    st.markdown(
        f"<span style='background-color:{color}; color:white; padding:0.25em 0.5em; "
        f"border-radius:0.25em; font-weight:bold'>{status.upper()}</span>",
        unsafe_allow_html=True
    )



# --- Poll status ---
def poll_job_status(job):
    try:
        if job["type"] == "docker":
            container = docker_client.containers.get(job["id"])
            status = container.status.upper()
            if status == "EXITED":
                return "SUCCEEDED"
            return status
        elif job["type"] == "batch":
            response = batch_client.describe_jobs(jobs=[job["id"]])
            status = response["jobs"][0]["status"].upper()
            return status
        else:
            return "UNKNOWN"
    except Exception as e:
        return "FAILED" if "not found" in str(e).lower() else "UNKNOWN"

# --- Update all jobs ---
def update_all_statuses():
    for job in st.session_state.jobs:
        if job["status"] not in {"SUCCEEDED", "FAILED"}:
            job["status"] = poll_job_status(job)