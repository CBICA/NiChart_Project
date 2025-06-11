from typing import Dict, List, Union, Optional, Any
from pydantic import BaseModel, Field, validator
import os
import yaml
from pathlib import Path
import subprocess
import streamlit as st
import json
import boto3
from botocore.exceptions import ClientError
import utils.utils_pollstatus as ps
import time

DEFAULT_TOOL_DEFINITION_PATH = Path(__file__).parent.parent.parent.parent / "resources/tools/"


def is_safe_path(base_dir: Union[str, Path], target_path: Union[str, Path]) -> bool:
    """
    Check that the target_path is a subpath of base_dir and doesn't escape it via symlinks or traversal.
    
    Parameters:
        base_dir: The base directory within which target_path must reside.
        target_path: The user-supplied path to validate.
        
    Returns:
        True if the path is valid and contained within base_dir, False otherwise.
    """
    base_dir = Path(base_dir).resolve(strict=False)
    target_path = Path(target_path).resolve(strict=False)

    try:
        target_path.relative_to(base_dir)
        return True
    except ValueError:
        return False

class IOField(BaseModel):
    type: str  # "file" or "directory"
    description: Optional[str] = None


class MountConfig(BaseModel):
    path_in_container: str
    mode: str = Field("ro", pattern="^(ro|rw)$")


class ResourceSpec(BaseModel):
    vcpus: int
    memory: str  # e.g., "8GB"
    gpus: int = 0


class ParameterSpec(BaseModel):
    type: str  # "int", "float", "bool", "str"
    default: Optional[Union[int, float, bool, str]] = None
    choices: Optional[List[Union[int, float, str]]] = None


class ToolSpec(BaseModel):
    name: str
    description: Optional[str]
    inputs: Dict[str, IOField]
    outputs: Dict[str, IOField]
    mounts: Dict[str, MountConfig]
    resources: ResourceSpec
    container: Dict[str, Union[str, List[str]]]
    parameters: Dict[str, ParameterSpec]

    def validate_params(self, user_params: Dict[str, Union[int, float, bool, str]]) -> Dict[str, Union[int, float, bool, str]]:
        validated = {}
        for key, spec in self.parameters.items():
            if key in user_params:
                value = user_params[key]
            elif spec.default is not None:
                value = spec.default
            else:
                raise ValueError(f"Missing required parameter: {key}")

            # Type check
            if spec.type == "int" and not isinstance(value, int):
                raise TypeError(f"Parameter {key} must be int")
            elif spec.type == "float" and not isinstance(value, float):
                raise TypeError(f"Parameter {key} must be float")
            elif spec.type == "bool" and not isinstance(value, bool):
                raise TypeError(f"Parameter {key} must be bool")
            elif spec.type == "str" and not isinstance(value, str):
                raise TypeError(f"Parameter {key} must be str")

            # Choice check
            if spec.choices and value not in spec.choices:
                raise ValueError(f"Invalid value for {key}. Must be one of {spec.choices}")

            validated[key] = value
        return validated


    def generate_docker_command(self, param_values: Dict[str, Union[int, float, bool, str]], mount_paths: Dict[str, str]) -> str:
        # This line will throw if validation fails
        param_values = self.validate_params(param_values)

        # Construct any default docker args here
        global_docker_args = ['--ipc=host', '--detach']
        if self.resources.gpus != 0:
            global_docker_args.append('--gpus all')

        # Apply substitution for command template
        command_template = self.container["command"]
        for key, mount in self.mounts.items():
            command_template = command_template.replace(f"{{{key}}}", mount.path_in_container)

        for key, val in param_values.items():
            command_template = command_template.replace(f"{{{key}}}", str(val))

        # Generate mount arguments
        mount_args = []
        for label, config in self.mounts.items():
            host_path = mount_paths[label]
            mode = config.mode
            is_output = False
            is_ofile = False
            if label in self.outputs:
                is_output = True
                if self.outputs[label].type == 'file':
                    is_ofile = True
            if is_ofile: 
                parent_host_path = Path(host_path).parent.resolve()
                parent_container_path = Path(config.path_in_container).parent
                mount_args.append(f"-v {parent_host_path}:{parent_container_path}:{mode}")
            else:
                mount_args.append(f"-v {host_path}:{config.path_in_container}:{mode}")

        docker_cmd = f"docker run {' '.join(global_docker_args)} {' '.join(mount_args)} {self.container['image']} {command_template}"
        return docker_cmd


def load_tool_spec_from_yaml(yaml_path: Union[str, Path]) -> ToolSpec:
    print(f"DEBUG: Loading tool spec from yaml at path {yaml_path}")
    with open(yaml_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    return ToolSpec(**raw_data)

def ensure_and_validate_mount_paths(
    mount_paths: Dict[str, str],
    base_dir: Union[str, Path],
    input_labels: set,
    output_labels: set
) -> None:
    """
    Validate that input paths exist and are within the requested root dir.
    For outputs, ensure parent directories exist (creating them if needed).
    
    Parameters:
        mount_paths: Dictionary of host paths keyed by label.
        base_dir: Root directory to constrain access.
        input_labels: Set of input labels.
        output_labels: Set of output labels.
    
    Raises:
        FileNotFoundError or ValueError for unsafe or invalid paths.
    """
    base_dir = Path(base_dir).resolve(strict=False)

    for label, path in mount_paths.items():
        resolved = Path(path).resolve(strict=False)

        if not is_safe_path(base_dir, resolved):
            raise ValueError(f"Unsafe mount path for label '{label}': {resolved} escapes {base_dir}")

        if label in input_labels:
            if not resolved.exists():
                raise FileNotFoundError(f"Input path does not exist for '{label}': {resolved}")

        elif label in output_labels:
            parent = resolved.parent
            if not parent.exists():
                print(f"Creating parent directory for output path '{label}': {parent}")
                parent.mkdir(parents=True, exist_ok=True)


def validate_user_request(tool_name: str, user_params: Dict, user_mounts: Dict[str, str], tool_registry_path: Union[str, Path] = DEFAULT_TOOL_DEFINITION_PATH) -> str:
    # Assumes streamlit session state is set up.
    base_mount_dir = Path(st.session_state.paths["dir_out"]).resolve()
    tool_dir = Path(tool_registry_path).resolve()
    yaml_file = tool_dir / f"{tool_name}.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"Tool definition {tool_name} not found in registry at location {tool_dir}.")
    
    tool_spec = load_tool_spec_from_yaml(yaml_file)
    
    input_labels = set(tool_spec.inputs.keys())
    output_labels = set(tool_spec.outputs.keys())

    ensure_and_validate_mount_paths(user_mounts, base_mount_dir, input_labels, output_labels)

    return tool_spec.generate_docker_command(user_params, user_mounts)

def stringify_mounts(mounts_dict: Dict[str, Union[str, Path]]) -> Dict[str, str]:
    '''Utility function that converts mount dicts of Path objects to string form. 
    This is useful/necessary for JSON encoding these, which is necessary
    for cloud job submission.
    '''
    return {k: str(v) for k, v in mounts_dict.items()}


def submit_job(
    tool_name: str,
    user_params: Dict,
    user_mounts: Dict[str, str],
    id_token: str | None = None,
    execution_mode: str = "any", #"cloud", "local",
) -> Union[str, subprocess.Popen]:
    """
    Submits a job either locally or via an AWS Lambda depending on Streamlit session state.
    Assumes streamlit session state is available/initialized.
    """
    
    aws_lambda_function_name = "cbica-nichart-submitjob"
    try:
        if (st.session_state.get("has_cloud_session", False) and execution_mode.lower() == 'cloud'):
            # === CLOUD MODE ===
            print("DEBUG: Cloud mode job submission.")
            id_token = st.session_state.get("cloud_session_token", None)
            if id_token is None:
                raise ValueError("Lambda error: An ID token must be provided to submit cloud jobs and none was found in the session state.")
            payload = {
                "id_token": id_token,
                "tool_name": tool_name,
                "user_params": user_params,
                "user_mounts": user_mounts
            }

            lambda_client = boto3.client("lambda", region_name='us-east-1')
            response = lambda_client.invoke(
                FunctionName=aws_lambda_function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )

            response_payload = json.load(response['Payload'])
            print(f"Got response from Lambda: {response_payload}")
            if response.get("FunctionError"):
                return {
                    "success": False,
                    "mode": "cloud",
                    "job_id": None,
                    "handle": None,
                    "message": "Lambda function error",
                    "error": response_payload.get("errorMessage", "Unknown error")
                }

            res_job_id = response_payload.get("job_id", None)
            if res_job_id is None:
                return {
                    "success": False,
                    "mode": "cloud",
                    "job_id": None,
                    "handle": None,
                    "message": "No job ID returned from lambda",
                    "error": str(response_payload)
                }
            else:
                handle = ps.get_handle(mode='batch', raw_id=res_job_id)
                ps.add_job_to_session(handle)
                return {
                    "success": True,
                    "mode": "cloud",
                    "job_id": res_job_id,
                    "handle": handle,
                    "message": f"Added job {res_job_id}",
                    "error": None
                }

        else:
            # === LOCAL MODE ===
            print("DEBUG: Local mode job submission.")
            docker_command = validate_user_request(
                tool_name=tool_name,
                user_params=user_params,
                user_mounts=user_mounts,
            )

            print(f"Running on local docker: {docker_command}")
            # Launch container in detached mode
            result = subprocess.run(
                docker_command.split(' '),
                check=True,
                capture_output=True,
                text=True
            )
            container_id = result.stdout.strip()
            # Get the container name from inspect
            inspect_result = subprocess.run(
                ["docker", "inspect", "--format", "{{.Name}}", container_id],
                check=True,
                capture_output=True,
                text=True
            )
            container_name = inspect_result.stdout.strip().lstrip("/")
            handle = ps.get_handle(mode='docker', raw_id=container_name)
            ps.add_job_to_session(handle)
            return {
                "success": True,
                "mode": "local",
                "job_id": container_name,
                "handle": handle,
                "message": f"Added job {container_name}",
                "error": None
            }

    except FileNotFoundError as e:
        print(f"File error: {e}")
        return {
            "success": False,
            "mode": None,
            "job_id": None,
            "handle": None,
            "message": "File error",
            "error": str(e)
        }
    except ValueError as e:
        print(f"Validation error: {e}")
        return {
            "success": False,
            "mode": None,
            "job_id": None,
            "handle": None,
            "message": "Validation error",
            "error": str(e)
        }
    except TypeError as e:
        print(f"Parameter type error: {e}")
        return {
            "success": False,
            "mode": None,
            "job_id": None,
            "handle": None,
            "message": "Parameter type error",
            "error": str(e)
        }
    except ClientError as e:
        print(f"AWS Error {e.response['Error']['Message']}")
        return {
            "success": False,
            "mode": None,
            "job_id": None,
            "handle": None,
            "message": "AWS error",
            "error": e.response['Error']['Message']
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            "success": False,
            "mode": None,
            "job_id": None,
            "handle": None,
            "message": "Unexpected error",
            "error": str(e)
        }


def submit_and_run_job_sync(
    tool_name: str,
    user_params: Dict,
    user_mounts: Dict[str, str],
    id_token: str | None = None,
    execution_mode: str = "any",  # can be "cloud", "local", or "any"
    progress_bar=None,
    log=None,
    poll_interval: int = 5,
) -> Dict[str, Any]:
    
    result = submit_job(
        tool_name=tool_name,
        user_params=user_params,
        user_mounts=user_mounts,
        id_token=id_token,
        execution_mode=execution_mode,
    )

    if not result["success"]:
        # Submission failed
        error_message = f"[{result['message']}] {result['error']}"
        if log:
            log.error(error_message)
        if progress_bar:
            progress_bar.error("Job submission failed.")
        return {
            "mode": result.get("mode"),
            "status": "submission_failed",
            "error_message": error_message
        }

    # Submission succeeded
    handle = result["handle"]
    job_id = result["job_id"]
    mode = result["mode"]

    if log:
        log.info(f"Job {job_id} submitted in {mode} mode.")

    if progress_bar:
        progress_bar.text(f"Job {job_id} running...")

    # === CLOUD MODE (AWS Batch) ===
    if mode == "cloud":
        batch_client = boto3.client("batch", region_name="us-east-1")

        while True:
            try:
                response = batch_client.describe_jobs(jobs=[job_id])
                job_info = response["jobs"][0]
                status = job_info["status"]

                if log:
                    log.info(f"Batch job status: {status}")
                if progress_bar:
                    progress_bar.text(f"AWS Batch job status: {status}")

                if status in ["SUCCEEDED", "FAILED"]:
                    break

                time.sleep(poll_interval)
            except Exception as e:
                if log:
                    log.error(f"Error checking job status: {e}")
                return {
                    "mode": "cloud",
                    "status": "error",
                    "error_message": f"Error polling AWS Batch job: {e}"
                }

        return {
            "mode": "cloud",
            "status": status.lower(),
            "job_id": job_id
        }

    # === LOCAL MODE (Docker) ===
    elif mode == "local":
        while True:
            status = handle.get_status()
            if log:
                log.info(f"Docker job status: {status}")
            if progress_bar:
                progress_bar.text(f"Docker container status: {status}")

            if status in ["exited", "failed", "finished", "dead"]:
                break
            time.sleep(poll_interval)

        return {
            "mode": "local",
            "status": status,
            "job_id": job_id
        }

    else:
        # Unknown mode
        return {
            "mode": mode,
            "status": "error",
            "error_message": "Unexpected job execution mode"
        }
    