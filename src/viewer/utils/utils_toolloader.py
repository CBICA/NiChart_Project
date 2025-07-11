from typing import Dict, List, Union, Optional, Any
from pydantic import BaseModel, Field, validator
import os
import yaml
from pathlib import Path
import subprocess
import shutil
import streamlit as st
import json
import boto3
import datetime
from botocore.exceptions import ClientError
import utils.utils_pollstatus as ps
import time
import re
from collections import defaultdict, deque

DEFAULT_TOOL_DEFINITION_PATH = Path(__file__).parent.parent.parent.parent / "resources/tools/"
DEFAULT_PIPELINE_DEFINITION_PATH = Path(__file__).parent.parent.parent.parent / "resources/pipelines"


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
    memory: int  # in MB, e.g 16000 for 16GB
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

    def pull_image(self):
        image_tag = self.container["image"]
        print(f"DEBUG: Pulling image {image_tag} for local run")
        result_code = os.system(f"docker pull {image_tag}")
        if result_code != 0:
            return False # Pull failed for one reason or another
        return True # Pull success
    
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
            is_ifile = False
            if label in self.outputs:
                is_output = True
                if self.outputs[label].type == 'file':
                    is_ofile = True
            elif label in self.inputs: # necessarily this is an input
                if self.inputs[label].type =='file':
                    is_ifile = True
            else:
                print(f"Mount label {label} not found in tool spec inputs or outputs.")
                raise ValueError(f"Mount label {label} not found in tool spec inputs or outputs.")
        
            if is_ofile: 
                parent_host_path = Path(host_path).parent.resolve()
                parent_container_path = Path(config.path_in_container).parent
                mount_args.append(f"-v {parent_host_path}:{parent_container_path}:{mode}")
            elif is_ifile:
                mount_args.append(f"--mount type=bind,source={host_path},target={config.path_in_container}")
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
    tool_spec.pull_image()

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
    do_s3_cli_transfer: bool = False # True will bypass FSX to use direct S3 file upload/download
) -> Union[str, subprocess.Popen]:
    """
    Submits a job either locally or via an AWS Lambda depending on Streamlit session state.
    Assumes streamlit session state is available/initialized.
    """
    
    aws_lambda_function_name = "cbica-nichart-submitjob"
    if (execution_mode.lower() == 'cloud'):
        # === CLOUD MODE ===
        print("DEBUG: Cloud mode job submission.")
        id_token = st.session_state.get("cloud_session_token", None)
        if id_token is None:
            raise ValueError("Lambda error: An ID token must be provided to submit cloud jobs and none was found in the session state.")
        
        # Handle S3 upload, if needed
        
        if do_s3_cli_transfer:
            print("DEBUG: Syncing mount paths to S3 via AWS CLI.")
            for mount_dir in user_mounts.values():
                print(f"DEBUG: Syncing user-mount path {mount_dir}")
                cmd = f"aws s3 sync {mount_dir} s3://cbica-nichart-io/{mount_dir} --delete --exact-timestamps"
                os.system(cmd)
            print("DEBUG: Done syncing to S3 in preparation for job submission.")
        # Payload for job-submission lambda    
        payload = {
            "id_token": id_token,
            "tool_name": tool_name,
            "user_params": user_params,
            "user_mounts": user_mounts,
            "do_s3_cli_transfer": str(do_s3_cli_transfer)
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
        res_body = response_payload.get("body", None)
        if res_body is None:
            return {
                "success": False,
                "mode": "cloud",
                "job_id": None,
                "handle": None,
                "message": "No message body from Lambda",
                "error": str(response_payload)
            }
        else:
            res_body_json = json.loads(res_body)
        res_job_id = res_body_json.get("job_id", None)
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
        container_id = result.stdout.strip() # stdout from detached is just container id
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


def submit_and_run_job_sync(
    tool_name: str,
    user_params: Dict,
    user_mounts: Dict[str, str],
    id_token: str | None = None,
    execution_mode: str = "any",  # can be "cloud", "local", or "any"
    progress_bar=None,
    log=None,
    metadata_path: Path = None,
    poll_interval: int = 15,
    do_s3_cli_transfer: bool = False # True will bypass FSX to use direct S3 file upload/download
) -> Dict[str, Any]:
    
    result = submit_job(
        tool_name=tool_name,
        user_params=user_params,
        user_mounts=user_mounts,
        id_token=id_token,
        execution_mode=execution_mode,
        do_s3_cli_transfer=do_s3_cli_transfer
    )

    if not result["success"]:
        # Submission failed
        error_message = f"[{result['message']}] {result['error']}"
        if log:
            log.error(error_message)
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
        progress_bar.set_description(f"Job {job_id} running...")
    

    # === CLOUD MODE (AWS Batch) ===
    if mode == "cloud":
        batch_client = boto3.client("batch", region_name="us-east-1")

        while True:
            try:
                response = batch_client.describe_jobs(jobs=[job_id])
                job_info = response["jobs"][0]
                status = job_info["status"]

                if progress_bar:
                    progress_bar.set_description(f"Cloud job status: {status}")

                if log:
                    current_logs = handle.get_logs()
                    log.update_live(current_logs)

                if status in ["SUCCEEDED", "FAILED"]:
                    if log:
                        log.commit(current_logs)
                        log.clear_live()
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

        # Sync dirs back
       
        if do_s3_cli_transfer:
            if log:
                log.info(f"Performing post-job sync for job {job_id}.")
            print("DEBUG: Syncing from S3 to mount paths via AWS CLI.")
            for mount_path in user_mounts.values():
                #absolute_mount_path = Path(mount_path).resolve()
                print(f"DEBUG: Syncing user-mount path {mount_path}")
                cmd = f"aws s3 sync s3://cbica-nichart-io/{mount_path} {mount_path} --exact-timestamps"
                returncode = os.system(cmd)
                if returncode > 0:
                    print(f"DEBUG: Post-job sync failed!")
                    log.error(f"Post job sync failed for job {job_id}.")
                    raise RuntimeError(f"Cloud job {job_id} completed successfully, but post-job sync failed. Please submit an issue report.")
            print("DEBUG: Done syncing from S3 after job completion.")  
            if log:
                log.info(f"Done post-job sync for job {job_id}.")  

        if status.lower() == "succeeded":
            return {
                "mode": "cloud",
                "status": "success",
                "job_id": job_id,
            }
        else: 
            raise RuntimeError(f"Cloud job {job_id} failed. Please see error logs and submit an issue report.")
            return {
                "mode": "cloud",
                "status": "error",
                "job_id": job_id
            }

    # === LOCAL MODE (Docker) ===
    elif mode == "local":
        while True:
            status = handle.status()
            if log:
                current_logs = handle.get_logs()
                log.update_live(current_logs)
            if progress_bar:
                progress_bar.set_description(f"Local container job status: {status}")

            if status in ["exited", "paused", "removing", "dead"]:
                exitcode = handle.exitcode()
                log.commit(current_logs)
                log.clear_live()
                break
            time.sleep(poll_interval)

        if status.lower() == 'exited' and exitcode == 0:
            return {
                "mode": "local",
                "status": "success",
                "job_id": job_id
            }
        else: ## Job failed, fail loudly
            raise RuntimeError(f"Docker container job {job_id} failed with exit code {exitcode}")
            

    else:
        # Unknown mode
        raise RuntimeError("An unexpected job execution mode was passed. Please submit an issue report.")
        return {
            "mode": mode,
            "status": "error",
            "error_message": "Unexpected job execution mode"
        }



def resolve_vars(template: Dict[str, str], global_vars: Dict[str, str], step_outputs: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    def repl(match):
        var = match.group(1)
        if ".outputs." in var:
            step_id, key = var.split(".outputs.")
            return step_outputs[step_id][key]
        return global_vars.get(var, match.group(0))
    
    return {k: re.sub(r"\$\{([^}]+)\}", repl, v) for k, v in template.items()}

def parse_pipeline_steps(pipeline_yaml):
    steps = pipeline_yaml["steps"]
    step_map = {s["id"]: s for s in steps}
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for s in steps:
        matches = re.findall(r"\$\{(\w+)\.outputs\.(\w+)\}", yaml.dump(s.get("inputs", {})))
        for dep_id, _ in matches:
            graph[dep_id].append(s["id"])
            in_degree[s["id"]] += 1

    queue = deque([s["id"] for s in steps if in_degree[s["id"]] == 0])
    execution_order = []
    while queue:
        sid = queue.popleft()
        execution_order.append(sid)
        for neighbor in graph[sid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return execution_order, step_map

def load_metadata(metadata_path: Path) -> Dict:
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_metadata(metadata_path: Path, metadata: Dict):
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def generate_metadata_key(tool_id: str,
                          inputs: Dict,
                          params: Dict):
    def sorted_str(d: Dict) -> str:
        return json.dumps(d, sort_keys=True)
    return f"{tool_id}|{sorted_str(inputs)}|{sorted_str(params)}"

def should_skip_step(metadata_path: Path,
                     tool_id: str,
                     inputs: Dict,
                     outputs: Dict,
                     params: Dict) -> bool:
    metadata =  load_metadata(metadata_path)
    key = generate_metadata_key(tool_id, inputs, params)
    if key not in metadata:
        return False
    
    record = metadata[key]
    if record["status"] != "success":
        return False
    
    finished_time = datetime.fromisoformat(record["finished_time"])

    # Check mtime of all input files/dirs
    input_mtime = 0
    for path_str in inputs.values():
        path = Path(path_str)
        if path.is_file():
            input_mtime = max(input_mtime, path.stat().st_mtime)
        elif path.is_dir():
            mtime = max((f.stat().st_mtime for f in path.rglob('*')), default=0)
            input_mtime = max(input_mtime, mtime)

    if finished_time.timestamp() > input_mtime:
        # Copy outputs to new locations if output paths differ
        for key_out, prev_output_path in record["outputs"].items():
            new_output_path = outputs.get(key_out)
            if new_output_path and new_output_path != prev_output_path:
                src = Path(prev_output_path)
                dst = Path(new_output_path)
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        return True
    return False

def record_step_submission(metadata_path: Path,
                           tool_id: str,
                           inputs: Dict,
                           outputs: Dict,
                           params: Dict
                           ):
    metadata = load_metadata(metadata_path)
    key = generate_metadata_key(tool_id, inputs, params)
    metadata[key] = {
        "tool": tool_id,
        "inputs": inputs,
        "params": params,
        "outputs": outputs,
        "submitted_time": datetime.utcnow().isoformat(),
        "status": "pending"
    }
    save_metadata(metadata_path, metadata)

def record_step_completion(metadata_path: Path,
                           tool_id: str,
                           inputs: Dict,
                           outputs: Dict,
                           params: Dict,
                           status: str = "success"
                           ):
    metadata = load_metadata(metadata_path)
    key = generate_metadata_key(tool_id, inputs, params)
    
    entry = metadata.get(key, {
        "tool": tool_id,
        "inputs": inputs,
        "params": params,
        "outputs": outputs or {},
        "submitted_time": datetime.utcnow().isoformat()
    })

    entry["finished_time"] = datetime.utcnow().isoformat()
    entry["status"] = status
    if outputs:
        entry["outputs"] = outputs
    
    metadata["key"] = entry
    save_metadata(metadata_path, metadata)
    

def clear_all_metadata(metadata_path: Path):
    if metadata_path.exists():
        metadata_path.unlink()

def clear_step_metadata(metadata_path: Path,
                        tool_id: str,
                        inputs: Dict,
                        params: Dict):
    metadata = load_metadata(metadata_path)
    key = generate_metadata_key(tool_id, inputs, params)
    if key in metadata:
        del metadata[key]
        save_metadata(metadata_path, metadata)

def run_pipeline(pipeline_id: str,
                global_vars: Dict[str, str],
                pipeline_progress_bar=None,
                process_progress_bar=None,
                execution_mode='cloud',
                log=None,
                metadata_location=None,
                reuse_cached_steps=True,
                ):
    if metadata_location is not None:
        metadata_location = Path(metadata_location)

    # Resolve pipeline file
    pipeline_path = DEFAULT_PIPELINE_DEFINITION_PATH / f"{pipeline_id}.yaml"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline definition '{pipeline_id}' not found at {pipeline_path}")

    with open(pipeline_path, 'r') as f:
        pipeline_yaml = yaml.safe_load(f)

    log.info(f"Starting pipeline {pipeline_id}.")
    order, step_map = parse_pipeline_steps(pipeline_yaml)
    step_outputs = {}
    total_steps = len(order)
    current_step = 0
    if pipeline_progress_bar:
        pipeline_progress_bar.reset(total=total_steps)
    for sid in order:
        if process_progress_bar:
            process_progress_bar.reset(total=4)
        if pipeline_progress_bar:
            pipeline_progress_bar.update(1)
        step = step_map[sid]
        tool_id = step["tool"]

        log.info(f"Starting execution of pipeline step {tool_id}.")
        tool_yaml = DEFAULT_TOOL_DEFINITION_PATH / f"{tool_id}.yaml"
        tool = load_tool_spec_from_yaml(tool_yaml)

        # Resolve input/output paths with variable substitution
        resolved_inputs = resolve_vars(step.get("inputs", {}), global_vars, step_outputs)
        resolved_outputs = resolve_vars(step.get("outputs", {}), global_vars, step_outputs)
        resolved_total_mounts = stringify_mounts({**resolved_inputs, **resolved_outputs})
        resolved_params = step.get("params", {})
       
        print(f"Submitting job: {sid} ({tool.name})")
        if process_progress_bar:
            process_progress_bar.set_description(f"Running tool {tool_id}...")

        ## Fill this in with deduplication logic
        if metadata_location is not None:
            # Can use metadata to deduplicate pipeline steps, check it
            if not reuse_cached_steps:
                clear_step_metadata(
                    metadata_path=metadata_location,
                    tool_id=tool_id,
                    inputs=resolved_inputs,
                    params=resolved_params
                )
            elif should_skip_step(
                metadata_path=metadata_location,
                tool_id=tool_id,
                inputs=resolved_inputs,
                params=resolved_params,
                outputs=resolved_outputs
            ):
                log.info(f"[CACHE] Skipping step: {tool_id} because it was determined that a previous execution could be reused.")
                continue # Skip to next pipeline step
        
        # If we reach here, the step must be executed.
        record_step_submission(metadata_path=metadata_location,
                           tool_id=tool_id,
                           inputs=resolved_inputs,
                           outputs=resolved_outputs,
                           params=resolved_params)
        
        result = submit_and_run_job_sync(
                    tool_name=tool_id,
                    user_params=resolved_params,
                    user_mounts=resolved_total_mounts,
                    execution_mode=execution_mode,
                    progress_bar=process_progress_bar,
                    log=log,
                    metadata_path=metadata_location
        )

        if result['status'] == 'success':
            record_step_completion(metadata_path=metadata_location,
                                   tool_id=tool_id,
                                   inputs=resolved_inputs,
                                   outputs=resolved_outputs,
                                   params=resolved_params,
                                   status="success")
            print(f"Step {sid}, {tool_id} finished succesfully.")
            log.info(f"Pipeline step {tool_id} finished successfully")
        else: # Step failed, loudly fail
            record_step_completion(metadata_path=metadata_location,
                                   tool_id=tool_id,
                                   inputs=resolved_inputs,
                                   outputs=resolved_outputs,
                                   params=resolved_params,
                                   status="failure")
            log.error(f"Pipeline step {tool_id} failed with status {result['status']}.")
            print(f"Step {sid}, {tool_id} failed with status {result["status"]}, see error log:")
            print(f"Error message: {result["error_message"]}")
            raise RuntimeError(result["error_message"])
        step_outputs[sid] = resolved_outputs  # Used for future interpolation
    log.info(f"Pipeline {pipeline_id} completed successfully.")
    return step_outputs
