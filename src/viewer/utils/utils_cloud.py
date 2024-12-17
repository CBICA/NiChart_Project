import requests
import json
import os

def load_cloud_config(config_file: str):
    """
    Load Lambda URLs from a local JSON file.
    """
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing the configuration file '{config_file}'.")

def update_stats_db(user_id, job_type, count):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Traverse up to the top-level directory (assuming 'config.json' is in the repo root)
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".." ))  
    print(f"repo root: {repo_root}")

    # Construct the full path to the config file
    config_file = os.path.join(repo_root, "cloud-config.json")
    cloud_config = load_cloud_config(config_file)

    lambda_urls = cloud_config.get("lambda_urls", None)
    if lambda_urls is None:
        print("Lambda URL not provided in cloud config!")

    update_stats_url = lambda_urls["update_stats"]
    
    payload = {
        "user_id": user_id,
        "job_type": job_type,
        "count": count
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(update_stats_url, json=payload, headers=headers)
        response.raise_for_status()
        print("Success:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error:", e)
