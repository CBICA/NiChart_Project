import json
import os
from typing import Any
import boto3

import requests


def load_cloud_config(config_file: str) -> Any:
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


def update_stats_db(user_id: str, job_type: str, count: int) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Traverse up to the top-level directory (assuming 'config.json' is in the repo root)
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    print(f"repo root: {repo_root}")

    # Construct the full path to the config file
    config_file = os.path.join(repo_root, "cloud-config.json")
    cloud_config = load_cloud_config(config_file)

    lambda_urls = cloud_config.get("lambda_urls", None)
    if lambda_urls is None:
        print("Lambda URL not provided in cloud config!")

    update_stats_url = lambda_urls["update_stats"]

    payload = {"user_id": user_id, "job_type": job_type, "count": count}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(update_stats_url, json=payload, headers=headers)
        # response.raise_for_status()
        print("Success:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error:", e)


def get_credentials_from_token(id_token: str, user_pool_id: str, identity_pool_id: str, region: str):
    cognito_identity = boto3.client('cognito-identity', region_name=region)

    # Step 1: Get identity ID using token from the user pool
    identity_response = cognito_identity.get_id(
        IdentityPoolId=identity_pool_id,
        Logins={
            f'cognito-idp.{region}.amazonaws.com/{user_pool_id}': id_token
        }
    )
    identity_id = identity_response['IdentityId']

    # Step 2: Get temporary AWS credentials
    credentials_response = cognito_identity.get_credentials_for_identity(
        IdentityId=identity_id,
        Logins={
            f'cognito-idp.{region}.amazonaws.com/{user_pool_id}': id_token
        }
    )

    return credentials_response['Credentials']

def invoke_lambda_as_user(credentials, lambda_name, payload, region='us-east-1'):
    session = boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretKey'],
        aws_session_token=credentials['SessionToken'],
        region_name=region
    )

    lambda_client = session.client('lambda')
    response = lambda_client.invoke(
        FunctionName=lambda_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )

    return json.load(response['Payload'])