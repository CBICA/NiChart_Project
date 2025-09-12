import subprocess
import yaml
from pathlib import Path
import os
import argparse

def find_yaml_files(directory):
    return list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))

def extract_image_from_yaml(file_path):
    try:
        with open(file_path, "r") as f:
            content = yaml.safe_load(f)
            if isinstance(content, dict):
                container = content.get("container", {})
                image = container.get("image")
                return image
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
    return None

def docker_pull_images(images):
    for img in sorted(images):
        print(f"Pulling: {img}")
        os.system(f'docker pull {img}')

def main():
    parser = argparse.ArgumentParser(description="Pull Docker images from YAML tool definitions.")
    parser.add_argument(
        "tool_dir",
        nargs="?",
        default=Path("/tmp/nichart_static/tools"),
        help="Path to the tools directory (defaults to /tmp/nichart_static/tools)"
    )
    args = parser.parse_args()

    tool_dir = Path(args.tool_dir)
    images = set()

    for yaml_file in find_yaml_files(tool_dir):
        image = extract_image_from_yaml(yaml_file)
        if image:
            images.add(image)

    docker_pull_images(images)

if __name__ == "__main__":
    main()
