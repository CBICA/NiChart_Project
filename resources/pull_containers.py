import subprocess
import yaml
from pathlib import Path
import os

def find_yaml_files(directory: Path):
    return list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))

def extract_image_from_yaml(file_path: Path):
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
    tool_dir = Path("/tmp/nichart_static/tools")
    images = set()

    for yaml_file in find_yaml_files(tool_dir):
        image = extract_image_from_yaml(yaml_file)
        if image:
            images.add(image)

    docker_pull_images(images)

if __name__ == "__main__":
    main()
