import time
from typing import Any

import numpy as np


def process_image(image: np.ndarray) -> None:
    time.sleep(5)


def process_images(images: list, callback: Any) -> None:
    """
    Processes a list of images sequentially and calls a callback function for each completed image.

    Args:
      images: A list of image paths.
      callback: A callback function that takes the current progress as an argument.
    """

    total_images = len(images)
    for i, image in enumerate(images):
        # Process the image here
        process_image(image)

        # Call the callback function with the progress
        progress = (i + 1) / total_images * 100
        callback(progress, i)


def my_callback(progress: float, i: int) -> Any:
    """
    A callback function that prints the progress.
    Args:
      progress: The current progress as a percentage.
    """
    print(f"Progress: {progress: .2f}%image{i}")

    return progress, i


# Example usage
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
process_images(images, my_callback)
