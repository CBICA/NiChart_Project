from pathlib import Path

import setuptools
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="NiChart",
    version="0.0.1",
    description="Investigate your neuroimaging data with high-quality automatic segmentation, machine learning-based analysis and visualization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Guray Erus, Wu Di, Kyunglok Baik, George Aidinis, Alexander Getka",
    author_email="software@cbica.upenn.edu",
    maintainer="Guray Erus, Kyunglok Baik, Spiros Maggioros, Alexander Getka",
    license="By installing/using NiChart, the user agrees to the following license: See https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html",
    url="https://github.com/CBICA/NiChart_Project",
    python_requires=">=3.8",
    install_requires=required,
    entry_points={"console_scripts": ["NiChart = src.NiChart_Viewer.entrypoint:main"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
    ],
    packages=find_packages(exclude=[".github", "assets", "docs", "resources", "test"]),
    include_package_data=True,
    keywords=[
        "deep learning",
        "neuroimaging",
        "image segmentation",
        "semantic segmentation",
        "medical image analysis",
        "medical image segmentation",
        "data analysis",
    ],
    package_data={
        "NiChart": ["VERSION"]
    },
)