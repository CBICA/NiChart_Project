from pathlib import Path

import setuptools
from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="NiChart_DLMUSE",
    version="0.1.7",
    description="Run NiChart_DLMUSE on your data(currently only structural pipeline is supported).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ashish Singh, Guray Erus, George Aidinis",
    author_email="software@cbica.upenn.edu",
    license="MIT",
    url="https://github.com/CBICA/NiChart_DLMUSE",
    install_requires=[
        'torch<2.1; platform_system=="Darwin"',  # macOS
        'torch<2.1; platform_system!="Darwin"',  # Linux and Windows
        "nnunet==1.7.1",
        "tqdm",
        "dicom2nifti",
        "scikit-image>=0.14",
        "medpy",
        "scipy",
        "batchgenerators>=0.23",
        "requests",
        "tifffile",
        "setuptools",
        "nipype",
        "matplotlib>=3.3.3",
        "dill>=0.3.4",
        "h5py",
        "hyperopt==0.2.5",
        "keras==2.6.0",
        "numpy",
        "protobuf==3.17.3",
        "pymongo==3.12.0",
        "scikit-learn",
        "nibabel==3.2.1",
        "resource==0.2.1",
        "networkx>=2.5.1",
        "pandas==1.2.5",
        "pathlib",
    ],
    entry_points={"console_scripts": ["NiChart_DLMUSE = NiChart_DLMUSE.__main__:main"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
