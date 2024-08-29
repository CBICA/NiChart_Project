"""Setup tool for NiChart_Tissue_Segmentation."""
from setuptools import setup, find_packages
import setuptools
from pathlib import Path
import io
import os

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def read(*paths, **kwargs):
    """Read the contents of a text file safely."""

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(name='NiChart_Tissue_Segmentation',
    version=read("NiChart_Tissue_Segmentation", "VERSION"),
    description='Run NiChart_Tissue_Segmentation on your data(currently only structural pipeline is supported).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ashish Singh, Guray Erus, George Aidinis',
    author_email='software@cbica.upenn.edu',
    license='MIT',
    url="https://github.com/CBICA/NiChart_Tissue_Segmentation",
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        'console_scripts': [
            'NiChart_Tissue_Segmentation = NiChart_Tissue_Segmentation.__main__:main'
        ]        
    },    
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix',
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,	
      )
