from setuptools import setup, find_packages, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fmri_pnet",
    version="1.0.1a",
    description="pNet: a python package for computing personalized fucntional networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MLDataAnalytics/pNet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages = find_namespace_packages(where='src'), 
    package_dir = {"": "src"},
    package_data= {
    "pnet.Report": ["*.html"],
    "pnet.Color": ["*.mat"],
    "pnet.Brain_Template.FreeSurfer_fsaverage5": ["*.mat", "*.zip", "*.log", "*.gz"],
	"pnet.Brain_Template.HCP_Surface": ["*.mat", "*.zip", "*.log", "*,gz"], 
 	"pnet.Brain_Template.HCP_Surface_Volume": ["*.mat", "*.zip", "*.log", "*.gz"],
	"pnet.Brain_Template.HCP_Volume": ["*.mat", "*.zip", "*.log", "*.gz"],
	"pnet.Brain_Template.MNI_Volume": ["*.mat", "*.zip", "*.log", "*.gz"],
    "pnet.examples":["*.toml", "*.txt"]
    },
    include_package_data = True,
    install_requires=[
        'ggplot',
        'matplotlib', 
        'plotnine',
        'statsmodels',
        'surfplot',
        'tomli'
    ],
)
