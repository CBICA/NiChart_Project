from setuptools import setup, find_packages

setup(
    name="NiChart_Viewer",
    version="1.0.4",
    description="Viewer to visualize neuroimaging chart (NiChart) image descriptors and biomarkers",
    author="Guray Erus, Ashish Singh, George Aidinis",
    author_email="guray.erus@pennmedicine.upenn.edu, Ashish.Singh@pennmedicine.upenn.edu, George.Aidinis@pennmedicine.upenn.edu",
    url="https://github.com/CBICA/NiChart_Viewer",
    project_urls={
        "Documentation": "https://github.com/CBICA/NiChart_Viewer",
        "Source": "https://github.com/CBICA/NiChart_Viewer",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    packages=find_packages(include=["NiChart_Viewer", "NiChart_Viewer.*"]),
    include_package_data=True,
    package_data={
        "": ["**/*.ui", "**/*.yapsy-plugin", "*.png"],
    },
    python_requires=">=3.8,<3.11",
    install_requires=[
        "briefcase>=0.3.5,<0.4.0",
        "cycler",
        "joblib",
        "MarkupSafe",
        "matplotlib>=3.4.2,<4.0.0",
        "nibabel>=3.2.1,<4.0.0",
        "numpy>=1.21,<2.0.0",
        "pandas==2.0.1",
        "Pillow>=9.0.0,<10.0.0",
        "pyparsing>=2.4.7,<3.0.0",
        "PyQt5>=5.15.4,<6.0.0",
        "PyQt5_Qt5>=5.15.2,<6.0.0",
        "PyQt5_sip>=12.9.0,<13.0.0",
        "dill>=0.3.4,<0.4.0",
        "python_dateutil>=2.8.1,<3.0.0",
        "pytz>=2021.1,<2022.0",
        "scikit_learn>=1.0.2,<2.0.0",
        "scipy>=1.6.3,<2.0.0",
        "seaborn==0.12.2",
        "six>=1.16.0,<2.0.0",
        "statsmodels>=0.13.0,<0.14.0",
        "Yapsy>=1.12.2,<2.0.0",
        "Jinja2>=2.11.3,<3.0.0",
        "pytest==7.0.1",
        "pytest-qt==4.0.2"
    ],
    entry_points={
        "console_scripts": [
            "NiChart_Viewer=NiChart_Viewer:main",
        ],
    },
    license="MIT",
)