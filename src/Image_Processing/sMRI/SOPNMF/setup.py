import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sopnmf",
    version="0.0.4",
    author="junhao.wen",
    author_email="junhao.wen89@email.com",
    description="Stochastic Orthogonal Projective Non-negative Matrix Factorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anbai106/SOPNMF",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'sopnmf = sopnmf.main:main',
        ],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
