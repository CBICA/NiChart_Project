from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='spare_scores',
      version='1.2.0',
      description='Compute characteristic brain signatures of your case population.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Gyujoon Hwang, George Aidinis',
      author_email='ghwang1106@gmail.com, aidinisg@pennmedicine.upenn.edu',
      license='https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html',
      packages=find_packages(),
      package_data={'spare_scores':['mdl/*.pkl.gz','data/*.csv']},
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'scikit-learn', 'torch==1.11', 'matplotlib', 'optuna'],
      entry_points={
        'console_scripts': ['spare_score=spare_scores.cli:main']
        },
      )