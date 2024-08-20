from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='NiChartHarmonize',
        version='2.2.0',
        description='Harmonization tools for multi-center neuroimaging studies.',
	long_description=long_description,
	long_description_content_type='text/markdown',
        url='https://github.com/rpomponiohttps://github.com/gurayerus/NiChartHarmonize',
        author='Guray Erus',
        author_email='guray.erus@pennmedicine.upenn.edu',
        license='MIT',
        packages=['NiChartHarmonize'],
        install_requires=['numpy', 'pandas', 'nibabel', 'statsmodels>=0.12.0'],
        entry_points={
            'console_scripts': ['neuroharm=NiChartHarmonize.nh_cli:main']
            },
        zip_safe=False)
