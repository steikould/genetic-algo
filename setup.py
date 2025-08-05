from setuptools import setup, find_packages

setup(
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'deap',
        'pydantic',
        'matplotlib',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'ga-optimizer=ga_optimizer.cli.main:main',
        ],
    },
)
