from setuptools import find_packages, setup


setup(
    name="redrob_ranker",
    version="0.1.0",
    description="Candidate ranking toolkit for the Redrob Intelligent Candidate Discovery challenge.",
    author="Redrob Challenge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "redrob-ranker=redrob_ranker.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
