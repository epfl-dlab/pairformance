from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pairformance',
    version='0.0.1',
    description='Tool to perform paired evaluation of automatic systems.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["pairformance"],
    packages=['pairformance'],
    url="https://github.com/epfl-dlab/pairformance",
    author="Maxime Peyrard",
    author_email="maxime.peyrard@epfl.ch",
    install_requires=[
        "matplotlib",
        "numpy>=1.10.4",
        "pandas",
        "scipy>=1.0.1",
        "seaborn>=0.9.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)