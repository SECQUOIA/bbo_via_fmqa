from setuptools import setup, find_packages

setup(
    name="bbo_via_fmqa",
    version="0.1.0",
    description="A package for Black-Box Optimization via FMQA",
    author="Woosik Kim, Albert Lee, and David E. Bernal Neira",
    author_email="kim3124@purdue.edu",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        # "scipy>=1.7.0",
        "dimod>=0.1.4",
        "pandas>=1.3.0",
        "matplotlib>=3.9.2"
        "qci-client>=0.1.0",
        "eqc-models>=0.14.1",
        
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)