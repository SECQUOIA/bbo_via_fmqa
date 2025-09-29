from setuptools import setup, find_packages

setup(
    name="bbo_via_fmqa",
    version="0.1.0",
    description="A package for Black-Box Optimization via FMQA",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g.:
        # "numpy>=1.21.0",
        # "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)