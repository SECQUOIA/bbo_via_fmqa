from setuptools import setup

setup(
    name="bbo_via_fmqa",
    version="0.1.0",
    description="A package for Black-Box Optimization via FMQA",
    author="Woosik Kim, Albert Lee, and David E. Bernal Neira",
    author_email="kim3124@purdue.edu",
    packages=["bbo_via_fmqa"],
    package_dir={"bbo_via_fmqa": "."},
    package_data={"bbo_via_fmqa": ["dataset/*.csv"]},
    py_modules=["read_grid", "FM_surrogate", "ising_machine", "qhd_grid_generator", "fmqa_simulated", "fmqa_qci"],
    install_requires=[
        "numpy>=1.21.0",
        "dimod>=0.12.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "fmqa>=0.0.1",
    ],
    extras_require={
        "qci": ["qci-client>=0.1.0", "eqc-models>=0.14.1"],
        "dwave": ["dwave-system"],
    },
    entry_points={"console_scripts": ["fmqa-generate-grid=qhd_grid_generator:main"]},
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
