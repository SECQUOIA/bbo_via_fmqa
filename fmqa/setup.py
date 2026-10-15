import setuptools

setuptools.setup(
    name="fmqa",
    version="0.0.1",
    author="Koki Kitai",
    author_email="kitai.koki@gmail.com",
    description="Trainable Binary Quadratic Model based on Factorization Machine",
    license="MIT",
    packages=["fmqa"],
    python_requires=">=3.10",
    install_requires=[
        "dimod>=0.12.0",
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ]
)
