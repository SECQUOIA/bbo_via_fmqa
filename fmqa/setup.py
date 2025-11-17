import setuptools

setuptools.setup(
    name="fmqa",
    version="0.0.1",
    author="Koki Kitai",
    author_email="kitai.koki@gmail.com",
    description="Trainable Binary Quadratic Model based on Factorization Machine",
    license="MIT",
    packages=["fmqa"],
    install_requires=[
        "dimod",
        "torch>=2.0.0",
        "numpy>=1.15.0",
    ]
)
