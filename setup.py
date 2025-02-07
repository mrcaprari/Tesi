from setuptools import find_packages, setup

setup(
    name="wololo",
    version="0.2.0",
    author="mrcaprari",
    description="A PyTorch Framework for Probabilistic Model Conversion",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
    ],
    extras_require={
        "test": [
            "matplotlib",
            "numpy",
            "tqdm",
        ],
    },
)
