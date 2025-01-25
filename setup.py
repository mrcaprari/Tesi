from setuptools import find_packages, setup

# unpack requirements
with open("requirements.txt", "r", encoding="utf-8") as fr:
    install_requires = [
        line.strip() for line in fr if line.strip() and not line.startswith("#")
    ]


setup(
    name="wololo",
    version="0.1.0",
    author="mrcaprari",
    description="A PyTorch Framework for Probabilistic Model Conversion",
    packages=find_packages(),
    install_requires=install_requires,
)
