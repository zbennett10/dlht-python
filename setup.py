from setuptools import setup, find_packages

setup(
    name="dlht",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "sortedcontainers>=2.4.0",
    ],
)
