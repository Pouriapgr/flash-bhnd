from setuptools import setup, find_packages

setup(
    name="flash_attn_bhnd",
    version="0.1.0",
    author="Pouria Sarmadi",
    description="A Triton-based Flash Attention implementation optimized for BHND memory layout.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com//flash-bhnd",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch",
        "triton",
    ],
)