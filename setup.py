"""
Setup script for SEESR with SD Turbo
"""

from setuptools import setup, find_packages

setup(
    name="seesr-sd-turbo",
    version="1.0.0",
    description="SEESR with SD Turbo for efficient super-resolution",
    author="alexgenovese",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "diffusers>=0.21.4",
        "transformers>=4.33.2",
        "accelerate>=0.22.0",
        "opencv-python>=4.8.1",
        "Pillow>=10.0.0",
        "numpy>=1.24.3",
        "scipy>=1.11.2",
        "scikit-image>=0.21.0",
        "timm>=0.9.7",
        "xformers>=0.0.21",
        "safetensors>=0.3.3",
        "pytorch-lightning>=2.0.7",
        "omegaconf>=2.3.0",
        "einops>=0.6.1",
        "huggingface_hub>=0.16.4",
        "PyWavelets>=1.4.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8", 
            "mypy",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)