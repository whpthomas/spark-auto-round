from setuptools import setup, find_packages

setup(
    name="spark-auto-round",
    version="0.14.2",
    description="Trimmed-down auto-round for CUDA, torch_compile, W4A16 on GB10",
    author="Spark Team",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "datasets",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "psutil",
        "tqdm",
        "packaging",
        "py-cpuinfo",
        "accelerate",
        "pydantic",
        "flash-linear-attention>=0.3.0",
        "causal-conv1d>=1.4.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "inference": ["gptqmodel>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "spark-auto-round=auto_round.__main__:run",
            "spark-asqa-substitute=auto_round.asqa.__main__:run",
        ],
    },
)
