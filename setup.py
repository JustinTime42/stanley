"""Setup configuration for agent-swarm package."""

from setuptools import setup, find_packages

setup(
    name="agent-swarm",
    version="0.1.0",
    description="Multi-agent workflow system with LangGraph",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "redis[hiredis]>=5.0.0",
        "qdrant-client>=1.7.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-qdrant>=0.1.0",
        "langgraph>=0.0.20",
        "langgraph-checkpoint>=0.0.1",
        "openai>=1.0.0",
        "tiktoken>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "black>=23.0.0",
        ],
    },
)
