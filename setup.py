import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
from codecs import open

if sys.version_info[:3] < (3, 0, 0):
    print("Requires Python 3 to run.")
    sys.exit(1)

with open("README.md", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="saplings",
    description="Build smarter AI agents with tree search",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="v6.2.0",
    packages=find_packages(),
    python_requires=">=3",
    url="https://github.com/shobrook/saplings",
    author="shobrook",
    author_email="shobrookj@gmail.com",
    # classifiers=[],
    install_requires=[
        "json-repair",
        "litellm",
    ],
    keywords=[
        "llm",
        "agent",
        "openai",
        "search-algorithm",
        "search",
        "ai",
        "monte carlo tree search",
        "mcts",
        "a*",
        "greedy",
    ],
    license="MIT",
)
