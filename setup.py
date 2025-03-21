# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("kmv/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("kmv/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()


with open("kmv/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in kmv/__init__.py"
version: str = version_re.group(1)


setup(
    name="kmv",
    version=version,
    description="The kscale-mujoco-viewer project",
    author="Ali Kuwajerwala, Wesley Maa",
    url="https://github.com/kscalelabs/kscale-mujoco-viewer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={"dev": requirements_dev},
    packages=find_packages(),
    # entry_points={
    #     "console_scripts": [
    #         "kmv.cli:main",
    #     ],
    # },
)
