#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

import re
import time
from pathlib import Path

MAJOR = 0
MINOR = 1
PATCH = 0
SUFFIX = ""
if PATCH != "":
    VERSION = "{}.{}.{}{}".format(MAJOR, MINOR, PATCH, SUFFIX)
else:
    VERSION = "{}.{}{}".format(MAJOR, MINOR, SUFFIX)

VERSION_FILE = "text2video/version.py"


def write_version_py():
    """
    Write the version of text2video module into a file.
    """
    content = f"""# Generated Version File
# Time: {time.asctime()}
__version__ = '{VERSION}'
"""
    Path(VERSION_FILE).write_text(content)


def get_version():
    """
    Reads the module version from the standard version file considering
    the pattern "__version__ = 'current_version'", where current_version is
    considered the module version.

    Returns:
        (string) text2video current version

    """
    result = re.search(r"'(.*?)'", Path(VERSION_FILE).read_text())
    if result is not None:
        locals()["__version__"] = result.group(0).replace("'", "")
    return locals()["__version__"]


if __name__ == "__main__":
    write_version_py()
    setup(
        name="text2video",
        author="Karina Bogdan",
        author_email="karina.bogdan@gmail.com",
        version=get_version(),
        description="Text to video generation by feature alignment",
        keywords="text2video, text to video",
        packages=find_packages(),
        install_requires=[],
        # python_requires=">=3.9",
    )
