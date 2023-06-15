# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-15 12:16:51
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 12:23:23
import sys
import subprocess

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools import find_packages, setup, Extension


class Build(build_ext):
    """Customized setuptools build command - builds protos on build."""

    def run(self):
        protoc_command = ["make"]
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)
        super().run()


class BuildPy(build_py):
    """Customized setuptools build command - builds protos on build."""

    def run(self):
        protoc_command = ["make"]
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)
        super().run()


setup(
    name="lfr",
    version="0.0",
    description="LFR benchmark for networks",
    packages=["lfr"],
    has_ext_modules=lambda: True,
    cmdclass={
        "build_ext": Build,
    },
)
