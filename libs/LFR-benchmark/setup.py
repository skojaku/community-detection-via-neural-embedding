import sys
import subprocess

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


class Build(build_ext):
    def run(self):
        if subprocess.call(["make"]) != 0:
            sys.exit(-1)
        super().run()


class BuildPy(build_py):
    def run(self):
        if subprocess.call(["make"]) != 0:
            sys.exit(-1)
        super().run()


setup(
    has_ext_modules=lambda: True,
    cmdclass={
        "build_ext": Build,
        "build_py": BuildPy,
    },
)
