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
        #protoc_command = ["make", "python"]
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
    name='belief_propagation',
    version='0.0',
    description='Python Distribution Utilities',
    packages=['belief_propagation'],
    has_ext_modules=lambda: True,
    #cmdclass={
    #    'build_ext': Build,
    #}
    cmdclass={
        'build_ext': Build,
    }
)
