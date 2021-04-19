import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='robopose',
    version='1.0.0',
    description='RoboPose',
    packages=find_packages(),
    ext_modules=[],
    cmdclass={}
)
