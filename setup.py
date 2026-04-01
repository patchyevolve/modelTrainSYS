"""
Setup configuration for building C++ extensions
Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import torch
from torch.utils import cpp_extension

class get_pybind_include:
    """Helper for getting pybind11 include directory"""
    def __str__(self):
        return cpp_extension.include_paths(torch=True)[0]


# C++ Extensions
ext_modules = [
    Extension(
        'mlsystem.cpp.mamba_kernel',
        ['mlsystem/cpp/mamba_kernel.cpp'],
        include_dirs=[
            get_pybind_include(),
            torch.utils.cpp_extension.include_paths(torch=True)[0]
        ],
        language='c++',
        extra_compile_args=['-O3', '-march=native'],
    ),
    Extension(
        'mlsystem.cpp.reflector_kernel',
        ['mlsystem/cpp/reflector_kernel.cpp'],
        include_dirs=[
            get_pybind_include(),
            torch.utils.cpp_extension.include_paths(torch=True)[0]
        ],
        language='c++',
        extra_compile_args=['-O3', '-march=native'],
    ),
]

setup(
    name='mlsystem',
    version='1.0.0',
    author='ML System Contributors',
    description='Hierarchical Mamba + Transformer ML System',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
        'torch>=2.0.0',
        'numpy',
        'Pillow',
        'pybind11>=2.6.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
