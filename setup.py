"""
Setup configuration for building C++ extensions
Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from pathlib import Path
import setuptools
import torch
from torch.utils import cpp_extension

class get_pybind_include:
    """Helper for getting pybind11 include directory"""
    def __str__(self):
        return cpp_extension.include_paths()[0]


ROOT = Path(__file__).parent
ext_modules = []

mamba_src = ROOT / "mamba_kernel.cpp"
if mamba_src.exists():
    ext_modules.append(
        Extension(
            "mamba_kernel",
            [str(mamba_src)],
            include_dirs=[
                get_pybind_include(),
                torch.utils.cpp_extension.include_paths()[0],
            ],
            language="c++",
            extra_compile_args=["-O3"],
        )
    )

setup(
    name='mlsystem',
    version='1.0.0',
    author='ML System Contributors',
    description='Hierarchical Mamba + Transformer ML System',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
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
