import sys
from setuptools import setup, Extension

pybind11_available = False
try:
    import pybind11
    pybind11_available = True
except ImportError:
    pass

if not pybind11_available:
    print("pybind11 not installed. Install with: pip install pybind11")
    print("C++ logit processors will not be built — pure Python fallback will be used.")
    sys.exit(0)

ext = Extension(
    "logit_processors",
    sources=["logit_processors.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3", "-march=native", "-flto"],
    extra_link_args=["-flto"],
)

setup(
    name="logit_processors",
    version="1.0.0",
    description="C++ logit processors for efficient text generation",
    ext_modules=[ext],
    python_requires=">=3.8",
)
