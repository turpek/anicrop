import os
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

if os.name != "nt":
    extra_compile_args = ["-O3", "-march=native", "-ffast-math", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
else:
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []

ext_modules = []

if USE_CYTHON:
    ext_modules += cythonize(
        Extension(
            "anicrop.native.blend",
            sources=["src/anicrop/native/blend.pyx"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        compiler_directives={"language_level": "3"},
    )

setup(
    ext_modules=ext_modules,
)
