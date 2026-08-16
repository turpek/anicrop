import os
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

extra_compile_args = ["-O3", "-mavx2"] if os.name != "nt" else ["/O2", "/arch:AVX2"]

ext_modules = []

if USE_CYTHON:
    ext_modules += cythonize(
        Extension(
            "anicrop.native.blend",
            sources=["src/anicrop/native/blend.pyx"],
            extra_compile_args=extra_compile_args,
        ),
        compiler_directives={"language_level": "3"},
    )

setup(
    ext_modules=ext_modules,
)
