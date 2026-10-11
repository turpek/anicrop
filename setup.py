import os
import sys

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

has_c_sources = (
    os.path.exists("src/anicrop/native/blend.c")
    and os.path.exists("src/anicrop/native/color.c")
)

require_native = os.environ.get("ANICROP_REQUIRE_NATIVE", "0") == "1"
if require_native and not USE_CYTHON and not has_c_sources:
    raise RuntimeError(
        "ANICROP_REQUIRE_NATIVE=1 foi definido, mas Cython não está disponível e os arquivos .c não foram encontrados."
    )


def get_compile_and_link_args() -> tuple[list[str], list[str]]:
    """Retorna os argumentos de compilação e linkedição para a plataforma atual."""
    if os.name == "nt":
        return ["/O2", "/openmp"], []

    args = ["-O3"]
    link: list[str] = []

    # Apple Clang no macOS não inclui suporte nativo a OpenMP por padrão
    if sys.platform != "darwin":
        args.append("-fopenmp")
        link.append("-fopenmp")

    # Flag opcional para builds locais de desenvolvimento que buscam instruções específicas da CPU
    if os.environ.get("ANICROP_NATIVE") == "1":
        args.append("-march=native")

    return args, link


extra_compile_args, extra_link_args = get_compile_and_link_args()
ext_modules = []

if USE_CYTHON:
    extensions = [
        Extension(
            "anicrop.native.blend",
            sources=["src/anicrop/native/blend.pyx"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
        Extension(
            "anicrop.native.color",
            sources=["src/anicrop/native/color.pyx"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
    ]
    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
elif has_c_sources:
    ext_modules = [
        Extension(
            "anicrop.native.blend",
            sources=["src/anicrop/native/blend.c"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
        Extension(
            "anicrop.native.color",
            sources=["src/anicrop/native/color.c"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
    ]

setup(
    ext_modules=ext_modules,
)
