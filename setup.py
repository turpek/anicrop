import os
import sys

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

pyx_blend = "src/anicrop/native/blend.pyx"
pyx_color = "src/anicrop/native/color.pyx"
c_blend = "src/anicrop/native/blend.c"
c_color = "src/anicrop/native/color.c"

has_pyx = os.path.exists(pyx_blend) and os.path.exists(pyx_color)
has_c = os.path.exists(c_blend) and os.path.exists(c_color)

require_native = os.environ.get("ANICROP_REQUIRE_NATIVE", "0") == "1"
if require_native and not (USE_CYTHON and has_pyx) and not has_c:
    raise RuntimeError(
        "ANICROP_REQUIRE_NATIVE=1 foi definido, mas nem Cython com .pyx nem arquivos .c foram encontrados."
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

if USE_CYTHON and has_pyx:
    extensions = [
        Extension(
            "anicrop.native.blend",
            sources=[pyx_blend],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
        Extension(
            "anicrop.native.color",
            sources=[pyx_color],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
    ]
    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
elif has_c:
    ext_modules = [
        Extension(
            "anicrop.native.blend",
            sources=[c_blend],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
        Extension(
            "anicrop.native.color",
            sources=[c_color],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=not require_native,
        ),
    ]

setup(
    ext_modules=ext_modules,
)
