from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager

from anicrop.interfaces.container import AbstractContainer
from anicrop.interfaces.layer import AbstractBaseLayer


def walk_nodes(
    root: AbstractBaseLayer | AbstractContainer | Iterable[AbstractBaseLayer],
) -> Generator[AbstractBaseLayer, None, None]:
    """Gera uma travessia preguiçosa de todos os nós a partir de uma raiz ou coleção (DFS)."""
    if isinstance(root, AbstractBaseLayer):
        yield root

    if isinstance(root, (AbstractContainer, Iterable)) and not isinstance(
        root, (str, bytes)
    ):
        for child in root:
            yield from walk_nodes(child)


@contextmanager
def freeze_geometry(
    container: AbstractContainer | Iterable[AbstractBaseLayer],
) -> Generator[None, None, None]:
    """Congela temporariamente o cálculo de matrizes e regiões com snapshot sob demanda para todos os nós."""

    def _toggle_freeze(enable: bool) -> None:
        for node in walk_nodes(container):
            for strategy in (node.base, node.frame):
                strategy._cached_matrix = None
                strategy._cached_region = None
                strategy._cached_global_region = None
                strategy._resolve_matrix = (
                    strategy._lazy_matrix if enable else strategy._direct_matrix
                )
                strategy._resolve_region = (
                    strategy._lazy_region if enable else strategy._direct_region
                )
                strategy._resolve_global_region = (
                    strategy._lazy_global_region
                    if enable
                    else strategy._direct_global_region
                )

    _toggle_freeze(True)
    try:
        yield
    finally:
        _toggle_freeze(False)
