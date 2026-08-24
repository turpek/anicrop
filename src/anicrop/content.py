from __future__ import annotations
from ovld import ovld

from anicrop.canvas import Canvas
from anicrop.container import BaseLayer
from anicrop.layer import Layer
from anicrop.spatial import Region
from anicrop.layout import resolve_region


def resolve_crop_region(
    ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
) -> Region:
    """Converte tupla, Region, Canvas ou BaseLayer para uma instância de Region."""
    return resolve_region(ref)


class FitContext:
    """Encapsula o par (target, ref) e os fatores de alinhamento para cálculo proporcional."""

    def __init__(
        self,
        target: Layer,
        ref: tuple | Region | Canvas | BaseLayer,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
    ):
        self.target = target
        self.ref_region = resolve_crop_region(ref)
        self.x_factor = x_factor
        self.y_factor = y_factor

    @property
    def fit_contain(self) -> tuple[Layer, Region]:
        """Calcula o enquadramento proporcional 'contain' e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.fit_contain(self.ref_region, self.x_factor, self.y_factor)
        return self.target, resolved

    @property
    def fit_cover(self) -> tuple[Layer, Region]:
        """Calcula o enquadramento proporcional 'cover' e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.fit_cover(self.ref_region, self.x_factor, self.y_factor)
        return self.target, resolved

    @property
    def scale_width(self) -> tuple[Layer, Region]:
        """Calcula a escala proporcional ajustando à largura de ref e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.scale_width(self.ref_region.width)
        return self.target, resolved

    @property
    def scale_height(self) -> tuple[Layer, Region]:
        """Calcula a escala proporcional ajustando à altura de ref e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.scale_height(self.ref_region.height)
        return self.target, resolved


class Content:
    """Motor de manipulação, transformação e ajuste de conteúdo/pixels em camadas."""

    def crop(
        self,
        target: Layer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        return target.content.crop(ref)

    def resize(
        self,
        target: Layer,
        width: int,
        height: int,
    ) -> bool:
        return target.content.resize(width, height)

    @ovld
    def fit(
        self,
        target: Layer,
        ref: tuple | Region | Canvas | BaseLayer,
    ) -> bool:
        return target.content.fit(ref)

    @ovld  # type: ignore[no-redef]
    def fit(  # noqa: F811
        self,
        payload: tuple,
    ) -> bool:
        """Ajusta o conteúdo da camada a partir de uma tupla (target, ref_resolvida)."""
        target, ref = payload
        return target.content.fit(ref)

    def flip_x(self, target: Layer) -> bool:
        return target.content.flip_x()

    def flip_y(self, target: Layer) -> bool:
        return target.content.flip_y()
