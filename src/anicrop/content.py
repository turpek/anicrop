from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from ovld import ovld

from anicrop.enums import BlendMode, ImageFormat
from anicrop.image import Image
from anicrop.interfaces.canvas import AbstractCanvas
from anicrop.interfaces.container import AbstractBaseLayer, AbstractGroupLayer
from anicrop.interfaces.content import ContentStrategy
from anicrop.interfaces.layer import AbstractLayer
from anicrop.layout import resolve_region
from anicrop.spatial import Region
from anicrop.transform import mat_inverse, transform_vector
from anicrop.utils import walk_nodes


def resolve_crop_region(
    ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
) -> Region:
    """Converte tupla, Region, Canvas ou BaseLayer para uma instância de Region."""
    return resolve_region(ref)


class BaseContentStrategy(ContentStrategy):
    """Estratégia base de manipulação e transformação de conteúdo geométrico em elementos visuais."""

    def __init__(self, target: AbstractBaseLayer) -> None:
        self.target = target

    def resize(
        self,
        width: int,
        height: int,
    ) -> bool:
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Dimensões inválidas para resize: ({width}, {height}). Devem ser positivas."
            )

        cur_w, cur_h = self.target.global_region.size
        if cur_w <= 0 or cur_h <= 0:
            return False

        if (cur_w, cur_h) == (width, height):
            return False

        scale_x = width / cur_w
        scale_y = height / cur_h

        self.target.transform.scale(scale_x, scale_y)
        return True

    def fit(
        self,
        ref: tuple | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> bool:
        if isinstance(ref, tuple) and len(ref) == 2 and isinstance(ref[1], Region):
            ref_region = ref[1]
        else:
            ref_region = resolve_region(ref)  # type: ignore[arg-type]

        cur_region = self.target.global_region

        if cur_region == ref_region:
            return False

        if cur_region.width <= 0 or cur_region.height <= 0:
            return False

        scale_x = ref_region.width / cur_region.width
        scale_y = ref_region.height / cur_region.height

        if scale_x <= 0 or scale_y <= 0:
            return False

        self.target.transform.scale(scale_x, scale_y)

        new_global_region = self.target.global_region.align(ref_region, 0.0, 0.0)
        dx, dy = transform_vector(
            mat_inverse(self.target.parent.matrix),
            self.target.global_region,
            new_global_region,
        )
        self.target.transform.translate(dx, dy)
        return True

    def flip_x(self) -> bool:
        self.target.transform.scale(-1, 1)
        return True

    def flip_y(self) -> bool:
        self.target.transform.scale(1, -1)
        return True


class LayerContentStrategy(BaseContentStrategy):
    """Estratégia de manipulação e ajuste de conteúdo/pixels em uma camada folha (Layer)."""

    def __init__(self, target: AbstractLayer) -> None:
        super().__init__(target)
        self.target: AbstractLayer = target

    def crop(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> bool:
        crop_region = resolve_region(ref)

        if not self.target.layout.fit(crop_region):
            return False

        mask_region = self.target.global_region
        mask_image = Image.new(mask_region.size, ImageFormat.GRAY, color=255)
        self.target.add_edit(mask_image, mask_region, blend_mode=BlendMode.CLIP)
        return True


class GroupContentStrategy(BaseContentStrategy):
    """Estratégia de manipulação e ajuste de conteúdo em um grupo de camadas (GroupLayer)."""

    def __init__(self, target: AbstractGroupLayer) -> None:
        super().__init__(target)
        self.target: AbstractGroupLayer = target

    def _has_content(self) -> bool:
        """Verifica se existe ao menos uma camada folha (Layer) dentro da hierarquia do grupo."""
        return any(isinstance(node, AbstractLayer) for node in walk_nodes(self.target))

    def resize(
        self,
        width: int,
        height: int,
    ) -> bool:
        if not self._has_content():
            return False
        return super().resize(width, height)

    def fit(
        self,
        ref: tuple | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> bool:
        if not self._has_content():
            return False
        return super().fit(ref)

    def crop(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> bool:
        if not self._has_content():
            return False

        crop_region = resolve_region(ref)

        if not self.target.layout.fit(crop_region):
            return False

        mask_region = self.target.global_region
        mask_image = Image.new(mask_region.size, ImageFormat.GRAY, color=255)
        self.target.set_mask(mask_image, mask_region)
        return True


class FitContext:
    """Encapsula o par (target, ref) para cálculos proporcionais de enquadramento."""

    def __init__(
        self,
        target: AbstractBaseLayer,
        ref: tuple | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> None:
        self.target = target
        self.ref_region = resolve_crop_region(ref)  # type: ignore[arg-type]

    def fit_contain(
        self, x_factor: float = 0.5, y_factor: float = 0.5
    ) -> tuple[AbstractBaseLayer, Region]:
        """Calcula o enquadramento proporcional 'contain' e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.fit_contain(
            self.ref_region, x_factor, y_factor
        )
        return self.target, resolved

    def fit_cover(
        self, x_factor: float = 0.5, y_factor: float = 0.5
    ) -> tuple[AbstractBaseLayer, Region]:
        """Calcula o enquadramento proporcional 'cover' e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.fit_cover(
            self.ref_region, x_factor, y_factor
        )
        return self.target, resolved

    def scale_width(self) -> tuple[AbstractBaseLayer, Region]:
        """Calcula a escala proporcional ajustando à largura de ref e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.scale_width(self.ref_region.width)
        return self.target, resolved

    def scale_height(self) -> tuple[AbstractBaseLayer, Region]:
        """Calcula a escala proporcional ajustando à altura de ref e retorna (target, ref_resolvida)."""
        resolved = self.target.global_region.scale_height(self.ref_region.height)
        return self.target, resolved


class Content:
    """Motor de manipulação, transformação e ajuste de conteúdo/pixels em camadas e grupos."""

    def crop(
        self,
        target: AbstractBaseLayer,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> bool:
        return target.content.crop(ref)

    def resize(
        self,
        target: AbstractBaseLayer,
        width: int,
        height: int,
    ) -> bool:
        return target.content.resize(width, height)

    if TYPE_CHECKING:

        @overload
        def fit(
            self,
            target: AbstractBaseLayer,
            ref: tuple[float, float, float, float]
            | Region
            | AbstractCanvas
            | AbstractBaseLayer,
        ) -> bool:
            pass

        @overload
        def fit(
            self,
            payload: tuple[AbstractBaseLayer, Region],
        ) -> bool:
            pass

        def fit(self, *args: Any, **kwargs: Any) -> bool:
            pass

    else:

        @ovld
        def fit(
            self,
            target: AbstractBaseLayer,
            ref: tuple | Region | AbstractCanvas | AbstractBaseLayer,
        ) -> bool:
            return target.content.fit(ref)

        @ovld
        def fit(  # noqa: F811
            self,
            payload: tuple,
        ) -> bool:
            """Ajusta o conteúdo do elemento a partir de uma tupla (target, ref_resolvida)."""
            target, ref = payload
            return target.content.fit(ref)

    def flip_x(self, target: AbstractBaseLayer) -> bool:
        return target.content.flip_x()

    def flip_y(self, target: AbstractBaseLayer) -> bool:
        return target.content.flip_y()
