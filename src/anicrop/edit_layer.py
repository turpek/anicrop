from __future__ import annotations

import math

import numpy as np

from anicrop.blend import BLEND_MODE, blend_clip
from anicrop.enums import BlendMode
from anicrop.image import Image
from anicrop.spatial import Region
from anicrop.transform import (
    mat_position,
    mat_scale,
)


class EditLayer:
    """Represents a destructive edit applied to a base layer.

    Attributes:
        image: The pixel data of the edit (the 'patch').
        position: The region where this edit is applied relative to the parent layer.
        name: An optional name for the edit (useful for history/undo stacks).
        opacity: The opacity of this specific edit (0.0 to 1.0).
        blend_mode: How this edit blends with the base layer.
    """

    def __init__(
        self,
        image: Image,
        region: Region,
        matrix: np.ndarray,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = "Edit",
        visible: bool = True,
    ):
        self._image = image
        self._region = region
        self.blend_mode = blend_mode
        self.name = name
        self.visible = visible
        self._matrix = matrix
        self._lod_cache: dict[int, Image] = {}

        if self._image.is_zarr:
            self._prebuild_lod_cache()

    def _resize(self, lod_factor: float) -> Image:
        new_w = max(1, int(self._image.width * lod_factor))
        new_h = max(1, int(self._image.height * lod_factor))
        return self._image.resize((new_w, new_h))

    def _prebuild_lod_cache(self) -> None:
        """Pré-constrói a pirâmide de LODs para imagens Zarr usando a fábrica inteligente Image.resize."""
        w, h = self._image.size
        n = 1
        while True:
            lod_factor = 2.0 ** (-n)
            new_w = max(1, int(w * lod_factor))
            new_h = max(1, int(h * lod_factor))

            if new_w < 64 or new_h < 64:
                break

            self._lod_cache[n] = self._image.resize((new_w, new_h))
            n += 1

    @property
    def region(self) -> Region:
        return self._region

    @property
    def image(self) -> Image:
        return self._image

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def local_matrix(self) -> np.ndarray:
        return self.matrix @ mat_position(self.region)

    def offset(self, offset_x: int, offset_y: int) -> None:
        self._region += (offset_x, offset_y)

    def clear_lod_cache(self) -> None:
        """Limpa o cache de LOD de imagens grandes."""
        self._lod_cache.clear()

    def get_lod(self, scale_factor: float) -> tuple[Image, np.ndarray]:
        """Returns (lod_image, m_local) based on the target scale factor."""
        n = math.floor(-math.log2(scale_factor))

        if scale_factor >= 1.0 or n <= 0:
            return self._image, self.local_matrix

        lod_factor = 2.0 ** (-n)
        m_adjust = mat_scale(1.0 / lod_factor, 1.0 / lod_factor)
        m_local = self.local_matrix @ m_adjust

        if n in self._lod_cache:
            return self._lod_cache[n], m_local

        lod_image = self._resize(lod_factor)
        return lod_image, m_local

    def blend_into(
        self, layer_image: Image, edit_image: Image, dst_region: Region
    ) -> None:
        """Aplica a mesclagem padrão dos pixels da edição dentro do retângulo de destino."""
        blend = BLEND_MODE[self.blend_mode]
        blend(layer_image.view(dst_region), edit_image)


class CropEditLayer(EditLayer):
    """Edição de recorte que aplica BlendMode.CLIP e zera toda a área externa da camada."""

    def __init__(
        self,
        image: Image,
        region: Region,
        matrix: np.ndarray,
        blend_mode: BlendMode = BlendMode.CLIP,
        name: str = "Crop",
        visible: bool = True,
    ):
        super().__init__(
            image,
            region,
            matrix,
            blend_mode=BlendMode.CLIP,
            name=name,
            visible=visible,
        )

    def blend_into(
        self, layer_image: Image, edit_image: Image, dst_region: Region
    ) -> None:
        """Aplica o recorte e zera qualquer pixel fora do retângulo de destino."""
        blend_clip(layer_image.view(dst_region), edit_image)
        layer_image.clear_rect(dst_region, invert=True)


EDIT_LAYER_MAP: dict[BlendMode, type[EditLayer]] = {
    BlendMode.CLIP: CropEditLayer,
}
