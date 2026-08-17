from __future__ import annotations
from typing import Any

from anicrop.canvas import Canvas
from anicrop.container import BaseLayer
from anicrop.enums import BlendMode, ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.layout import LayerLayoutStrategy, resolve_region
from anicrop.spatial import Region


def resolve_crop_region(
    ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
) -> Region:
    """Converte tupla, Region, Canvas ou BaseLayer para uma instância de Region."""
    return resolve_region(ref)


class Content:
    """Motor de manipulação, transformação e ajuste de conteúdo/pixels em camadas."""

    def crop(
        self,
        target: Layer,
        region: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        """
        Aplica um corte não-destrutivo sobre o conteúdo da camada via EditLayer e BlendMode.CLIP.
        Ajusta a moldura via LayerLayoutStrategy.fit e anexa a máscara de recorte nos edits.
        """
        crop_region = resolve_crop_region(region)

        if not LayerLayoutStrategy.fit(target, crop_region):
            return False

        mask_region = target.global_region
        mask_image = Image.new(mask_region.size, ImageFormat.GRAY, color=255)
        target.add_edit(mask_image, mask_region, blend_mode=BlendMode.CLIP)
        return True
