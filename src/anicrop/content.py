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
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        """
        Aplica um corte não-destrutivo sobre o conteúdo da camada via EditLayer e BlendMode.CLIP.
        Ajusta a moldura via LayerLayoutStrategy.fit e anexa a máscara de recorte nos edits.
        """
        crop_region = resolve_crop_region(ref)

        if not LayerLayoutStrategy.fit(target, crop_region):
            return False

        mask_region = target.global_region
        mask_image = Image.new(mask_region.size, ImageFormat.GRAY, color=255)
        target.add_edit(mask_image, mask_region, blend_mode=BlendMode.CLIP)
        return True

    def resize(
        self,
        target: Layer,
        width: int,
        height: int,
    ) -> bool:
        """
        Redimensiona o conteúdo da camada para a nova largura e altura especificadas.
        Aplica a escala na transformação da camada de forma não-destrutiva.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensões inválidas para resize: ({width}, {height}). Devem ser positivas.")

        cur_w, cur_h = target.global_region.size
        if (cur_w, cur_h) == (width, height):
            return False

        scale_x = width / cur_w
        scale_y = height / cur_h

        target.transform.scale(scale_x, scale_y)
        return True

    def fit(
        self,
        target: Layer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        """
        Ajusta e centraliza o conteúdo da camada para caber dentro da referência `ref`,
        preservando a proporção de aspecto (aspect ratio) original de forma não-destrutiva.
        """
        ref_region = resolve_region(ref)
        cur_region = target.global_region

        if cur_region.size == (0, 0) or ref_region.size == (0, 0):
            return False

        scale = min(
            ref_region.width / cur_region.width,
            ref_region.height / cur_region.height,
        )

        if scale <= 0:
            return False

        target.transform.scale(scale, scale)

        updated_region = target.global_region
        ref_cx = (ref_region.x.start + ref_region.x.end) / 2.0
        ref_cy = (ref_region.y.start + ref_region.y.end) / 2.0
        cur_cx = (updated_region.x.start + updated_region.x.end) / 2.0
        cur_cy = (updated_region.y.start + updated_region.y.end) / 2.0

        dx = int(round(ref_cx - cur_cx))
        dy = int(round(ref_cy - cur_cy))

        target.transform.translate(dx, dy)
        return True
