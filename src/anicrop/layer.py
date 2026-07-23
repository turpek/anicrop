from __future__ import annotations
from abc import ABC

from anicrop.canvas import Canvas
from anicrop.enums import BlendMode, RenderFlags, WarpMode
from anicrop.image import Image
from anicrop.spatial import Region, Span
from anicrop.type import Id, Rotation, RotationInput, Scale, ScaleInput
from anicrop.transform import (
    calculate_new_bbox_from_layer,
    mat_global,
    mat_inverse,
    mat_scale,
    mat_position,
    Transform,
    Composer,
    ComposerRel,
)
from collections import deque
from typing import Optional

import math
import numpy as np


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
        name: str = 'Edit'
    ):

        self._image = image
        self._region = region
        self.blend_mode = blend_mode
        self.name = name
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
        """Returns (lod_image, m_local) based on the target scale factor.

        Rules:
        - If scale_factor >= 1.0 or n <= 0, returns the original image and original local_matrix.
        - If scale_factor < 1.0, calculates discrete level n = floor(-log2(scale_factor)) and lod_factor = 2^-n.
        - Computes the adjusted local matrix (local_matrix @ m_adjust) compensating for LOD dimensions.
        - Caches and reuses generated LOD images in _lod_cache.
        """

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


class Layer(ABC):

    def __init__(
        self,
        image: Image,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'Layer',
        canvas: Optional[Canvas] = None,
    ):
        self._id = Id()
        self._name = name
        self._opacity = opacity
        self._blend_mode = blend_mode
        self._region = Region.from_size(*image.size)
        self._edits: deque[EditLayer] = deque()
        self._transform: Composer = ComposerRel(self.region.size)
        self._opacity_mask: Optional[np.ndarray] = None
        self.visible = True

        self.add_edit(image, self._region, blend_mode)
        self._image = self._edits[0]
        self._old_matrix = np.zeros((3, 3))
        self._render_flags = RenderFlags.ALL_DIRTY
        self._warp_mode = WarpMode.AFFINE
        self._canvas = canvas

    def __repr__(self) -> str:
        return f"Layer(x={self.x.start}, y={self.y.start}, size={self.image.size})"

    def __eq__(self, other):
        return isinstance(other, Layer) and self._id == other._id

    def __hash__(self):
        return hash(self._id)

    def _resolve_render(self) -> RenderFlags:
        current_matrix = mat_global(self)

        # 2. Compara a parte 2x2 (rotação/escala)
        if not np.allclose(current_matrix[:2, :2], self._old_matrix[:2, :2]):
            self._render_flags |= RenderFlags.PIXELS

        # 3. Compara a translação [tx, ty]
        if not np.allclose(current_matrix[:2, 2], self._old_matrix[:2, 2]):
            self._render_flags |= RenderFlags.POSITION

        if not np.allclose(current_matrix[2, :2], [0, 0]):
            self._warp_mode = WarpMode.PERSPECTIVE
        else:
            self._warp_mode = WarpMode.AFFINE

        return self._render_flags

    def _commit_render_state(self):
        self._old_matrix = mat_global(self)
        self._render_flags = RenderFlags.NONE

    @property
    def format(self):
        return self.image.format

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def x(self) -> Span:
        return self.region.x

    @x.setter
    def x(self, value: int | Span):
        if isinstance(value, Span):
            self._region = Region(value, self.y)
        elif isinstance(value, int):
            self._region = Region(Span(value, self.x.length), self.y)

    @property
    def y(self) -> Span:
        return self.region.y

    @y.setter
    def y(self, value: int | Span):
        if isinstance(value, Span):
            self._region = Region(self.x, value)
        elif isinstance(value, int):
            self._region = Region(self.x, Span(value, self.y.length))

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float) -> None:
        self._opacity = opacity

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        if not isinstance(region, Region):
            raise TypeError(f"Expected Region, got {type(region).__name__}")
        self._region = region

    @property
    def image(self) -> Image:
        return self._image.image

    @property
    def blend_mode(self) -> BlendMode:
        return self._blend_mode

    @blend_mode.setter
    def blend_mode(self, blend_mode: BlendMode) -> None:
        self._blend_mode = blend_mode

    @property
    def global_region(self) -> Region:
        """Retorna o BBox (AABB) real do layer no espaço do Canvas."""

        x, y, w, h = calculate_new_bbox_from_layer(self)
        return Region(Span(x, w), Span(y, h))

    @property
    def canvas_size(self) -> tuple[int, int]:
        cv_size = self._canvas.size if self._canvas else self.region.size
        return cv_size

    def add_edit(
        self,
        image: Image,
        region: Region,
        blend_mode: BlendMode = BlendMode.NORMAL
    ) -> None:

        name = f'Edit-{len(self._edits) + 1}'
        matrix = mat_inverse(mat_global(self))
        self._edits.append(EditLayer(image, region, matrix, blend_mode, name))

    @property
    def transform(self) -> Composer:
        return self._transform

    def transform_clear(self) -> None:
        self._transform = ComposerRel(self.region.size)

    def set_transform(
        self,
        transform: Transform,
        reference: Optional[Canvas | Layer] = None,
    ) -> None:
        self._transform = transform.create_composer(self.region.size)
        ref_size = reference.region.size if reference is not None else self.region.size
        self._transform.add_transform(transform, reference_size=ref_size)
