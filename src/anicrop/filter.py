from __future__ import annotations
import math
from typing import TYPE_CHECKING
import cv2
import numpy as np

from anicrop.effect import Effect
from anicrop.enums import BlurMode, ImageFormat
from anicrop.image import Image

if TYPE_CHECKING:
    from anicrop.frame import BaseFrame


class BlurFilter(Effect):
    """Filtro de desfoque versátil suportando múltiplos algoritmos e dimensões independentes."""

    def __init__(
        self,
        radius: float | tuple[float, float] = 5.0,
        mode: BlurMode = BlurMode.GAUSSIAN,
        affect_alpha: bool = True,
        strength: float = 1.0,
        name: str = "BlurFilter",
    ):
        if isinstance(radius, (tuple, list)):
            self.radius_x = float(radius[0])
            self.radius_y = float(radius[1])
        else:
            self.radius_x = float(radius)
            self.radius_y = float(radius)

        self.mode = mode
        self.affect_alpha = affect_alpha
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.name = name

    def prepare(self, frame: BaseFrame) -> None:
        """Etapa de preparação preliminar para o frame."""
        pass

    def get_padding(self) -> tuple[int, int, int, int]:
        """Calcula a margem de expansão (top, right, bottom, left) necessária para o desfoque."""
        if not self.affect_alpha or self.strength <= 0.0:
            return (0, 0, 0, 0)

        if self.mode == BlurMode.GAUSSIAN:
            pad_x = int(math.ceil(self.radius_x * 3.0))
            pad_y = int(math.ceil(self.radius_y * 3.0))
        else:
            pad_x = int(math.ceil(self.radius_x))
            pad_y = int(math.ceil(self.radius_y))

        return (pad_y, pad_x, pad_y, pad_x)

    def apply(self, image: Image, matrix: np.ndarray | None = None) -> Image:
        """Processa e desfoca o buffer de imagem."""
        if self.strength <= 0.0 or (self.radius_x <= 0.0 and self.radius_y <= 0.0):
            return image

        src_data = image[...]
        processed = src_data

        if self.mode == BlurMode.GAUSSIAN:
            processed = cv2.GaussianBlur(
                src_data,
                (0, 0),
                sigmaX=self.radius_x,
                sigmaY=self.radius_y,
                borderType=cv2.BORDER_REFLECT_101,
            )
        elif self.mode == BlurMode.BOX:
            kx = max(1, int(round(self.radius_x * 2.0 + 1.0)))
            ky = max(1, int(round(self.radius_y * 2.0 + 1.0)))
            if kx % 2 == 0:
                kx += 1
            if ky % 2 == 0:
                ky += 1
            processed = cv2.boxFilter(
                src_data,
                -1,
                (kx, ky),
                borderType=cv2.BORDER_REFLECT_101,
            )
        elif self.mode == BlurMode.MEDIAN:
            k = max(1, int(round(self.radius_x * 2.0 + 1.0)))
            if k % 2 == 0:
                k += 1
            processed = cv2.medianBlur(src_data, k)

        if not self.affect_alpha and image.format.has_alpha:
            processed = np.copy(processed)
            processed[..., -1] = src_data[..., -1]

        if self.strength < 1.0:
            blended = (src_data.astype(np.float32) * (1.0 - self.strength) +
                       processed.astype(np.float32) * self.strength)
            processed = np.clip(blended, 0, 255).astype(np.uint8)

        return Image(processed, image.format)

    def merge(self, other: Effect, matrix: np.ndarray) -> BlurFilter | None:
        """Combina dois BlurFilters gaussianos contínuos somando seus raios quadraticamente."""
        if not isinstance(other, BlurFilter):
            return None

        if self.mode != BlurMode.GAUSSIAN or other.mode != BlurMode.GAUSSIAN:
            return None

        if self.affect_alpha != other.affect_alpha:
            return None

        combined_rx = math.hypot(self.radius_x, other.radius_x)
        combined_ry = math.hypot(self.radius_y, other.radius_y)
        combined_strength = min(1.0, self.strength * other.strength)

        return BlurFilter(
            radius=(combined_rx, combined_ry),
            mode=self.mode,
            affect_alpha=self.affect_alpha,
            strength=combined_strength,
            name=self.name,
        )
