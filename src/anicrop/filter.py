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
    """Filtro de desfoque versátil com suporte a raio 1D/2D, ângulo direcional e múltiplos modos."""

    def __init__(
        self,
        radius: float | tuple[float, float] = 5.0,
        angle: float = 0.0,
        mode: BlurMode = BlurMode.GAUSSIAN,
        affect_alpha: bool = True,
        strength: float = 1.0,
        matrix: np.ndarray | None = None,
        name: str = "BlurFilter",
    ):
        if isinstance(radius, (tuple, list)):
            self.radius_x = float(radius[0])
            self.radius_y = float(radius[1])
        else:
            self.radius_x = float(radius)
            self.radius_y = float(radius)

        self.angle = float(angle)
        self.mode = mode
        self.affect_alpha = affect_alpha
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.matrix = matrix if matrix is not None else np.identity(3, dtype=np.float32)
        self.name = name

    def prepare(self, frame: BaseFrame) -> None:
        """Etapa de preparação preliminar para o frame."""
        pass

    def get_padding(self) -> tuple[int, int, int, int]:
        """Calcula a margem de expansão (top, right, bottom, left) necessária para o desfoque."""
        if not self.affect_alpha or self.strength <= 0.0:
            return (0, 0, 0, 0)

        multiplier = 3.0 if self.mode == BlurMode.GAUSSIAN else 1.0

        if self.angle != 0.0:
            rad = math.radians(self.angle)
            len_x = self.radius_x * multiplier
            len_y = self.radius_y * multiplier if self.radius_y > 0 else len_x
            pad_x = int(math.ceil(abs(math.cos(rad)) * len_x + abs(math.sin(rad)) * len_y))
            pad_y = int(math.ceil(abs(math.sin(rad)) * len_x + abs(math.cos(rad)) * len_y))
        else:
            pad_x = int(math.ceil(self.radius_x * multiplier)) if self.radius_x > 0 else 0
            pad_y = int(math.ceil(self.radius_y * multiplier)) if self.radius_y > 0 else 0

        return (pad_y, pad_x, pad_y, pad_x)

    def _apply_directional(self, src_data: np.ndarray, length: float, angle_deg: float) -> np.ndarray:
        """Aplica desfoque direcional linear no ângulo especificado usando kernel 2D rotacionado."""
        multiplier = 3.0 if self.mode == BlurMode.GAUSSIAN else 2.0
        ksize = max(3, int(math.ceil(length * multiplier))) | 1

        kernel = np.zeros((ksize, ksize), dtype=np.float32)

        if self.mode == BlurMode.GAUSSIAN:
            x = np.linspace(-3.0, 3.0, ksize)
            line = np.exp(-0.5 * x**2)
            kernel[ksize // 2, :] = line / np.sum(line)
        else:
            kernel[ksize // 2, :] = 1.0 / ksize

        # Rotaciona a linha do kernel para a inclinação desejada
        center = (ksize / 2.0 - 0.5, ksize / 2.0 - 0.5)
        rot_mat = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        rotated_kernel = cv2.warpAffine(kernel, rot_mat, (ksize, ksize))

        k_sum = np.sum(rotated_kernel)
        if k_sum > 0:
            rotated_kernel /= k_sum

        return cv2.filter2D(src_data, -1, rotated_kernel, borderType=cv2.BORDER_REFLECT_101)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Processa e desfoca o buffer de imagem adaptando ângulo e escala a partir da matriz afim."""
        if self.strength <= 0.0 or (self.radius_x <= 0.0 and self.radius_y <= 0.0):
            return image

        src_data = image[...]

        # Resolve matriz delta combinando a matriz do frame com a âncora inversa do efeito
        delta_matrix = matrix @ self.matrix

        # Resolve ângulo e escalas efetivas no espaço de renderização a partir da matriz delta
        mat_angle_deg = math.degrees(math.atan2(float(delta_matrix[1, 0]), float(delta_matrix[0, 0])))
        effective_angle = self.angle + mat_angle_deg

        scale_x = math.hypot(float(delta_matrix[0, 0]), float(delta_matrix[1, 0]))
        scale_y = math.hypot(float(delta_matrix[0, 1]), float(delta_matrix[1, 1]))
        effective_rx = self.radius_x * scale_x
        effective_ry = self.radius_y * scale_y

        effective_angle = (effective_angle + 180.0) % 360.0 - 180.0

        if abs(effective_angle) > 1e-4 and self.mode in (BlurMode.GAUSSIAN, BlurMode.BOX):
            processed = self._apply_directional(src_data, length=effective_rx, angle_deg=effective_angle)
        elif self.mode == BlurMode.GAUSSIAN:
            if effective_rx > 0 and effective_ry <= 1e-4:
                kx = int(math.ceil(effective_rx * 3.0)) * 2 + 1
                processed = cv2.GaussianBlur(
                    src_data, (kx, 1), sigmaX=effective_rx, sigmaY=0, borderType=cv2.BORDER_REFLECT_101
                )
            elif effective_rx <= 1e-4 and effective_ry > 0:
                ky = int(math.ceil(effective_ry * 3.0)) * 2 + 1
                processed = cv2.GaussianBlur(
                    src_data, (1, ky), sigmaX=0, sigmaY=effective_ry, borderType=cv2.BORDER_REFLECT_101
                )
            else:
                kx = int(math.ceil(effective_rx * 3.0)) * 2 + 1
                ky = int(math.ceil(effective_ry * 3.0)) * 2 + 1
                processed = cv2.GaussianBlur(
                    src_data, (kx, ky), sigmaX=effective_rx, sigmaY=effective_ry, borderType=cv2.BORDER_REFLECT_101
                )
        elif self.mode == BlurMode.BOX:
            kx = max(1, int(round(effective_rx * 2.0 + 1.0))) | 1 if effective_rx > 0 else 1
            ky = max(1, int(round(effective_ry * 2.0 + 1.0))) | 1 if effective_ry > 0 else 1
            processed = cv2.boxFilter(src_data, -1, (kx, ky), borderType=cv2.BORDER_REFLECT_101)
        elif self.mode == BlurMode.MEDIAN:
            k = max(1, int(round(effective_rx * 2.0 + 1.0))) | 1
            processed = cv2.medianBlur(src_data, k)
        else:
            processed = src_data

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

        if self.angle != other.angle or self.affect_alpha != other.affect_alpha:
            return None

        combined_rx = math.hypot(self.radius_x, other.radius_x)
        combined_ry = math.hypot(self.radius_y, other.radius_y)
        combined_strength = min(1.0, self.strength * other.strength)

        return BlurFilter(
            radius=(combined_rx, combined_ry),
            angle=self.angle,
            mode=self.mode,
            affect_alpha=self.affect_alpha,
            strength=combined_strength,
            matrix=matrix,
            name=self.name,
        )
