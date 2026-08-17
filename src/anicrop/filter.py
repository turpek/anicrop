from __future__ import annotations
import math
from typing import TYPE_CHECKING
import cv2
import numpy as np

from anicrop.effect import Effect
from anicrop.enums import BlurMode
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
        visible: bool = True,
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
        self.visible = visible
        self.name = name

    def prepare(self, frame: BaseFrame) -> None:
        """Etapa de preparação preliminar para o frame."""
        pass

    def get_padding(self) -> tuple[int, int, int, int]:
        """Calcula a margem de expansão (top, right, bottom, left) necessária para o desfoque."""
        if not self.visible or not self.affect_alpha or self.strength <= 0.0:
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
        if not self.visible or self.strength <= 0.0 or (self.radius_x <= 0.0 and self.radius_y <= 0.0):
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

        if abs(effective_rx - effective_ry) < 1e-4 and effective_rx > 0 and self.mode == BlurMode.GAUSSIAN and effective_angle == 0.0:
            sigma = effective_rx
            processed = cv2.GaussianBlur(src_data, (0, 0), sigmaX=sigma, sigmaY=sigma)
        elif self.mode in (BlurMode.GAUSSIAN, BlurMode.BOX) and (effective_angle != 0.0 or abs(effective_rx - effective_ry) >= 1e-4):
            if effective_angle != 0.0:
                length = max(effective_rx, effective_ry)
                processed = self._apply_directional(src_data, length, effective_angle)
            else:
                # Anisotrópico puro alinhado aos eixos cartesianos X e Y
                if self.mode == BlurMode.GAUSSIAN:
                    processed = src_data
                    if effective_rx > 0:
                        kx = max(3, int(math.ceil(effective_rx * 3.0))) | 1
                        processed = cv2.GaussianBlur(processed, (kx, 1), sigmaX=effective_rx, sigmaY=0)
                    if effective_ry > 0:
                        ky = max(3, int(math.ceil(effective_ry * 3.0))) | 1
                        processed = cv2.GaussianBlur(processed, (1, ky), sigmaX=0, sigmaY=effective_ry)
                else:
                    kx = max(1, int(round(effective_rx * 2.0 + 1.0))) | 1
                    ky = max(1, int(round(effective_ry * 2.0 + 1.0))) | 1
                    processed = cv2.blur(src_data, (kx, ky))
        elif self.mode == BlurMode.BOX:
            k = max(1, int(round(effective_rx * 2.0 + 1.0))) | 1
            processed = cv2.blur(src_data, (k, k))
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
        """Combina dois BlurFilters gaussianos somando seus tensores de covariância no espaço matricial."""
        if not isinstance(other, BlurFilter):
            return None

        if self.mode != BlurMode.GAUSSIAN or other.mode != BlurMode.GAUSSIAN:
            return None

        if self.affect_alpha != other.affect_alpha or self.visible != other.visible:
            return None

        # 1. Resolve matriz delta e parâmetros de self no espaço de renderização
        delta_m1 = matrix @ self.matrix
        ang1_rad = math.radians(self.angle + math.degrees(math.atan2(float(delta_m1[1, 0]), float(delta_m1[0, 0]))))
        s_x1 = math.hypot(float(delta_m1[0, 0]), float(delta_m1[1, 0]))
        s_y1 = math.hypot(float(delta_m1[0, 1]), float(delta_m1[1, 1]))
        r_x1 = self.radius_x * s_x1
        r_y1 = self.radius_y * s_y1

        # Matriz de covariância Sigma 1
        c1, s1 = math.cos(ang1_rad), math.sin(ang1_rad)
        sig1_11 = (r_x1 ** 2) * (c1 ** 2) + (r_y1 ** 2) * (s1 ** 2)
        sig1_22 = (r_x1 ** 2) * (s1 ** 2) + (r_y1 ** 2) * (c1 ** 2)
        sig1_12 = (r_x1 ** 2 - r_y1 ** 2) * c1 * s1

        # 2. Resolve matriz delta e parâmetros de other no espaço de renderização
        delta_m2 = matrix @ other.matrix
        ang2_rad = math.radians(other.angle + math.degrees(math.atan2(float(delta_m2[1, 0]), float(delta_m2[0, 0]))))
        s_x2 = math.hypot(float(delta_m2[0, 0]), float(delta_m2[1, 0]))
        s_y2 = math.hypot(float(delta_m2[0, 1]), float(delta_m2[1, 1]))
        r_x2 = other.radius_x * s_x2
        r_y2 = other.radius_y * s_y2

        # Matriz de covariância Sigma 2
        c2, s2 = math.cos(ang2_rad), math.sin(ang2_rad)
        sig2_11 = (r_x2 ** 2) * (c2 ** 2) + (r_y2 ** 2) * (s2 ** 2)
        sig2_22 = (r_x2 ** 2) * (s2 ** 2) + (r_y2 ** 2) * (c2 ** 2)
        sig2_12 = (r_x2 ** 2 - r_y2 ** 2) * c2 * s2

        # 3. Soma das matrizes de covariância (Teorema da Convolução de Gaussianas)
        sig_11 = sig1_11 + sig2_11
        sig_22 = sig1_22 + sig2_22
        sig_12 = sig1_12 + sig2_12

        # 4. Decomposição espectral da matriz simétrica 2x2 resultante
        trace = sig_11 + sig_22
        diff = sig_11 - sig_22
        discriminant = math.sqrt(max(0.0, diff ** 2 + 4.0 * (sig_12 ** 2)))

        lambda_1 = max(0.0, (trace + discriminant) / 2.0)
        lambda_2 = max(0.0, (trace - discriminant) / 2.0)

        rx_screen = math.sqrt(lambda_1)
        ry_screen = math.sqrt(lambda_2)
        angle_screen_deg = 0.5 * math.degrees(math.atan2(2.0 * sig_12, diff))

        # 5. Converte do espaço da tela de volta para o espaço da matriz de destino
        mat_angle_deg = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        s_x_mat = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
        s_y_mat = math.hypot(float(matrix[0, 1]), float(matrix[1, 1]))

        new_angle = (angle_screen_deg - mat_angle_deg + 180.0) % 360.0 - 180.0
        new_rx = rx_screen / s_x_mat if s_x_mat > 0 else rx_screen
        new_ry = ry_screen / s_y_mat if s_y_mat > 0 else ry_screen
        combined_strength = min(1.0, self.strength * other.strength)

        return BlurFilter(
            radius=(new_rx, new_ry),
            angle=new_angle,
            mode=self.mode,
            affect_alpha=self.affect_alpha,
            strength=combined_strength,
            matrix=matrix,
            visible=self.visible,
            name=self.name,
        )
