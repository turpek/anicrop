from __future__ import annotations
from typing import TYPE_CHECKING, Any
import cv2
import numpy as np

from anicrop.edit_layer import EditLayer
from anicrop.effect import Effect
from anicrop.enums import BlendMode, ImageFormat
from anicrop.image import Image
from anicrop.spatial import Region, rect_to_region
from anicrop.transform import calculate_new_rect

if TYPE_CHECKING:
    from anicrop.frame import BaseFrame


class Mask(EditLayer, Effect):
    """Retalho de edição que modula a opacidade / canal Alfa de uma camada ou grupo."""

    def __init__(
        self,
        image: Image,
        region: Region,
        matrix: np.ndarray,
        invert: bool = False,
        visible: bool = True,
        name: str = "Mask",
    ):
        super().__init__(image, region, matrix, BlendMode.NORMAL, name)
        self.invert = invert
        self.visible = visible

    def __getitem__(self, item: Any) -> np.ndarray:
        """Acesso direto à fatia do buffer de imagem da máscara."""
        return self._image[item]

    def __setitem__(self, item: Any, value: Any) -> None:
        """Escrita direta na fatia do buffer de imagem da máscara."""
        self._image[item] = value

    def projected_region(self, matrix: np.ndarray) -> Region:
        """Calcula a Region projetada da máscara combinando a matriz externa com a sua matriz local."""
        mask_matrix = matrix @ self.local_matrix
        rect = calculate_new_rect(mask_matrix, self.region.size)
        return rect_to_region(rect)

    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna a margem necessária. Máscaras apenas restringem área, portanto padding é zero."""
        return (0, 0, 0, 0)

    def _extract_luma(self, mask_img: Image) -> np.ndarray:
        """Extrai matriz 2D de luminância normalizada [0.0, 1.0] a partir de qualquer formato de imagem."""
        data = mask_img[...]
        if mask_img.format in (ImageFormat.GRAY, ImageFormat.GRAY_ALPHA):
            luma = data[..., 0].astype(np.float32)
        elif mask_img.format in (ImageFormat.RGB, ImageFormat.RGBA):
            rgb = data[..., :3]
            luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            luma = data[..., 0].astype(np.float32)

        if self.invert:
            return (255.0 - luma) / 255.0
        return luma / 255.0

    def apply_modulation(self, target_image: Image, mask_image: Image) -> Image:
        """Modula o canal Alfa da imagem de destino utilizando a imagem de máscara fornecida."""
        luma_factor = self._extract_luma(mask_image)
        alpha = target_image[..., -1].astype(np.float32)
        target_image[..., -1] = (alpha * luma_factor).astype(np.uint8)
        return target_image

    def modulate_blend(self, original: Image, filtered: Image) -> Image:
        """Interpola linearmente entre a imagem original e a filtrada usando o mapa de luminância da máscara."""
        mask_data = self._extract_luma(self.image)
        orig_data = original[...].astype(np.float32)
        filt_data = filtered[...].astype(np.float32)

        if mask_data.shape[:2] != orig_data.shape[:2]:
            mask_data = cv2.resize(mask_data, (orig_data.shape[1], orig_data.shape[0]), interpolation=cv2.INTER_LINEAR)

        luma = mask_data[..., np.newaxis]
        blended = orig_data * (1.0 - luma) + filt_data * luma
        return Image(np.clip(blended, 0, 255).astype(np.uint8), original.format)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Aplica a modulação de máscara sobre a imagem recebendo a matriz de transformação."""
        if not self.visible:
            return image
        return self.apply_modulation(image, self.image)

    def merge(self, other: Effect, matrix: np.ndarray) -> Mask | None:
        """Combina duas máscaras na região de união utilizando a matriz fornecida."""
        if not isinstance(other, Mask):
            return None

        if self.visible != other.visible:
            return None

        union_region = self.region | other.region
        combined_img = Image.new(union_region.size, self.image.format)

        if union_region.overlaps(self.region):
            self_target = union_region.overlap_with(self.region)
            combined_img[self_target] = self.image[...]

        if union_region.overlaps(other.region):
            other_target = union_region.overlap_with(other.region)
            combined_img[other_target] = other.image[...]

        return Mask(combined_img, union_region, matrix, invert=self.invert, visible=self.visible, name=self.name)
