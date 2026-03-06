from __future__ import annotations
from enum import auto, Enum, IntFlag, StrEnum
import cv2


class BlendMode(Enum):
    """Defines how an edit layer blends with the underlying content."""
    NORMAL = 'normal'
    NORMAL_LINEAR = 'normal_linear'
    MULTIPLY = 'multiply'
    HARD_MASKING = 'hard_masking'


class InterpolationOption(Enum):
    """Opções de interpolação do OpenCV para o motor de renderização."""

    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS = cv2.INTER_LANCZOS4

    @property
    def padding(self) -> int:
        """
        Retorna a quantidade de pixels extras (margem) necessária
        para evitar artefatos de borda durante a transformação.
        """

        # Mapeamento baseado no tamanho do Kernel de cada algoritmo
        return {
            InterpolationOption.NEAREST: 0,
            InterpolationOption.LINEAR: 1,   # Kernel 2x2 (precisa de 1 de margem)
            InterpolationOption.CUBIC: 2,    # Kernel 4x4 (precisa de 2 de margem)
            InterpolationOption.AREA: 1,
            InterpolationOption.LANCZOS: 4   # Kernel 8x8 (precisa de 4 de margem)
        }.get(self, 2)                       # Padrão seguro de 2 pixels


class ImageFormat(StrEnum):
    GRAY = "gray"
    GRAY_ALPHA = "gray_alpha"
    RGB = "rgb"
    RGBA = "rgba"
    CMYK = "cmyk"
    CMYK_ALPHA = "cmyk_alpha"

    @property
    def has_alpha(self) -> bool:
        return self in {
            ImageFormat.GRAY_ALPHA,
            ImageFormat.RGBA,
            ImageFormat.CMYK_ALPHA,
        }

    @property
    def channels(self) -> int:
        return {
            ImageFormat.GRAY: 1,
            ImageFormat.GRAY_ALPHA: 2,
            ImageFormat.RGB: 3,
            ImageFormat.RGBA: 4,
            ImageFormat.CMYK: 4,
            ImageFormat.CMYK_ALPHA: 5,
        }[self]

    def same_spaces(self, other: ImageFormat) -> bool:
        color_spaces = {
            "gray": "gray",
            "gray_alpha": "gray",
            "rgb": "rgb",
            "rgba": "rgb",
            "cmyk": "cmyk",
            "cmyk_alpha": "cmyk",
        }
        return color_spaces[other] == color_spaces[self]


class RenderDirty(IntFlag):
    NONE = 0
    POSITION = auto()  # Invalida a localização global no canvas
    PIXELS = auto()    # Invalida o buffer de pixels (exige re-renderizar o layer)
    ALL = POSITION | PIXELS
