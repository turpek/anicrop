from typing import Protocol, runtime_checkable
import numpy as np
import pytest

from anicrop.effect import Effect
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.mask import Mask
from anicrop.spatial import Region


class DummyEffect:
    """Implementação mock de efeito para teste de conformidade de protocolo."""

    def prepare(self, frame) -> None:
        pass

    def get_padding(self) -> tuple[int, int, int, int]:
        return (5, 5, 5, 5)

    def apply(self, image: Image, matrix: np.ndarray | None = None) -> Image:
        return image

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None


def test_effect_protocol_runtime_checkable():
    """Valida se classes compatíveis atendem ao protocolo Effect em tempo de execução."""
    dummy = DummyEffect()
    assert isinstance(dummy, Effect)


def test_mask_satisfies_effect_protocol():
    """Valida se a classe Mask atende formalmente ao protocolo Effect."""
    mask_img = Image(np.zeros((10, 10, 1), dtype=np.uint8), ImageFormat.GRAY)
    mask = Mask(mask_img, Region.from_size(10, 10), np.identity(3, dtype=np.float32))
    assert isinstance(mask, Effect)
