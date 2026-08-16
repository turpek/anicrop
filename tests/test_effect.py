from typing import Protocol, runtime_checkable
import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.effect import Effect, MaskedEffect
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.spatial import Region


class DummyEffect:
    """Implementação mock de efeito para teste de conformidade de protocolo."""

    def __init__(self, matrix: np.ndarray | None = None):
        self.matrix = matrix if matrix is not None else np.identity(3, dtype=np.float32)

    def prepare(self, frame) -> None:
        pass

    def get_padding(self) -> tuple[int, int, int, int]:
        return (5, 5, 5, 5)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
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


def test_masked_effect_decorates_effect_and_modulates():
    """Valida se MaskedEffect encapsula um efeito e aplica modulação baseada na máscara."""
    class InvertEffect(DummyEffect):
        def apply(self, image: Image, matrix: np.ndarray) -> Image:
            data = np.copy(image[...])
            data[..., :3] = 255 - data[..., :3]
            return Image(data, image.format)

    # Imagem base branca
    data = np.full((10, 10, 4), 255, dtype=np.uint8)
    base_img = Image(data, ImageFormat.RGBA)

    # Máscara: metade esquerda branca (255 = efeito total), metade direita preta (0 = sem efeito)
    mask_data = np.zeros((10, 10, 1), dtype=np.uint8)
    mask_data[:, :5] = 255
    mask = Mask(Image(mask_data, ImageFormat.GRAY), Region.from_size(10, 10), np.identity(3, dtype=np.float32))

    masked_invert = MaskedEffect(InvertEffect(), mask)
    result = masked_invert.apply(base_img, np.identity(3, dtype=np.float32))

    # Metade esquerda invertida para preto (0)
    assert result[5, 2, 0] == 0
    # Metade direita preservada branca (255)
    assert result[5, 8, 0] == 255


def test_base_layer_add_effect_binds_inverse_matrix_and_copies():
    """Valida se BaseLayer.add_effect clona o efeito e anexa a matriz inversa global da camada."""
    layer = Layer(Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA))
    layer.transform.rotate(45)

    original_effect = DummyEffect()
    bound_effect = layer.add_effect(original_effect)

    assert bound_effect is not original_effect
    assert len(layer.effects) == 1
    # Matriz vinculada deve ser a inversa da matriz global da camada
    expected_inv = np.linalg.inv(layer.control.base.matrix)
    np.testing.assert_array_almost_equal(bound_effect.matrix, expected_inv)
