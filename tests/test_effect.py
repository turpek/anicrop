import numpy as np

from anicrop.effect import BoundEffect, Effect
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.spatial import Region
from anicrop.transform import mat_global


class DummyEffect:
    """Implementação mock de efeito para teste de conformidade de protocolo."""

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


def test_bound_effect_decorates_effect_and_modulates():
    """Valida se BoundEffect encapsula um efeito, calcula matriz delta e modula com máscara."""

    class InvertEffect(DummyEffect):
        def apply(self, image: Image, matrix: np.ndarray) -> Image:
            data = np.copy(image[...])
            data[..., :3] = 255 - data[..., :3]
            return Image(data, image.format)

    data = np.full((10, 10, 4), 255, dtype=np.uint8)
    base_img = Image(data, ImageFormat.RGBA)

    # Máscara: metade esquerda branca (255 = efeito total), metade direita preta (0 = sem efeito)
    mask_data = np.zeros((10, 10, 1), dtype=np.uint8)
    mask_data[:, :5] = 255
    mask = Mask(
        Image(mask_data, ImageFormat.GRAY),
        Region.from_size(10, 10),
        np.identity(3, dtype=np.float32),
    )

    bound_invert = BoundEffect(
        InvertEffect(), matrix=np.identity(3, dtype=np.float32), mask=mask
    )
    result = bound_invert.apply(base_img, np.identity(3, dtype=np.float32))

    # Metade esquerda invertida para preto (0)
    assert result[5, 2, 0] == 0
    # Metade direita preservada branca (255)
    assert result[5, 8, 0] == 255


def test_base_layer_bind_effect_attaches_inverse_matrix():
    """Valida se BaseLayer.bind_effect cria BoundEffect associado à matriz inversa sem alterar o original."""
    layer = Layer(Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA))
    layer.transform.rotate(45)

    original_effect = DummyEffect()
    bound_effect = layer.bind_effect(original_effect)

    assert bound_effect.effect is original_effect
    assert len(layer.effects) == 1
    expected_inv = np.linalg.inv(mat_global(layer))
    np.testing.assert_array_almost_equal(bound_effect.matrix, expected_inv)


def test_base_layer_add_and_remove_effect():
    """Valida adição direta e remoção de efeitos na fila da camada."""
    layer = Layer(Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA))
    e1 = DummyEffect()
    e2 = DummyEffect()

    layer.add_effect(e1)
    layer.add_effect(e2)
    assert len(layer.effects) == 2

    layer.remove_effect(e1)
    assert len(layer.effects) == 1
    assert layer.effects[0] is e2

    layer.clear_effects()
    assert len(layer.effects) == 0
