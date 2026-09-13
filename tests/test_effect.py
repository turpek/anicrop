import numpy as np

from anicrop.canvas import Canvas
from anicrop.effect import BoundEffect, Effect
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.render import CanvasRender, has_active_post_processing
from anicrop.spatial import Region
from anicrop.transform import mat_global


class DummyEffect(Effect):
    """Implementação mock de efeito para teste de conformidade de ABC."""

    def get_padding(self) -> tuple[int, int, int, int]:
        return (5, 5, 5, 5)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        return image

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None


def test_effect_abc_instance():
    """Valida se classes concretas herdando de Effect são instâncias válidas de Effect."""
    dummy = DummyEffect()
    assert isinstance(dummy, Effect)
    assert dummy.visible is True
    assert dummy.name == "Effect"


def test_mask_satisfies_effect_protocol():
    """Valida se a classe Mask atende formalmente à classe abstrata Effect."""
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


def test_has_active_post_processing_detection():
    """Valida o cálculo de has_active_post_processing considerando visibilidade de efeitos."""
    layer = Layer(Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA))
    assert has_active_post_processing(layer) is False

    effect_hidden = DummyEffect(visible=False)
    layer.add_effect(effect_hidden)
    assert has_active_post_processing(layer) is False

    layer.clear_effects()
    effect_visible = DummyEffect(visible=True)
    layer.add_effect(effect_visible)
    assert has_active_post_processing(layer) is True


def test_has_active_post_processing_mask_visibility():
    """Valida que máscara só ativa has_active_post_processing se visible for True."""
    layer = Layer(Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA))
    mask_img = Image(np.zeros((20, 20, 1), dtype=np.uint8), ImageFormat.GRAY)
    mask = layer.set_mask(mask_img, Region.from_size(20, 20), visible=False)
    assert has_active_post_processing(layer) is False

    mask.visible = True
    assert has_active_post_processing(layer) is True


def test_render_layer_isolates_buffer_against_in_place_effect_mutation():
    """Valida que efeitos com mutação in-place não corrompem a imagem original do EditLayer."""
    class MutatingCutEffect(Effect):
        def get_padding(self) -> tuple[int, int, int, int]:
            return (0, 0, 0, 0)

        def apply(self, image: Image, matrix: np.ndarray) -> Image:
            image[0:10, 0:10, -1] = 0
            return image

        def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
            return None

    arr = np.full((30, 30, 4), 255, dtype=np.uint8)
    img = Image(arr, ImageFormat.RGBA)
    layer = Layer(img)
    layer.add_effect(MutatingCutEffect())

    canvas = Canvas.from_size(30, 30)
    renderer = CanvasRender()

    out1 = renderer.render_scene([layer], canvas)
    assert np.all(out1[0:10, 0:10, 3] == 0)
    assert np.all(img[0:10, 0:10, 3] == 255)

    out2 = renderer.render_scene([layer], canvas)
    assert np.all(out2[0:10, 0:10, 3] == 0)
    assert np.all(img[0:10, 0:10, 3] == 255)
