from anicrop.image import Image
from pytest import raises
import numpy as np
import pytest


@pytest.mark.parametrize(
    "shape",
    [(0, 0), (0, 1), (1, 0)],
    ids=["0x0", "0x1", "1x0"]
)
def test_Image_rejeita_dimensao_zero(shape):
    with pytest.raises(ValueError, match="image dimensions must be greater than zero"):
        Image(np.zeros(shape))


@pytest.mark.parametrize(
    "shape",
    [[1], (1, 1, 1, 1)],
    ids=["1D", "4D"]
)
def test_Image_rejeita_se_nao_for_2D_ou_3D(shape):
    with raises(ValueError, match="image array must be 2D or 3D"):
        Image(np.zeros(shape))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),        # grayscale mínimo
        (10, 20),      # grayscale comum
        (1, 1, 1),     # grayscale com canal explícito
        (10, 20, 1),   # grayscale com canal
        (10, 20, 3),   # RGB
        (10, 20, 4),   # RGBA
    ],
    ids=[
        "gray-1x1",
        "gray-10x20",
        "gray-1x1x1",
        "gray-10x20x1",
        "rgb",
        "rgba",
    ]
)
def test_Image_aceita_formatos_validos(shape):
    Image(np.zeros(shape))


def test_Image_rejeita_0_canais():
    with raises(ValueError, match="image must have at least one channel"):
        Image(np.zeros((1, 1, 0)))


def test_Image_com_width_10():
    assert Image(np.zeros((10, 10))).width == 10


def test_Image_com_height_10():
    assert Image(np.zeros((10, 10))).height == 10


def test_Image_com_size_10x10():
    assert Image(np.zeros((10, 10))).size == (10, 10)


@pytest.mark.parametrize(
    "shape, expect",
    [
        ((1, 1), 1),       # Canal implícito
        ((1, 1, 1), 1),    # Canal explícito
        ((1, 1, 2), 2),    # Canal cinza+alpha
        ((1, 1, 3), 3),    # Canal RGB
        ((1, 1, 4), 4),    # Canal RGBA
        ((1, 1, 10), 10),  # Muito Canais

    ],
    ids=[
        "Grayscale_implicito",
        "Grayscale_explicito",
        "Grayscale+Alpha",
        "RGB",
        "RGBA",
        "Muito_Canais",
    ]
)
def test_Image_com_varios_canais(shape, expect):
    assert Image(np.zeros(shape)).channels == expect
