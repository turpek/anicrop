from anicrop.image import Image
from anicrop.spatial import Region, Vector
from pytest import raises
import numpy as np
import pytest


def region_(size=3, offset=0):
    return Region.from_size(size, size)


def make_region(w=3, h=3):
    return Region.from_size(w, h)


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


def test_Imagem_getitem_com_Region():
    region = Region.from_size(3, 3) + 3
    data = np.arange(10 * 10 * 3).reshape(10, 10, 3)
    img = Image(data)
    sub = img[region]
    assert np.array_equal(sub, data[3:6, 3:6])


def test_Image_getitem_region_preserva_canais():
    img = Image(np.zeros((10, 10, 5)))
    region = Region.from_size(2, 2)
    assert img[region].shape[2] == 5


def test_Image_getitem_region_grayscale():
    img = Image(np.zeros((10, 10)))
    region = Region.from_size(4, 4)
    assert img[region].shape == (4, 4)


def test_Image_getitem_region_com_slice_de_canais():
    img = Image(np.zeros((10, 10, 5)))
    region = Region.from_size(2, 2)
    assert img[region, :2].shape[2] == 2


@pytest.mark.parametrize(
    "args",
    [
        (make_region(), slice(1, 2), make_region(5, 1)),
        (slice(0, 5), slice(2, 4), slice(1, 2), make_region(5, 1)),
        (..., slice(1, 2), make_region(5, 1)),
    ],
    ids=[
        "region_slice_region",
        "slice_slice_region",
        "ellipsis_slice_region",
    ],
)
def test_Image_getitem_com_entradas_invalidas(args):
    with raises(TypeError, match="Region must be the first and only spatial argument"):
        img = Image(np.zeros((10, 10, 5)))
        img[args]
