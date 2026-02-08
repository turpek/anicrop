from anicrop.image import Image, ImageFormat
from anicrop.layer import EditLayer, Layer
from anicrop.receiver import RotationHandler
from anicrop.spatial import Region
from anicrop.proxy import ProxyLayer
import operator
# from pytest import raises
import numpy as np
import pytest


W = H = 10  # Tamanhos padrão das imgs
VALUES = (0, -45, 45)   # note: use 1 em vez de 0 para divisão segura
VALUES_DIV = (1, -2, 2, -45, 45)
OPS = [
    (operator.add, "add"),
    (operator.sub, "sub"),
    (operator.mul, "mul"),
]


def make_region(w=3, h=3, offset=0):
    return Region.from_size(w, h) + offset


def make_img(w=W, h=H, channel=4):
    channels = {
        1: ImageFormat.GRAY, 2: ImageFormat.GRAY_ALPHA, 3: ImageFormat.RGB,
        4: ImageFormat.RGBA, -4: ImageFormat.CMYK, 5: ImageFormat.CMYK_ALPHA
    }
    return Image(np.ones((h, w, abs(channel)), dtype=np.uint8), channels.get(channel))


@pytest.fixture
def img():
    return make_img()


@pytest.fixture
def layer():
    return Layer(make_img())


@pytest.fixture
def canvas_mock(mocker):
    mock = mocker.Mock()
    mock.region = Region.from_size(500, 500)
    return mock


@pytest.fixture
def proxy(mocker):
    mock = mocker.Mock()
    mock.region = Region.from_size(500, 500)
    layer = Layer(make_img())
    return ProxyLayer(layer, mock)


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_layer_padrao_sem_EditLayers(value, op, op_name, proxy):
    result = op(proxy.rotate, value)
    receiver = RotationHandler(proxy, result)
    receiver.rotate()
    assert proxy.rotate == result


@pytest.mark.parametrize('value', [-90, 0, 90], ids=('-90', '0', '+90'))
def test_RotationHandler_atribuicao(proxy, value):
    proxy.rotate = value
    assert proxy.rotate == value


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_layer_rotacionado_sem_EditLayers(value, op, op_name, proxy):
    proxy._layer.rotate += 45
    result = op(proxy.rotate, value)
    receiver = RotationHandler(proxy, result)
    receiver.rotate()
    assert proxy.rotate == result


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_layer_padrao_com_EditLayers_padrao(img, value, op, op_name, proxy):
    edits = [EditLayer(img), EditLayer(img), EditLayer(img), EditLayer(img)]
    proxy.add_edits(edits)
    result = op(proxy.rotate, value)
    receiver = RotationHandler(proxy, result)
    receiver.rotate()

    assert proxy.rotate == result
    for edit in edits:
        assert edit.rotate == result


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_com_valores_variados(img, value, op, op_name, proxy):
    initial_rotations = [-30, -15, 15, 30]
    edits = [EditLayer(img) for _ in range(len(initial_rotations))]
    for edit, rot in zip(edits, initial_rotations):
        edit.rotate = rot

    proxy._layer.rotate = 90
    proxy.add_edits(edits)

    result = op(proxy.rotate, value)
    receiver = RotationHandler(proxy, result)
    receiver.rotate()

    assert proxy.rotate == result
    for edit, rot in zip(edits, initial_rotations):
        assert edit.rotate == op(rot, value)


@pytest.mark.parametrize('value', VALUES_DIV)
def test_RotationHandler_divisao_com_valores_variados(img, value, proxy):
    op = operator.truediv
    initial_rotations = [-90, -45, 45, 90]
    edits = [EditLayer(img) for _ in range(len(initial_rotations))]
    for edit, rot in zip(edits, initial_rotations):
        edit.rotate = rot

    proxy._layer.rotate = 180
    proxy.add_edits(edits)

    result = op(proxy.rotate, value)
    receiver = RotationHandler(proxy, result)
    receiver.rotate()

    assert proxy.rotate == result
    for edit, rot in zip(edits, initial_rotations):
        assert edit.rotate == op(rot, value)


def test_RotationHandler_divisao_por_zero_levanta_erro(proxy):
    proxy.rotate = 90
    with pytest.raises(ZeroDivisionError):
        operator.truediv(proxy.rotate, 0)
