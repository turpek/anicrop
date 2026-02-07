from anicrop.image import Image, ImageFormat
from anicrop.layer import EditLayer, Layer
from anicrop.receiver import RotationHandler
from anicrop.spatial import Region
from anicrop.type import OperationFloat as Float
import operator
# from pytest import raises
import numpy as np
import pytest


W = H = 10  # Tamanhos padrão das imgs
VALUES = (0, -45, 45)   # note: use 1 em vez de 0 para divisão segura
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


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_layer_padrao_sem_EditLayers(img, value, op, op_name):
    layer = Layer(img)
    result = op(layer.rotate, value)
    receiver = RotationHandler(layer, [], result)
    receiver.rotate()
    assert layer.rotate == result


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_layer_rotacionado_sem_EditLayers(img, value, op, op_name):
    layer = Layer(img)
    layer.rotate += 45
    result = op(layer.rotate, value)
    receiver = RotationHandler(layer, [], result)
    receiver.rotate()
    assert layer.rotate == result


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_layer_padrao_com_EditLayers_padrao(img, value, op, op_name):
    edits = [EditLayer(img), EditLayer(img), EditLayer(img), EditLayer(img)]
    layer = Layer(img)
    result = op(layer.rotate, value)
    receiver = RotationHandler(layer, edits, result)
    receiver.rotate()

    assert layer.rotate == result
    for edit in edits:
        assert edit.rotate == result


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', VALUES, ids=('0', '-45', '+45'))
def test_RotationHandler_com_valores_variados(img, value, op, op_name):
    initial_rotations = [-30, -15, 15, 30]
    edits = [EditLayer(img) for _ in range(len(initial_rotations))]
    for edit, rot in zip(edits, initial_rotations):
        edit.rotate = rot

    layer = Layer(img)
    layer.rotate = 90

    result = op(layer.rotate, value)
    receiver = RotationHandler(layer, edits, result)
    receiver.rotate()

    assert layer.rotate == result
    for edit, rot in zip(edits, initial_rotations):
        assert edit.rotate == op(rot, value)
