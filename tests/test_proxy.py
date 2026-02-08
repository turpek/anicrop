from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.proxy import ProxyLayer
from anicrop.spatial import Region
from anicrop.type import Translation, Vector
from operator import add, sub
import numpy as np
import pytest


def make_img(w=100, h=100):
    return Image(np.zeros((h, w, 4), dtype=np.uint8), ImageFormat.RGBA)


@pytest.fixture
def canvas_mock(mocker):
    mock = mocker.Mock()
    mock.region = Region.from_size(500, 500)
    return mock


@pytest.fixture
def layer():
    return Layer(make_img(100, 100))


def test_ProxyLayer_translation_padrao(canvas_mock, layer):
    proxy = ProxyLayer(layer, canvas_mock)
    assert proxy.translation == Translation(0, 0)


def test_ProxyLayer_translation_atribuicao(canvas_mock, layer):
    proxy = ProxyLayer(layer, canvas_mock)

    new_trans = Translation(10, 20)
    proxy.translation = new_trans

    assert proxy.translation == new_trans


@pytest.mark.parametrize('op', (add, sub), ids=['add', 'sub'])
@pytest.mark.parametrize('value', [(-5, 10), (0, 0), (10, -5)], ids=['-5,10', "0,0", "10,-5"])
def test_ProxyLayer_translation_operacoes(canvas_mock, layer, value, op):
    proxy = ProxyLayer(layer, canvas_mock)
    expect = Translation(op(0, value[0]), op(0, value[1]))
    proxy.translation = op(proxy.translation, value)
    assert proxy.translation == expect


def test_ProxyLayer_position_padrao(canvas_mock, layer):
    proxy = ProxyLayer(layer, canvas_mock)
    assert proxy.position == Vector(0, 0)


def test_ProxyLayer_position_com_layer_deslocado_negativamente(canvas_mock, layer):
    canvas_mock.region += (300, 150)
    proxy = ProxyLayer(layer, canvas_mock)
    assert proxy.position == Vector(-300, -150)


def test_ProxyLayer_position_com_layer_deslocado_positivamente(canvas_mock, layer):
    layer.region += (300, 150)
    proxy = ProxyLayer(layer, canvas_mock)
    assert proxy.position == Vector(300, 150)
