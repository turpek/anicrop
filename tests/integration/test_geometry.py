import numpy as np
import pytest
from anicrop.document import Document
from anicrop.geometry import LayerGeometry
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.spatial import Region


def test_geometry_controller_offset_com_rotacao():
    """Valida se o GeometryController sincroniza a base.region considerando a rotacao ao mover a moldura."""
    img = Image.new((736, 1104), ImageFormat.RGBA)
    layer = Layer(img)

    # 1. Base em (0, 0, 736, 1104) e Layout em (100, 50, 400, 400) -> offset (-100, -50)
    layer.layout = LayerGeometry(layer, Region.from_rect(100, 50, 400, 400))
    assert layer.control._offset.top_left == (-100, -50)

    # 2. Aplica a rotação de 45 graus
    layer.transform.rotate(45)

    # 3. Move a moldura para (168, 352) -> sync deve calcular a base em (133, 246)
    layer.region = Region.from_rect(168, 352, 400, 400)

    assert layer.region.top_left == (168, 352)
    assert layer.base.region.top_left == (133, 246)
    assert layer.base.region.size == (736, 1104)


def test_geometry_consecutive_fit_align_fit():
    """Valida que chamadas consecutivas de fit, align e novo fit mantêm a coerência espacial do offset e da base."""
    doc = Document("Consecutive Fit Test", 800, 800, history=False)
    img = Image.new((600, 600), ImageFormat.RGBA)
    layer = Layer(img)
    doc.add(layer)

    # 1. 1º Fit (janela de 200x200 em 50, 50)
    doc.layout.fit(layer, Region.from_rect(50, 50, 200, 200))
    assert layer.region == Region.from_rect(50, 50, 200, 200)
    assert layer.control._offset.top_left == (-50, -50)

    # 2. Align ao centro do Canvas de 800x800 (move para 300, 300)
    doc.layout.align(layer, doc.canvas)
    assert layer.region == Region.from_rect(300, 300, 200, 200)
    assert layer.base.region.top_left == (250, 250)

    # 3. 2º Fit consecutivo (nova moldura de 300x300 em 200, 200)
    doc.layout.fit(layer, Region.from_rect(200, 200, 300, 300))
    assert layer.region == Region.from_rect(200, 200, 300, 300)
    assert layer.control._offset.top_left == (50, 50)
    assert layer.base.region.top_left == (250, 250)
