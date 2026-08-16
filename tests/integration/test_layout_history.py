import numpy as np
import pytest

from anicrop.document import Document
from anicrop.enums import ImageFormat
from anicrop.geometry import FitGeometry, FitGroupGeometry, GroupGeometry, LayerGeometry
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.layout import Layout
from anicrop.spatial import Region


@pytest.fixture
def make_doc_with_layer():
    def _factory(w: int = 1000, h: int = 1000, layer_w: int = 200, layer_h: int = 200):
        doc = Document(name="test_doc", width=w, height=h, wrap_proxy=True)
        img = Image(np.ones((layer_h, layer_w, 4), dtype=np.uint8) * 255, ImageFormat.RGBA)
        layer_proxy = doc.add_layer(Layer(img, name="layer1"))
        return doc, layer_proxy
    return _factory


@pytest.fixture
def make_doc_with_group():
    def _factory(w: int = 1000, h: int = 1000):
        doc = Document(name="test_doc", width=w, height=h, wrap_proxy=True)
        group_proxy = doc.create_group("group1")
        img1 = Image(np.ones((200, 200, 4), dtype=np.uint8) * 255, ImageFormat.RGBA)
        l1 = doc.add_layer(Layer(img1, name="l1"))
        l1.region = Region.from_rect(0, 0, 200, 200)
        img2 = Image(np.ones((300, 300, 4), dtype=np.uint8) * 255, ImageFormat.RGBA)
        l2 = doc.add_layer(Layer(img2, name="l2"))
        l2.region = Region.from_rect(100, 100, 300, 300)
        group_proxy.append(l1)
        group_proxy.append(l2)
        return doc, group_proxy
    return _factory


def test_layout_fit_proxy_layer_undo_redo(make_doc_with_layer):
    """Valida se layout.fit em ProxyLayer grava histórico e suporta ciclo completo de Undo/Redo."""
    doc, layer_proxy = make_doc_with_layer(1000, 1000, 200, 200)
    layout = Layout()
    initial_stack_size = len(doc.history._undo_stack)

    layout.fit(layer_proxy, (0, 0, 500, 500))

    assert len(doc.history._undo_stack) == initial_stack_size + 1
    assert layer_proxy.region == Region.from_size(500, 500)
    assert isinstance(layer_proxy.layout, FitGeometry)

    doc.history.undo()

    assert layer_proxy.region == Region.from_size(200, 200)
    assert isinstance(layer_proxy.layout, LayerGeometry)
    assert len(doc.history._redo_stack) == 1

    doc.history.redo()

    assert layer_proxy.region == Region.from_size(500, 500)
    assert isinstance(layer_proxy.layout, FitGeometry)
    assert len(doc.history._redo_stack) == 0


def test_layout_align_proxy_layer_undo_redo(make_doc_with_layer):
    """Valida se layout.align em ProxyLayer grava histórico e suporta ciclo completo de Undo/Redo."""
    doc, layer_proxy = make_doc_with_layer(1000, 1000, 200, 200)
    layout = Layout()
    initial_stack_size = len(doc.history._undo_stack)

    layout.align(layer_proxy, (0, 0, 1000, 1000), 1.0, 1.0)

    assert len(doc.history._undo_stack) == initial_stack_size + 1
    assert layer_proxy.region == Region.from_rect(800, 800, 200, 200)

    doc.history.undo()

    assert layer_proxy.region == Region.from_size(200, 200)
    assert len(doc.history._redo_stack) == 1

    doc.history.redo()

    assert layer_proxy.region == Region.from_rect(800, 800, 200, 200)
    assert len(doc.history._redo_stack) == 0


def test_layout_resize_bounds_proxy_layer_undo_redo(make_doc_with_layer):
    """Valida se layout.resize_bounds em ProxyLayer grava histórico e suporta Undo/Redo."""
    doc, layer_proxy = make_doc_with_layer(1000, 1000, 200, 200)
    layout = Layout()
    initial_stack_size = len(doc.history._undo_stack)

    layout.resize_bounds(layer_proxy, 400, 400, anchor_x=0.5, anchor_y=0.5)

    assert len(doc.history._undo_stack) == initial_stack_size + 1
    assert layer_proxy.region == Region.from_rect(-100, -100, 400, 400)

    doc.history.undo()

    assert layer_proxy.region == Region.from_size(200, 200)
    assert len(doc.history._redo_stack) == 1

    doc.history.redo()

    assert layer_proxy.region == Region.from_rect(-100, -100, 400, 400)


def test_layout_fit_group_proxy_undo_redo(make_doc_with_group):
    """Valida se layout.fit em GroupProxy grava histórico e suporta ciclo de Undo/Redo."""
    doc, group_proxy = make_doc_with_group(1000, 1000)
    layout = Layout()
    initial_stack_size = len(doc.history._undo_stack)

    layout.fit(group_proxy, (0, 0, 800, 800))

    assert len(doc.history._undo_stack) == initial_stack_size + 1
    assert group_proxy.global_region == Region.from_size(800, 800)
    assert isinstance(group_proxy.layout, FitGroupGeometry)

    doc.history.undo()

    assert group_proxy.global_region == Region.from_rect(0, 0, 400, 400)
    assert isinstance(group_proxy.layout, GroupGeometry)
    assert len(doc.history._redo_stack) == 1

    doc.history.redo()

    assert group_proxy.global_region == Region.from_size(800, 800)
    assert isinstance(group_proxy.layout, FitGroupGeometry)


def test_read_properties_does_not_clear_redo_stack(make_doc_with_layer):
    """Valida se leituras sucessivas de propriedades não criam ações intermediárias nem limpam o redo stack."""
    doc, layer_proxy = make_doc_with_layer(1000, 1000, 200, 200)
    layout = Layout()

    layout.fit(layer_proxy, (0, 0, 500, 500))
    doc.history.undo()

    assert len(doc.history._redo_stack) == 1

    # Leituras passivas
    _ = layer_proxy.region
    _ = layer_proxy.global_region
    _ = layer_proxy.layout
    _ = layer_proxy.matrix
    _ = layer_proxy.control.layout.region

    # Garante que o redo_stack continua intacto após leituras
    assert len(doc.history._redo_stack) == 1

    doc.history.redo()

    assert layer_proxy.region == Region.from_size(500, 500)
    assert len(doc.history._redo_stack) == 0
