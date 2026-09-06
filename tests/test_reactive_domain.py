from __future__ import annotations

import numpy as np

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer, LayerStack, NullContainer
from anicrop.content import Content
from anicrop.document import Document
from anicrop.enums import ImageFormat
from anicrop.history import GlobalHistory
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.layout import Layout
from anicrop.mask import Mask
from anicrop.reactive import (
    BaseHistoryProxy,
    GroupProxy,
    LayerStackProxy,
    ProxyCanvas,
    ProxyLayer,
    ProxyMask,
    get_registry_for_history,
)
from anicrop.spatial import Region


def make_layer(
    color: tuple[int, ...] = (255, 0, 0, 255),
    size: tuple[int, int] = (100, 100),
    name: str = "Layer",
) -> Layer:
    """Cria uma camada com imagem solida para testes."""
    img = Image.new(size, ImageFormat.RGBA, color=color)
    return Layer(img, name=name)


def test_proxy_mask_setitem_records_and_undo_redo():
    """Garante que escrita por slice em ProxyMask grava micro-snapshot com Undo e Redo."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    raw_img = Image(np.full((40, 40, 1), 255, dtype=np.uint8), ImageFormat.GRAY)
    mask = Mask(raw_img, Region.from_size(40, 40), np.identity(3, dtype=np.float32))
    proxy_mask: ProxyMask = registry.get_or_create(mask)

    proxy_mask[10:20, 10:20] = 0
    assert proxy_mask[15, 15, 0] == 0

    history.undo()
    assert proxy_mask[15, 15, 0] == 255

    history.redo()
    assert proxy_mask[15, 15, 0] == 0


def test_proxy_mask_scalar_properties_record_and_undo():
    """Garante que alteracao de visible e invert em ProxyMask grava historico com Undo."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    raw_img = Image(np.full((30, 30, 1), 255, dtype=np.uint8), ImageFormat.GRAY)
    mask = Mask(raw_img, Region.from_size(30, 30), np.identity(3, dtype=np.float32))
    proxy_mask: ProxyMask = registry.get_or_create(mask)

    proxy_mask.visible = False
    assert proxy_mask.visible is False

    history.undo()
    assert proxy_mask.visible is True

    proxy_mask.invert = True
    assert proxy_mask.invert is True

    history.undo()
    assert proxy_mask.invert is False


def test_proxy_layer_scalar_properties_record_and_undo():
    """Garante que alteracoes de opacity, visible e name em ProxyLayer suportam Undo e Redo."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    layer = make_layer(name="L1")
    proxy: ProxyLayer = registry.get_or_create(layer)

    proxy.opacity = 0.5
    assert proxy.opacity == 0.5
    assert layer.opacity == 0.5

    history.undo()
    assert proxy.opacity == 1.0
    assert layer.opacity == 1.0

    history.redo()
    assert proxy.opacity == 0.5

    proxy.name = "Renamed"
    assert proxy.name == "Renamed"
    history.undo()
    assert proxy.name == "L1"


def test_proxy_layer_transform_fluent_chaining_single_command():
    """Garante que encadeamento fluente de transformacao grava exatamente um comando com Undo."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    layer = make_layer()
    proxy: ProxyLayer = registry.get_or_create(layer)

    proxy.transform.rotate(90).scale(2.0, 2.0).translate(10.0, 20.0)

    # Verifica se transformacao acumulou
    assert not np.allclose(proxy.matrix, np.identity(3, dtype=np.float32))

    history.undo()
    assert np.allclose(proxy.matrix, np.identity(3, dtype=np.float32))

    history.redo()
    assert not np.allclose(proxy.matrix, np.identity(3, dtype=np.float32))


def test_proxy_layer_mask_integration_and_identity():
    """Garante que set_mask adiciona mascara reativa com ProxyMask compartilhado."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    layer = make_layer()
    proxy: ProxyLayer = registry.get_or_create(layer)

    mask_img = Image(np.full((100, 100, 1), 255, dtype=np.uint8), ImageFormat.GRAY)
    proxy.set_mask(mask_img, Region.from_size(100, 100))

    assert proxy.mask is not None
    assert isinstance(proxy.mask, ProxyMask)

    history.undo()
    assert proxy.mask is None

    history.redo()
    assert proxy.mask is not None


def test_proxy_layer_layout_align_atomic_history():
    """Garante que layout.align em ProxyLayer gera transacao atomica revertida em 1 Undo."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    layer = make_layer(size=(50, 50))
    proxy: ProxyLayer = registry.get_or_create(layer)

    canvas_region = Region.from_size(200, 200)
    orig_region = proxy.region

    proxy.layout.align(canvas_region, anchor_x=1.0, anchor_y=1.0)
    assert proxy.region != orig_region

    history.undo()
    assert proxy.region == orig_region

    history.redo()
    assert proxy.region != orig_region


def test_proxy_container_append_remove_and_parent_identity():
    """Garante que append e remove em contêiner atualizam parent reativo e suportam Undo."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    stack = LayerStack()
    layer = make_layer()

    stack_proxy: LayerStackProxy = registry.get_or_create(stack)
    layer_proxy: ProxyLayer = registry.get_or_create(layer)

    stack_proxy.append(layer_proxy)
    assert len(stack_proxy) == 1
    assert layer_proxy.parent is stack_proxy

    history.undo()
    assert len(stack_proxy) == 0
    assert isinstance(layer_proxy.parent, NullContainer)

    history.redo()
    assert len(stack_proxy) == 1
    assert layer_proxy.parent is stack_proxy


def test_proxy_container_iteration_and_indexing_returns_proxies():
    """Garante que iteracao e indexacao em contêiner retornam proxies via Identity Map."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    stack = LayerStack()
    layer1 = make_layer(name="L1")
    layer2 = make_layer(name="L2")

    stack_proxy: LayerStackProxy = registry.get_or_create(stack)
    lp1: ProxyLayer = registry.get_or_create(layer1)
    lp2: ProxyLayer = registry.get_or_create(layer2)

    stack_proxy.append(lp1)
    stack_proxy.append(lp2)

    # Indexação retorna o mesmo proxy existente
    assert stack_proxy[0] is lp1
    assert stack_proxy[1] is lp2

    # Iteração produz proxies
    children = list(stack_proxy)
    assert children[0] is lp1
    assert children[1] is lp2


def test_group_proxy_combines_container_and_layer_operations():
    """Garante que GroupProxy suporta hierarquia de container e propriedades de camada com Undo."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    group = GroupLayer(name="Folder")
    layer = make_layer(name="Child")

    group_proxy: GroupProxy = registry.get_or_create(group)
    layer_proxy: ProxyLayer = registry.get_or_create(layer)

    group_proxy.append(layer_proxy)
    group_proxy.opacity = 0.7

    assert len(group_proxy) == 1
    assert group_proxy.opacity == 0.7
    assert layer_proxy.parent is group_proxy

    history.undo()
    assert group_proxy.opacity == 1.0
    assert len(group_proxy) == 1

    history.undo()
    assert len(group_proxy) == 0
    assert isinstance(layer_proxy.parent, NullContainer)


def test_proxy_instances_identity_stability():
    """Garante que acessos sucessivos a layout e content em proxies mantêm a mesma identidade de instância."""
    history = GlobalHistory()
    registry = get_registry_for_history(history)
    layer = make_layer()
    group = GroupLayer(name="G1")
    canvas = Canvas.from_size(800, 600)

    lp: ProxyLayer = registry.get_or_create(layer)
    gp: GroupProxy = registry.get_or_create(group)
    cp: ProxyCanvas = registry.get_or_create(canvas)

    assert lp.layout is lp.layout
    assert lp.content is lp.content
    assert gp.layout is gp.layout
    assert gp.content is gp.content
    assert cp.layout is cp.layout


def test_proxy_canvas_scalar_and_region_properties_undo_redo():
    """Garante que alteracoes de bg_color e region em ProxyCanvas suportam ciclo de Undo e Redo."""
    doc = Document("CanvasTest", 800, 600, history=True)
    assert doc.history is not None
    initial_region = doc.canvas.region
    initial_bg = doc.canvas.bg_color

    doc.canvas.bg_color = (255, 0, 0, 255)
    assert doc.canvas.bg_color == (255, 0, 0, 255)
    doc.history.undo()
    assert doc.canvas.bg_color == initial_bg
    doc.history.redo()
    assert doc.canvas.bg_color == (255, 0, 0, 255)

    doc.canvas.region = Region.from_size(1920, 1080)
    assert doc.canvas.region == Region.from_size(1920, 1080)
    doc.history.undo()
    assert doc.canvas.region == initial_region
    doc.history.redo()
    assert doc.canvas.region == Region.from_size(1920, 1080)


def test_proxy_canvas_layout_resize_bounds_undo_redo():
    """Garante que canvas.layout.resize_bounds e doc.layout.resize_bounds(canvas) suportam Undo e Redo."""
    doc = Document("CanvasLayoutTest", 1000, 1000, history=True)
    assert doc.history is not None
    initial_region = doc.canvas.region

    doc.canvas.layout.resize_bounds(500, 500)
    assert doc.canvas.region == Region.from_rect(250, 250, 500, 500)
    doc.history.undo()
    assert doc.canvas.region == initial_region
    doc.history.redo()
    assert doc.canvas.region == Region.from_rect(250, 250, 500, 500)

    doc.layout.resize_bounds(doc.canvas, 800, 800)
    assert doc.canvas.region == Region.from_rect(100, 100, 800, 800)
    doc.history.undo()
    assert doc.canvas.region == Region.from_rect(250, 250, 500, 500)


def test_proxy_layer_content_resize_and_crop_undo_redo():
    """Garante que layer.content.resize e layer.content.crop suportam Undo e Redo atomicamente."""
    doc = Document("LayerContentTest", 1000, 1000, history=True)
    assert doc.history is not None
    layer = doc.add(make_layer(size=(200, 200), name="L1"))
    doc.history._undo_stack.clear()

    layer.content.resize(100, 150)
    assert layer.global_region.size == (100.0, 150.0)
    doc.history.undo()
    assert layer.global_region.size == (200.0, 200.0)
    doc.history.redo()
    assert layer.global_region.size == (100.0, 150.0)
    doc.history.undo()

    layer.content.crop((10, 10, 50, 50))
    assert layer.global_region == Region.from_rect(10, 10, 50, 50)
    doc.history.undo()
    assert layer.global_region == Region.from_size(200, 200)
    doc.history.redo()
    assert layer.global_region == Region.from_rect(10, 10, 50, 50)


def test_group_proxy_layout_and_content_undo_redo():
    """Garante que group.layout.align e group.content.resize suportam Undo e Redo."""
    doc = Document("GroupOpsTest", 1000, 1000, history=True)
    assert doc.history is not None
    g = doc.add_group("G1")
    l1 = doc.add(make_layer(size=(200, 200), name="l1"))
    l2 = doc.add(make_layer(size=(300, 300), name="l2"))
    l2.region = Region.from_rect(100, 100, 300, 300)
    g.append(l1)
    g.append(l2)
    doc.history._undo_stack.clear()

    initial_group_region = g.global_region
    g.layout.align((0, 0, 1000, 1000), anchor_x=1.0, anchor_y=1.0)
    assert g.global_region == Region.from_rect(600, 600, 400, 400)
    doc.history.undo()
    assert g.global_region == initial_group_region
    doc.history.redo()
    assert g.global_region == Region.from_rect(600, 600, 400, 400)

    g.content.resize(600, 600)
    assert g.global_region == Region.from_rect(500, 500, 600, 600)
    doc.history.undo()
    assert g.global_region == Region.from_rect(600, 600, 400, 400)
    doc.history.redo()
    assert g.global_region == Region.from_rect(500, 500, 600, 600)


def test_document_pure_layout_and_content_instances():
    """Garante que doc.layout e doc.content sao instancias puras que delegam para proxies alvos."""
    doc = Document("DocTest", 1000, 1000, history=True)
    assert doc.history is not None
    layer = doc.add(make_layer(size=(200, 200), name="L1"))
    doc.history._undo_stack.clear()

    assert type(doc.layout) is Layout
    assert type(doc.content) is Content
    assert not isinstance(doc.layout, BaseHistoryProxy)
    assert not isinstance(doc.content, BaseHistoryProxy)

    doc.layout.align(layer, doc.canvas.region, anchor_x=1.0, anchor_y=1.0)
    assert layer.region == Region.from_rect(800, 800, 200, 200)
    doc.history.undo()
    assert layer.region == Region.from_rect(0, 0, 200, 200)
