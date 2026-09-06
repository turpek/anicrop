from __future__ import annotations

import numpy as np

from anicrop.container import GroupLayer, LayerStack, NullContainer
from anicrop.enums import ImageFormat
from anicrop.history import GlobalHistory
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.reactive import (
    GroupProxy,
    LayerStackProxy,
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
