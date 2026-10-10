from __future__ import annotations

import numpy as np

from anicrop.cache import LayerCache
from anicrop.document import Document
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer


def make_solid_image(
    w: int = 50,
    h: int = 50,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
) -> Image:
    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[:] = color
    return Image(data, ImageFormat.RGBA)


def test_render_com_cache_nao_polui_historico_com_comandos_fantasmas():
    """Valida se a renderizacao com cache nao gera comandos fantasmas no historico."""
    doc = Document("test_doc", 100, 100, history=True)
    doc.add(Layer(make_solid_image(), name="L1"))
    cache = LayerCache()
    cache.register(doc.stack)
    initial_cmds = len(doc.history._undo_stack)

    doc.render(cache=cache)

    assert len(doc.history._undo_stack) == initial_cmds


def test_undo_apos_render_com_cache_reverte_transformacao():
    """Valida se undo apos render com cache reverte a transformacao e os pixels com exatidao."""
    doc = Document("test_doc", 100, 100, history=True)
    layer = doc.add(Layer(make_solid_image(), name="L1"))
    cache = LayerCache()
    cache.register(doc.stack)

    img_initial = doc.render(cache=cache)
    layer.transform.rotate(45)
    img_rotated = doc.render(cache=cache)

    doc.history.undo()
    img_reverted = doc.render(cache=cache)

    assert not np.array_equal(img_initial[:], img_rotated[:])
    assert np.array_equal(img_initial[:], img_reverted[:])


def test_undo_redo_ciclo_completo_com_cache():
    """Valida se ciclo completo de undo e redo sincroniza os pixels com cache incremental."""
    doc = Document("test_doc", 100, 100, history=True)
    layer = doc.add(Layer(make_solid_image(), name="L1"))
    cache = LayerCache()
    cache.register(doc.stack)

    img1 = doc.render(cache=cache)
    layer.transform.translate(20, 10)
    img2 = doc.render(cache=cache)

    doc.history.undo()
    img_undo = doc.render(cache=cache)
    assert np.array_equal(img1[:], img_undo[:])

    doc.history.redo()
    img_redo = doc.render(cache=cache)
    assert np.array_equal(img2[:], img_redo[:])


def test_proxy_ignora_atributos_privados_sem_gravar_historico():
    """Valida se atribuicao a atributos privados no proxy nao cria comandos de historico."""
    doc = Document("test_doc", 100, 100, history=True)
    layer = doc.add(Layer(make_solid_image(), name="L1"))
    initial_cmds = len(doc.history._undo_stack)

    layer._opacity_mask = np.zeros((32, 32), dtype=np.uint8)

    assert len(doc.history._undo_stack) == initial_cmds
    assert layer._opacity_mask is not None


def test_cache_desempacota_proxies_corretamente():
    """Valida se LayerCache desempacota proxies e gerencia estado na camada real de dominio."""
    doc = Document("test_doc", 100, 100, history=True)
    layer = doc.add(Layer(make_solid_image(), name="L1"))
    cache = LayerCache()

    cache.register(layer)

    state = cache.get_state(layer)
    assert state is not None
    assert cache.is_dirty(layer) is True
