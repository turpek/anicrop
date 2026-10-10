from __future__ import annotations

import numpy as np

from anicrop.document import Document
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer


def make_layer(name: str = "Layer") -> Layer:
    img = Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA)
    return Layer(img, name=name)


def test_proxy_composer_copy_from_records_and_undo_redo():
    """Valida se ProxyComposer.copy_from grava historico e suporta Undo e Redo."""
    doc = Document("DocComposer", 200, 200, history=True)
    l1 = doc.add(make_layer("l1"))
    l2 = doc.add(make_layer("l2"))

    l1.transform.translate(50, 80)
    l2.transform.copy_from(l1.transform)
    assert np.allclose(l2.matrix[:2, 2], [50.0, 80.0])

    doc.history.undo()
    assert np.allclose(l2.matrix[:2, 2], [0.0, 0.0])

    doc.history.redo()
    assert np.allclose(l2.matrix[:2, 2], [50.0, 80.0])


def test_container_proxy_move_to_front_and_back_undo_redo():
    """Valida se move_to_front e move_to_back em GroupProxy suportam Undo e Redo."""
    doc = Document("DocGroup", 200, 200, history=True)
    g = doc.add_group("G1")
    l1 = make_layer("l1")
    l2 = make_layer("l2")
    l3 = make_layer("l3")
    g.append(l1)
    g.append(l2)
    g.append(l3)

    g.move_to_front(l1)
    assert [c.name for c in g] == ["l2", "l3", "l1"]

    doc.history.undo()
    assert [c.name for c in g] == ["l1", "l2", "l3"]

    doc.history.redo()
    assert [c.name for c in g] == ["l2", "l3", "l1"]

    g.move_to_back(l1)
    assert [c.name for c in g] == ["l1", "l2", "l3"]

    doc.history.undo()
    assert [c.name for c in g] == ["l2", "l3", "l1"]


def test_container_proxy_move_relative_undo_redo():
    """Valida se move_relative em GroupProxy suporta deslocamentos com Undo e Redo."""
    doc = Document("DocGroup", 200, 200, history=True)
    g = doc.add_group("G1")
    l1 = make_layer("l1")
    l2 = make_layer("l2")
    l3 = make_layer("l3")
    g.append(l1)
    g.append(l2)
    g.append(l3)

    g.move_relative(l1, 1)
    assert [c.name for c in g] == ["l2", "l1", "l3"]

    doc.history.undo()
    assert [c.name for c in g] == ["l1", "l2", "l3"]

    doc.history.redo()
    assert [c.name for c in g] == ["l2", "l1", "l3"]


def test_container_proxy_swap_undo_redo():
    """Valida se swap entre duas camadas em GroupProxy suporta Undo e Redo."""
    doc = Document("DocGroup", 200, 200, history=True)
    g = doc.add_group("G1")
    l1 = make_layer("l1")
    l2 = make_layer("l2")
    l3 = make_layer("l3")
    g.append(l1)
    g.append(l2)
    g.append(l3)

    g.swap(l1, l3)
    assert [c.name for c in g] == ["l3", "l2", "l1"]

    doc.history.undo()
    assert [c.name for c in g] == ["l1", "l2", "l3"]

    doc.history.redo()
    assert [c.name for c in g] == ["l3", "l2", "l1"]


def test_container_proxy_reverse_undo_redo():
    """Valida se reverse em GroupProxy inverte a ordem com suporte a Undo e Redo."""
    doc = Document("DocGroup", 200, 200, history=True)
    g = doc.add_group("G1")
    l1 = make_layer("l1")
    l2 = make_layer("l2")
    g.append(l1)
    g.append(l2)

    g.reverse()
    assert [c.name for c in g] == ["l2", "l1"]

    doc.history.undo()
    assert [c.name for c in g] == ["l1", "l2"]

    doc.history.redo()
    assert [c.name for c in g] == ["l2", "l1"]


def test_container_proxy_clear_atomic_undo_redo():
    """Valida se clear em GroupProxy remove filhos atomicamente com 1 unico Undo e Redo."""
    doc = Document("DocGroup", 200, 200, history=True)
    g = doc.add_group("G1")
    l1 = make_layer("l1")
    l2 = make_layer("l2")
    g.append(l1)
    g.append(l2)

    g.clear()
    assert len(g) == 0

    doc.history.undo()
    assert len(g) == 2
    assert [c.name for c in g] == ["l1", "l2"]

    doc.history.redo()
    assert len(g) == 0


def test_container_proxy_delitem_undo_redo():
    """Valida se del group[index] remove o elemento no indice com Undo e Redo."""
    doc = Document("DocGroup", 200, 200, history=True)
    g = doc.add_group("G1")
    l1 = make_layer("l1")
    l2 = make_layer("l2")
    g.append(l1)
    g.append(l2)

    del g[0]
    assert len(g) == 1
    assert g[0].name == "l2"

    doc.history.undo()
    assert len(g) == 2
    assert [c.name for c in g] == ["l1", "l2"]

    doc.history.redo()
    assert len(g) == 1
    assert g[0].name == "l2"


def test_container_delitem_direct_mode():
    """Valida se del container[index] funciona no modo direto sem historico."""
    doc = Document("DocDirect", 200, 200, history=False)
    g = doc.add_group("G1")
    l1 = make_layer("l1")
    l2 = make_layer("l2")
    g.append(l1)
    g.append(l2)

    del g[0]
    assert len(g) == 1
    assert g[0].name == "l2"


def test_layer_stack_proxy_clear_atomic_undo_redo():
    """Valida se doc.stack.clear() remove todas as camadas com 1 unico Undo e Redo."""
    doc = Document("DocStack", 200, 200, history=True)
    doc.add(make_layer("l1"))
    doc.add(make_layer("l2"))
    assert len(doc.stack) == 2

    doc.stack.clear()
    assert len(doc.stack) == 0

    doc.history.undo()
    assert len(doc.stack) == 2
    assert [c.name for c in doc.stack] == ["l1", "l2"]

    doc.history.redo()
    assert len(doc.stack) == 0
