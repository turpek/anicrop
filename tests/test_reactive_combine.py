from __future__ import annotations

import numpy as np

from anicrop.composition import Combine
from anicrop.document import Document
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.reactive.strategy import CombineProxy


def make_layer(color: tuple[int, ...] = (255, 0, 0, 255), name: str = "Layer") -> Layer:
    img = Image(np.full((20, 20, 4), color, dtype=np.uint8), ImageFormat.RGBA)
    return Layer(img, name=name)


def test_doc_combine_proxy_type_by_policy():
    """Valida se doc.combine retorna CombineProxy com history=True e Combine nativo com history=False."""
    doc_reactive = Document("ReactiveDoc", 200, 200, history=True)
    doc_direct = Document("DirectDoc", 200, 200, history=False)

    assert isinstance(doc_reactive.combine, CombineProxy)
    assert not isinstance(doc_direct.combine, CombineProxy)
    assert isinstance(doc_direct.combine, Combine)


def test_combine_merge_atomic_undo_redo():
    """Valida se doc.combine.merge agrupa remocoes e insercao em 1 unico Undo e Redo."""
    doc = Document("DocMerge", 200, 200, history=True)
    doc.add(make_layer(name="L1"))
    doc.add(make_layer(name="L2"))
    doc.add(make_layer(name="L3"))

    group = doc.combine.merge("L3", name="MergedG", count=1)
    assert len(doc.stack) == 2
    assert [c.name for c in doc.stack] == ["L1", "MergedG"]
    assert [c.name for c in group] == ["L2", "L3"]

    doc.history.undo()
    assert len(doc.stack) == 3
    assert [c.name for c in doc.stack] == ["L1", "L2", "L3"]
    assert doc.find("MergedG") is None

    doc.history.redo()
    assert len(doc.stack) == 2
    assert [c.name for c in doc.stack] == ["L1", "MergedG"]


def test_combine_flatten_atomic_undo_redo():
    """Valida se doc.combine.flatten substitui camadas atomicamente com Undo e Redo de 1 passo."""
    doc = Document("DocFlatten", 200, 200, history=True)
    doc.add(make_layer(name="L1"))
    doc.add(make_layer(name="L2"))
    doc.add(make_layer(name="L3"))

    flat = doc.combine.flatten("L3", name="FlatTop", count=1)
    assert len(doc.stack) == 2
    assert [c.name for c in doc.stack] == ["L1", "FlatTop"]
    assert flat.name == "FlatTop"

    doc.history.undo()
    assert len(doc.stack) == 3
    assert [c.name for c in doc.stack] == ["L1", "L2", "L3"]
    assert doc.find("FlatTop") is None

    doc.history.redo()
    assert len(doc.stack) == 2
    assert [c.name for c in doc.stack] == ["L1", "FlatTop"]


def test_combine_bake_group_atomic_undo_redo():
    """Valida se bake em GroupLayer restaura o grupo com filhos intactos ao desfazer."""
    doc = Document("DocBake", 200, 200, history=True)
    g = doc.add_group("G1")
    g.append(make_layer(name="Sub1"))
    g.append(make_layer(name="Sub2"))

    baked = doc.combine.bake("G1", name="BakedG")
    assert baked.name == "BakedG"
    assert len(doc.stack) == 1
    assert doc.stack[0].name == "BakedG"
    assert doc.find("G1") is None

    doc.history.undo()
    assert len(doc.stack) == 1
    restored_group = doc["G1"]
    assert restored_group.name == "G1"
    assert len(restored_group) == 2
    assert [c.name for c in restored_group] == ["Sub1", "Sub2"]
    assert doc.find("BakedG") is None

    doc.history.redo()
    assert len(doc.stack) == 1
    assert doc.stack[0].name == "BakedG"
    assert doc.find("G1") is None


def test_combine_bake_stack_atomic_undo_redo():
    """Valida se bake_stack achata a pilha inteira e restaura todas as camadas com 1 unico Undo."""
    doc = Document("DocBakeStack", 200, 200, history=True)
    doc.add(make_layer(name="L1"))
    doc.add(make_layer(name="L2"))
    doc.add(make_layer(name="L3"))

    doc.combine.bake_stack(name="FlatScene")
    assert len(doc.stack) == 1
    assert doc.stack[0].name == "FlatScene"

    doc.history.undo()
    assert len(doc.stack) == 3
    assert [c.name for c in doc.stack] == ["L1", "L2", "L3"]
    assert doc.find("FlatScene") is None

    doc.history.redo()
    assert len(doc.stack) == 1
    assert doc.stack[0].name == "FlatScene"


def test_combine_nested_group_merge_atomic_undo_redo():
    """Valida se merge dentro de um subgrupo aninhado restaura a hierarquia interna sob Undo."""
    doc = Document("DocNestedMerge", 200, 200, history=True)
    parent_g = doc.add_group("ParentG")
    sub0 = make_layer(name="Sub0")
    sub1 = make_layer(name="Sub1")
    sub2 = make_layer(name="Sub2")
    parent_g.append(sub0)
    parent_g.append(sub1)
    parent_g.append(sub2)

    inner = doc.combine.merge("Sub2", name="InnerG", count=1)
    assert len(parent_g) == 2
    assert [c.name for c in parent_g] == ["Sub0", "InnerG"]
    assert [c.name for c in inner] == ["Sub1", "Sub2"]

    doc.history.undo()
    assert len(parent_g) == 3
    assert [c.name for c in parent_g] == ["Sub0", "Sub1", "Sub2"]
    assert doc.find("InnerG") is None

    doc.history.redo()
    assert len(parent_g) == 2
    assert [c.name for c in parent_g] == ["Sub0", "InnerG"]
