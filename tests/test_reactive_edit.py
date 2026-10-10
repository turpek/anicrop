from __future__ import annotations

from typing import Any

import numpy as np

from anicrop.document import Document
from anicrop.edit_layer import EditLayer, EditStack
from anicrop.enums import BlendMode, ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.reactive.edit import ProxyEdit, ProxyEditStack
from anicrop.spatial import Region


def make_dummy_patch(name: str = "Patch", visible: bool = True) -> EditLayer:
    img = Image(np.zeros((10, 10, 4), dtype=np.uint8), ImageFormat.RGBA)
    region = Region.from_size(10, 10)
    matrix = np.identity(3, dtype=np.float32)
    return EditLayer(img, region, matrix, blend_mode=BlendMode.NORMAL, name=name, visible=visible)


def make_doc_with_layer() -> tuple[Document, Any]:
    doc = Document("TestDoc", 100, 100, history=True)
    img = Image(np.zeros((50, 50, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = doc.add(Layer(img, name="Layer1"))
    return doc, layer


def test_proxy_edit_stack_identity_and_type():
    """Valida se layer.edits em documento reativo entrega ProxyEditStack e conforma a EditStack."""
    doc, layer = make_doc_with_layer()

    assert isinstance(layer.edits, ProxyEditStack)
    assert isinstance(layer.edits, EditStack)
    assert isinstance(layer.edits[0], ProxyEdit)
    assert isinstance(layer.edits[0], EditLayer)


def test_proxy_edit_stack_add_undo_redo():
    """Valida adicao de patch na pilha com reversao e reaplicacao no historico."""
    doc, layer = make_doc_with_layer()
    patch = make_dummy_patch(name="P1")

    layer.edits.add(patch)
    assert len(layer.edits) == 2
    assert layer.edits[1].name == "P1"

    doc.history.undo()
    assert len(layer.edits) == 1

    doc.history.redo()
    assert len(layer.edits) == 2
    assert layer.edits[1].name == "P1"


def test_proxy_edit_stack_remove_undo_redo():
    """Valida remocao de patch com reversao e reaplicacao no historico."""
    doc, layer = make_doc_with_layer()
    patch = make_dummy_patch(name="P1")
    layer.edits.add(patch)

    layer.edits.remove("P1")
    assert len(layer.edits) == 1

    doc.history.undo()
    assert len(layer.edits) == 2
    assert layer.edits[1].name == "P1"

    doc.history.redo()
    assert len(layer.edits) == 1


def test_proxy_edit_stack_pop_undo_redo():
    """Valida remocao via pop com reversao e reaplicacao no historico."""
    doc, layer = make_doc_with_layer()
    patch = make_dummy_patch(name="P1")
    layer.edits.add(patch)

    popped = layer.edits.pop()
    assert popped.name == "P1"
    assert len(layer.edits) == 1

    doc.history.undo()
    assert len(layer.edits) == 2

    doc.history.redo()
    assert len(layer.edits) == 1


def test_proxy_edit_stack_clear_undo_redo():
    """Valida limpeza total de edits com reversao atomica no historico."""
    doc, layer = make_doc_with_layer()
    patch = make_dummy_patch(name="P1")
    layer.edits.add(patch)

    layer.edits.clear()
    assert len(layer.edits) == 0

    doc.history.undo()
    assert len(layer.edits) == 2

    doc.history.redo()
    assert len(layer.edits) == 0


def test_proxy_edit_stack_move_undo_redo():
    """Valida movimentacao de posicao entre edits com reversao e reaplicacao."""
    doc, layer = make_doc_with_layer()
    p1 = make_dummy_patch(name="P1")
    p2 = make_dummy_patch(name="P2")
    layer.edits.extend([p1, p2])

    layer.edits.move("P1", 2)
    assert [e.name for e in layer.edits] == ["EditLayer", "P2", "P1"]

    doc.history.undo()
    assert [e.name for e in layer.edits] == ["EditLayer", "P1", "P2"]

    doc.history.redo()
    assert [e.name for e in layer.edits] == ["EditLayer", "P2", "P1"]


def test_proxy_edit_stack_swap_undo_redo():
    """Valida troca atomica de posicoes entre edits com suporte a undo e redo."""
    doc, layer = make_doc_with_layer()
    p1 = make_dummy_patch(name="P1")
    p2 = make_dummy_patch(name="P2")
    layer.edits.extend([p1, p2])

    layer.edits.swap(1, 2)
    assert [e.name for e in layer.edits] == ["EditLayer", "P2", "P1"]

    doc.history.undo()
    assert [e.name for e in layer.edits] == ["EditLayer", "P1", "P2"]

    doc.history.redo()
    assert [e.name for e in layer.edits] == ["EditLayer", "P2", "P1"]


def test_proxy_edit_stack_delitem_undo_redo():
    """Valida operador del em ProxyEditStack com reversao no historico."""
    doc, layer = make_doc_with_layer()
    patch = make_dummy_patch(name="P1")
    layer.edits.add(patch)

    del layer.edits["P1"]
    assert len(layer.edits) == 1

    doc.history.undo()
    assert len(layer.edits) == 2
    assert layer.edits[1].name == "P1"

    doc.history.redo()
    assert len(layer.edits) == 1


def test_proxy_edit_mutate_properties_undo_redo():
    """Valida alteracao de propriedades em EditLayer rastreadas via AdaptiveCommand."""
    doc, layer = make_doc_with_layer()
    edit = layer.edits[0]

    edit.visible = False
    assert edit.visible is False

    doc.history.undo()
    assert edit.visible is True

    doc.history.redo()
    assert edit.visible is False

    edit.blend_mode = BlendMode.MULTIPLY
    assert edit.blend_mode == BlendMode.MULTIPLY

    doc.history.undo()
    assert edit.blend_mode == BlendMode.NORMAL

    doc.history.redo()
    assert edit.blend_mode == BlendMode.MULTIPLY


def test_proxy_layer_add_edit_convenience_method():
    """Valida conveniencia layer.add_edit em documento reativo com historico funcional."""
    doc, layer = make_doc_with_layer()
    patch_img = Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA)

    new_edit = layer.add_edit(patch_img, Region.from_size(20, 20), name="ConveniencePatch")
    assert len(layer.edits) == 2
    assert layer.edits[1] is new_edit

    doc.history.undo()
    assert len(layer.edits) == 1

    doc.history.redo()
    assert len(layer.edits) == 2
    assert layer.edits[1].name == "ConveniencePatch"
