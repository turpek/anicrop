from __future__ import annotations

import numpy as np
import pytest

from anicrop.edit_layer import EditLayer, EditStack
from anicrop.enums import BlendMode, ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.spatial import Region


def make_dummy_edit(name: str = "Edit", visible: bool = True) -> EditLayer:
    img = Image.new((10, 10), ImageFormat.RGBA)
    region = Region.from_size(10, 10)
    matrix = np.identity(3, dtype=np.float32)
    return EditLayer(img, region, matrix, blend_mode=BlendMode.NORMAL, name=name, visible=visible)


def test_edit_stack_empty_initialization():
    """Valida se EditStack inicializa vazio e com tamanho zero."""
    stack = EditStack()
    assert len(stack) == 0
    assert list(stack) == []
    assert repr(stack) == "EditStack([])"


def test_edit_stack_add_appends_and_returns_edit():
    """Valida se add insere o patch na pilha e o retorna diretamente."""
    stack = EditStack()
    e1 = make_dummy_edit(name="E1")
    ret = stack.add(e1)
    assert ret is e1
    assert len(stack) == 1
    assert stack[0] is e1
    assert e1 in stack


def test_edit_stack_add_type_validation():
    """Valida se add lanca TypeError ao receber objeto que nao herda de EditLayer."""
    stack = EditStack()
    with pytest.raises(TypeError, match="Expected EditLayer"):
        stack.add("not_an_edit")  # type: ignore[arg-type]


def test_edit_stack_indexing_by_int_slice_and_name():
    """Valida indexacao por int, slice e busca por nome."""
    stack = EditStack()
    e1 = make_dummy_edit(name="Alpha")
    e2 = make_dummy_edit(name="Beta")
    stack.extend([e1, e2])

    assert stack[0] is e1
    assert stack[0:2] == [e1, e2]
    assert stack["Beta"] is e2
    assert "Alpha" in stack
    assert "Omega" not in stack

    with pytest.raises(KeyError, match="Edit 'Omega' not found"):
        _ = stack["Omega"]


def test_edit_stack_remove_by_instance_and_name():
    """Valida remocao tanto por referencia do objeto quanto por nome."""
    stack = EditStack()
    e1 = make_dummy_edit(name="E1")
    e2 = make_dummy_edit(name="E2")
    stack.extend([e1, e2])

    stack.remove(e1)
    assert len(stack) == 1
    assert stack[0] is e2

    stack.remove("E2")
    assert len(stack) == 0

    with pytest.raises(ValueError):
        stack.remove("Inexistente")


def test_edit_stack_delitem_by_index_and_name():
    """Valida operador del com indice inteiro e nome de patch."""
    stack = EditStack()
    e1 = make_dummy_edit(name="E1")
    e2 = make_dummy_edit(name="E2")
    stack.extend([e1, e2])

    del stack["E1"]
    assert len(stack) == 1
    assert stack[0] is e2

    del stack[0]
    assert len(stack) == 0

    with pytest.raises(TypeError):
        del stack[3.14]  # type: ignore[arg-type]


def test_edit_stack_pop_and_clear():
    """Valida desempilhamento via pop e limpeza total com clear."""
    stack = EditStack()
    e1 = make_dummy_edit(name="E1")
    e2 = make_dummy_edit(name="E2")
    stack.extend([e1, e2])

    popped = stack.pop()
    assert popped is e2
    assert len(stack) == 1

    stack.clear()
    assert len(stack) == 0


def test_edit_stack_move_reorders_correctly():
    """Valida movimentacao de edicoes por objeto e por nome."""
    stack = EditStack()
    e1 = make_dummy_edit(name="E1")
    e2 = make_dummy_edit(name="E2")
    e3 = make_dummy_edit(name="E3")
    stack.extend([e1, e2, e3])

    stack.move("E1", 2)
    assert list(stack) == [e2, e3, e1]

    stack.move(e3, 0)
    assert list(stack) == [e3, e2, e1]


def test_edit_stack_swap_by_instances_and_indices():
    """Valida troca atomica de posicoes entre dois patches."""
    stack = EditStack()
    e1 = make_dummy_edit(name="E1")
    e2 = make_dummy_edit(name="E2")
    stack.extend([e1, e2])

    stack.swap(0, 1)
    assert list(stack) == [e2, e1]

    stack.swap(e2, e1)
    assert list(stack) == [e1, e2]


def test_edit_stack_close_calls_all_edits_close(mocker):
    """Valida se stack.close fecha todas as edicoes da colecao."""
    stack = EditStack()
    e1 = make_dummy_edit(name="E1")
    e2 = make_dummy_edit(name="E2")
    stack.extend([e1, e2])

    spy1 = mocker.spy(e1, "close")
    spy2 = mocker.spy(e2, "close")

    stack.close()
    assert spy1.call_count == 1
    assert spy2.call_count == 1


def test_layer_edits_property_returns_edit_stack():
    """Valida se layer.edits entrega uma instancia de EditStack funcional."""
    img = Image.new((100, 100), ImageFormat.RGBA)
    layer = Layer(img)

    assert isinstance(layer.edits, EditStack)
    assert len(layer.edits) == 1

    new_edit = make_dummy_edit(name="PatchExtra")
    layer.edits.add(new_edit)

    assert len(layer.edits) == 2
    assert layer.edits[1] is new_edit
    assert layer.edits["PatchExtra"] is new_edit
