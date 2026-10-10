from __future__ import annotations

import numpy as np
import pytest

from anicrop.effect import BoundEffect, Effect, EffectStack
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.transform import mat_inverse


class DummyEffect(Effect):
    def __init__(self, pad: int = 0, visible: bool = True, name: str = "Dummy"):
        super().__init__(visible=visible, name=name)
        self.pad = pad

    def get_padding(self) -> tuple[int, int, int, int]:
        return (self.pad, self.pad, self.pad, self.pad) if self.visible else (0, 0, 0, 0)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        return image

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None


def test_effect_stack_empty_initialization():
    """Valida se EffectStack inicializa vazio e com tamanho zero."""
    stack = EffectStack()
    assert len(stack) == 0
    assert list(stack) == []
    assert repr(stack) == "EffectStack([])"


def test_effect_stack_add_appends_and_returns_effect():
    """Valida se add insere o efeito na pilha e o retorna diretamente."""
    stack = EffectStack()
    e1 = DummyEffect(name="E1")
    ret = stack.add(e1)
    assert ret is e1
    assert len(stack) == 1
    assert stack[0] is e1
    assert e1 in stack


def test_effect_stack_add_type_validation():
    """Valida se add lanca TypeError ao receber objeto que nao herda de Effect."""
    stack = EffectStack()
    with pytest.raises(TypeError, match="Expected Effect"):
        stack.add("not_an_effect")  # type: ignore[arg-type]


def test_bound_effect_from_layer_anchors_to_inverse_matrix():
    """Valida se BoundEffect.from_layer ancora o efeito a matriz inversa da camada."""
    layer = Layer(Image.new((100, 100), ImageFormat.RGBA))
    layer.transform.translate(10, 20).rotate(45)

    e = DummyEffect(name="Inner")
    bound = BoundEffect.from_layer(layer, e)

    assert isinstance(bound, BoundEffect)
    assert bound.effect is e
    assert np.allclose(bound.matrix, mat_inverse(layer.matrix))
    stack = EffectStack()
    stack.add(bound)
    assert bound in stack
    assert e in stack


def test_effect_stack_getitem_int_slice_and_name():
    """Valida indexacao por inteiro, slice e busca por nome."""
    stack = EffectStack()
    e1 = DummyEffect(name="Alpha")
    e2 = DummyEffect(name="Beta")
    e3 = DummyEffect(name="Gamma")
    stack.add(e1)
    stack.add(e2)
    stack.add(e3)

    assert stack[0] is e1
    assert stack[-1] is e3
    assert stack[0:2] == [e1, e2]
    assert stack["Beta"] is e2
    assert "Alpha" in stack
    assert "Omega" not in stack

    with pytest.raises(KeyError, match="Effect 'Omega' not found"):
        _ = stack["Omega"]


def test_effect_stack_remove_by_instance_and_name():
    """Valida remocao tanto por referencia do objeto quanto por nome."""
    stack = EffectStack()
    e1 = DummyEffect(name="E1")
    e2 = DummyEffect(name="E2")
    stack.add(e1)
    stack.add(e2)

    stack.remove(e1)
    assert len(stack) == 1
    assert stack[0] is e2

    stack.remove("E2")
    assert len(stack) == 0

    with pytest.raises(ValueError):
        stack.remove("Inexistente")


def test_effect_stack_delitem_by_index_and_name():
    """Valida operador del com indice inteiro e nome de efeito."""
    stack = EffectStack()
    e1 = DummyEffect(name="E1")
    e2 = DummyEffect(name="E2")
    stack.add(e1)
    stack.add(e2)

    del stack[0]
    assert len(stack) == 1
    assert stack[0] is e2

    del stack["E2"]
    assert len(stack) == 0


def test_effect_stack_clear_and_pop():
    """Valida limpeza total e pop de elementos."""
    stack = EffectStack()
    e1 = DummyEffect(name="E1")
    e2 = DummyEffect(name="E2")
    stack.add(e1)
    stack.add(e2)

    popped = stack.pop()
    assert popped is e2
    assert len(stack) == 1

    stack.clear()
    assert len(stack) == 0


def test_effect_stack_insert_move_and_swap():
    """Valida insercao em posicao arbitraria, movimentacao e troca de posicoes."""
    stack = EffectStack()
    e1 = DummyEffect(name="E1")
    e2 = DummyEffect(name="E2")
    e3 = DummyEffect(name="E3")
    stack.add(e1)
    stack.add(e2)

    stack.insert(0, e3)
    assert list(stack) == [e3, e1, e2]

    stack.move(e3, 2)
    assert list(stack) == [e1, e2, e3]

    stack.swap(0, 2)
    assert list(stack) == [e3, e2, e1]

    stack.swap(e3, e1)
    assert list(stack) == [e1, e2, e3]


def test_effect_stack_get_padding_aggregates_max():
    """Valida calculo de margem agregada maxima entre todos os efeitos visiveis."""
    stack = EffectStack()
    stack.add(DummyEffect(pad=5, visible=True))
    stack.add(DummyEffect(pad=12, visible=True))
    stack.add(DummyEffect(pad=20, visible=False))

    assert stack.get_padding() == (12, 12, 12, 12)
