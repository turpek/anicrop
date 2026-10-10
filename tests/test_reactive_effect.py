from typing import Any

import numpy as np

from anicrop.document import Document
from anicrop.effect import BoundEffect, Effect, EffectStack
from anicrop.enums import ImageFormat
from anicrop.filter import BlurFilter
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.reactive.effect import ProxyEffect, ProxyEffectStack


class DummyTestEffect(Effect):
    """Efeito concreto auxiliar para testes de reatividade."""

    def __init__(self, visible: bool = True, name: str = "DummyTestEffect", strength: float = 1.0):
        super().__init__(visible=visible, name=name)
        self.strength = strength

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        return image

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None


def make_doc_with_layer() -> tuple[Document, Any]:
    doc = Document("TestDoc", 100, 100, history=True)
    img = Image(np.zeros((50, 50, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = doc.add(Layer(img, name="Layer1"))
    return doc, layer


def test_proxy_effect_stack_identity_and_type():
    """Valida se layer.effects em documento reativo entrega ProxyEffectStack e conforma a EffectStack."""
    doc, layer = make_doc_with_layer()

    assert isinstance(layer.effects, ProxyEffectStack)
    assert isinstance(layer.effects, EffectStack)


def test_proxy_effect_stack_add_undo_redo():
    """Valida adicao de efeito no pipeline com reversao e reaplicacao no historico."""
    doc, layer = make_doc_with_layer()
    eff = DummyTestEffect(name="E1")

    res = layer.effects.add(eff)

    assert len(layer.effects) == 1
    assert isinstance(res, ProxyEffect)
    assert layer.effects[0].name == "E1"

    doc.history.undo()
    assert len(layer.effects) == 0

    doc.history.redo()
    assert len(layer.effects) == 1
    assert layer.effects[0].name == "E1"


def test_proxy_effect_stack_remove_undo_redo():
    """Valida remocao de efeito por instancia e reversao no historico."""
    doc, layer = make_doc_with_layer()
    e1 = DummyTestEffect(name="E1")
    e2 = DummyTestEffect(name="E2")
    layer.effects.add(e1)
    layer.effects.add(e2)

    layer.effects.remove(e1)

    assert len(layer.effects) == 1
    assert layer.effects[0].name == "E2"

    doc.history.undo()
    assert len(layer.effects) == 2
    assert layer.effects[0].name == "E1"
    assert layer.effects[1].name == "E2"

    doc.history.redo()
    assert len(layer.effects) == 1
    assert layer.effects[0].name == "E2"


def test_proxy_effect_stack_clear_and_pop_undo_redo():
    """Valida limpeza total e pop de elementos com suporte a Undo/Redo."""
    doc, layer = make_doc_with_layer()
    e1 = DummyTestEffect(name="E1")
    e2 = DummyTestEffect(name="E2")
    layer.effects.add(e1)
    layer.effects.add(e2)

    popped = layer.effects.pop()
    assert isinstance(popped, ProxyEffect)
    assert popped.name == "E2"
    assert len(layer.effects) == 1

    doc.history.undo()
    assert len(layer.effects) == 2

    layer.effects.clear()
    assert len(layer.effects) == 0

    doc.history.undo()
    assert len(layer.effects) == 2
    assert layer.effects[0].name == "E1"
    assert layer.effects[1].name == "E2"

    doc.history.redo()
    assert len(layer.effects) == 0


def test_proxy_effect_stack_insert_move_swap_undo_redo():
    """Valida insercao, reordenacao e troca de posicoes com Undo/Redo."""
    doc, layer = make_doc_with_layer()
    e1 = DummyTestEffect(name="E1")
    e2 = DummyTestEffect(name="E2")
    e3 = DummyTestEffect(name="E3")
    layer.effects.add(e1)
    layer.effects.add(e2)

    layer.effects.insert(0, e3)
    assert [e.name for e in layer.effects] == ["E3", "E1", "E2"]

    doc.history.undo()
    assert [e.name for e in layer.effects] == ["E1", "E2"]

    doc.history.redo()
    assert [e.name for e in layer.effects] == ["E3", "E1", "E2"]

    layer.effects.swap(0, 2)
    assert [e.name for e in layer.effects] == ["E2", "E1", "E3"]

    doc.history.undo()
    assert [e.name for e in layer.effects] == ["E3", "E1", "E2"]

    doc.history.redo()
    assert [e.name for e in layer.effects] == ["E2", "E1", "E3"]

    layer.effects.move(e2, 2)
    assert [e.name for e in layer.effects] == ["E1", "E3", "E2"]

    doc.history.undo()
    assert [e.name for e in layer.effects] == ["E2", "E1", "E3"]

    doc.history.redo()
    assert [e.name for e in layer.effects] == ["E1", "E3", "E2"]


def test_proxy_effect_stack_delitem_undo_redo():
    """Valida operador del com indice inteiro e reversao no historico."""
    doc, layer = make_doc_with_layer()
    e1 = DummyTestEffect(name="E1")
    e2 = DummyTestEffect(name="E2")
    layer.effects.add(e1)
    layer.effects.add(e2)

    del layer.effects[0]
    assert len(layer.effects) == 1
    assert layer.effects[0].name == "E2"

    doc.history.undo()
    assert len(layer.effects) == 2
    assert layer.effects[0].name == "E1"
    assert layer.effects[1].name == "E2"

    doc.history.redo()
    assert len(layer.effects) == 1
    assert layer.effects[0].name == "E2"


def test_proxy_effect_getitem_and_iteration_returns_proxies():
    """Valida se acesso indexado, por nome e iteracao sempre devolvem instâncias de ProxyEffect."""
    doc, layer = make_doc_with_layer()
    e1 = DummyTestEffect(name="Alpha")
    e2 = DummyTestEffect(name="Beta")
    layer.effects.add(e1)
    layer.effects.add(e2)

    by_idx = layer.effects[0]
    by_name = layer.effects["Beta"]
    iter_items = list(layer.effects)

    assert isinstance(by_idx, ProxyEffect)
    assert isinstance(by_name, ProxyEffect)
    assert all(isinstance(x, ProxyEffect) for x in iter_items)
    assert by_idx.name == "Alpha"
    assert by_name.name == "Beta"


def test_proxy_effect_parameter_mutation_undo_redo():
    """Valida mutacao em propriedades escalares do efeito com registro de delta e Undo/Redo."""
    doc, layer = make_doc_with_layer()
    eff = DummyTestEffect(name="Mutable", visible=True, strength=1.0)
    layer.effects.add(eff)

    proxy_eff = layer.effects[0]
    proxy_eff.visible = False
    proxy_eff.strength = 2.5

    assert proxy_eff.visible is False
    assert proxy_eff.strength == 2.5

    doc.history.undo()
    assert proxy_eff.strength == 1.0

    doc.history.undo()
    assert proxy_eff.visible is True

    doc.history.redo()
    assert proxy_eff.visible is False

    doc.history.redo()
    assert proxy_eff.strength == 2.5


def test_proxy_effect_filter_attribute_mutation_undo_redo():
    """Valida mutacao em parametros especificos de BlurFilter com suporte a Undo/Redo."""
    doc, layer = make_doc_with_layer()
    blur = BlurFilter(radius=4.0)
    layer.effects.add(blur)

    proxy_blur = layer.effects[0]
    proxy_blur.radius_x = 12.0

    assert proxy_blur.radius_x == 12.0

    doc.history.undo()
    assert proxy_blur.radius_x == 4.0

    doc.history.redo()
    assert proxy_blur.radius_x == 12.0


def test_proxy_effect_bound_effect_from_layer_reactive():
    """Valida envelopamento BoundEffect.from_layer em ambiente reativo com controle de historico."""
    doc, layer = make_doc_with_layer()
    blur = BlurFilter(radius=2.0)
    bound = BoundEffect.from_layer(layer, blur, visible=True)
    layer.effects.add(bound)

    proxy_bound = layer.effects[0]

    assert isinstance(proxy_bound, ProxyEffect)
    assert proxy_bound.visible is True

    proxy_bound.visible = False
    assert proxy_bound.visible is False

    doc.history.undo()
    assert proxy_bound.visible is True

    doc.history.redo()
    assert proxy_bound.visible is False
