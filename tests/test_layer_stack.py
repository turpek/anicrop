from anicrop.layer_stack import LayerStack
from anicrop.image import Image, ImageFormat
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import re


class DummyLayer:
    """Mock leve para focar apenas na ordenação da LayerStack."""

    def __init__(self, id_val):
        self.id = id_val

    def __repr__(self) -> str:
        return f'Layer(id={self.id})'


def test_LayerStack_add_pilha_vazia():
    stack = LayerStack()
    l1 = DummyLayer(1)
    stack.add(l1)
    assert len(stack) == 1
    assert stack.get(0) is l1


def test_LayerStack_add_no_topo_por_padrao():
    stack = LayerStack()
    l1 = DummyLayer(1)
    l2 = DummyLayer(2)
    stack.add(l1)
    stack.add(l2)
    assert len(stack) == 2
    assert stack.get(1) is l2
    assert [e.id for e in stack] == [1, 2]


def test_LayerStack_add_rejeita_instancia_duplicada():
    stack = LayerStack()
    l1 = DummyLayer(1)
    stack.add(l1)
    msg = "Layer instance already exists in the LayerStack. Use layer.clone() to duplicate."
    with pytest.raises(ValueError, match=re.escape(msg)):
        stack.add(l1)


@pytest.mark.parametrize(
    "index_to_add, expected_order",
    [(0, [3, 1, 2]), (1, [1, 3, 2]), (-1, [1, 3, 2]), (99, [1, 2, 3])],
    ids=["base_0", "meio_1", "neg_-1", "out_99"]
)
def test_LayerStack_add_com_indices_especificos(index_to_add, expected_order):
    stack = LayerStack()
    l1, l2, l3 = DummyLayer(1), DummyLayer(2), DummyLayer(3)
    stack.add(l1)
    stack.add(l2)
    stack.add(l3, index=index_to_add)
    assert [e.id for e in stack] == expected_order


@pytest.fixture
def populated_stack():
    stack = LayerStack()
    l1, l2, l3 = DummyLayer(1), DummyLayer(2), DummyLayer(3)
    stack.add(l1)
    stack.add(l2)
    stack.add(l3)
    return stack, l1, l2, l3


@pytest.mark.parametrize(
    "index, expected_ids",
    [(0, [2, 3]), (1, [1, 3]), (2, [1, 2]), (-1, [1, 2])],
    ids=["idx_0", "idx_1", "idx_2", "idx_-1"]
)
def test_LayerStack_remove_por_indice_caminho_feliz(populated_stack, index, expected_ids):
    stack, *layers = populated_stack
    removed = stack.remove(index)
    assert len(stack) == 2
    assert [e.id for e in stack] == expected_ids
    assert removed.id not in expected_ids


def test_LayerStack_remove_por_instancia_caminho_feliz(populated_stack):
    stack, l1, l2, l3 = populated_stack
    removed = stack.remove(l2)
    assert len(stack) == 2
    assert [e.id for e in stack] == [1, 3]
    assert removed is l2


@pytest.mark.parametrize("invalid_index", [5, -99])
def test_LayerStack_remove_indice_invalido(populated_stack, invalid_index):
    stack, *layers = populated_stack
    with pytest.raises(IndexError):
        stack.remove(invalid_index)
    assert len(stack) == 3


def test_LayerStack_remove_objeto_inexistente(populated_stack):
    stack, *layers = populated_stack
    with pytest.raises(ValueError):
        stack.remove(DummyLayer(99))
    assert len(stack) == 3


def test_LayerStack_remove_em_pilha_vazia():
    stack = LayerStack()
    with pytest.raises(IndexError):
        stack.remove(0)
    with pytest.raises(ValueError):
        stack.remove(DummyLayer(1))


@pytest.fixture
def indexed_stack():
    stack = LayerStack()
    l10, l20, l30 = DummyLayer(10), DummyLayer(20), DummyLayer(30)
    stack.add(l10)
    stack.add(l20)
    stack.add(l30)
    return stack, l10, l20, l30


@pytest.mark.parametrize("index, expected_id", [(0, 10), (1, 20), (2, 30)])
def test_LayerStack_get_indices_positivos(indexed_stack, index, expected_id):
    stack, *layers = indexed_stack
    assert stack.get(index).id == expected_id


@pytest.mark.parametrize("index, expected_id", [(-1, 30), (-3, 10)])
def test_LayerStack_get_indices_negativos(indexed_stack, index, expected_id):
    stack, *layers = indexed_stack
    assert stack.get(index).id == expected_id


@pytest.mark.parametrize("out_of_bounds_index", [3, 99, -4])
def test_LayerStack_get_indices_fora_dos_limites(indexed_stack, out_of_bounds_index):
    stack, *layers = indexed_stack
    with pytest.raises(IndexError):
        stack.get(out_of_bounds_index)


def test_LayerStack_get_em_pilha_vazia():
    stack = LayerStack()
    with pytest.raises(IndexError):
        stack.get(0)


@pytest.mark.parametrize(
    "idx_a, idx_b, expected_ids",
    [(0, 1, [20, 10, 30]), (0, 2, [30, 20, 10]),
     (-1, 0, [30, 20, 10]), (1, 1, [10, 20, 30])],
    ids=["adj", "dist", "neg", "idem"]
)
def test_LayerStack_swap_caminhos_felizes(indexed_stack, idx_a, idx_b, expected_ids):
    stack, *layers = indexed_stack
    stack.swap(idx_a, idx_b)
    assert [e.id for e in stack] == expected_ids


@pytest.mark.parametrize("idx_a, idx_b", [(0, 99), (-5, 1)])
def test_LayerStack_swap_indices_invalidos(indexed_stack, idx_a, idx_b):
    stack, *layers = indexed_stack
    original_ids = [e.id for e in stack]
    with pytest.raises(IndexError):
        stack.swap(idx_a, idx_b)
    assert [e.id for e in stack] == original_ids


@pytest.mark.parametrize(
    "index, expected_ids",
    [(0, [20, 10, 30]), (1, [10, 30, 20]), (-2, [10, 30, 20])],
    ids=["base_up", "meio_up", "neg_meio_up"]
)
def test_LayerStack_move_up_caminhos_felizes(indexed_stack, index, expected_ids):
    stack, *layers = indexed_stack
    stack.move_up(index)
    assert [e.id for e in stack] == expected_ids


@pytest.mark.parametrize("index", [2, -1])
def test_LayerStack_move_up_noop_no_topo(indexed_stack, index):
    stack, *layers = indexed_stack
    original_ids = [e.id for e in stack]
    stack.move_up(index)
    assert [e.id for e in stack] == original_ids


@pytest.mark.parametrize("invalid_index", [99, -5])
def test_LayerStack_move_up_indices_invalidos(indexed_stack, invalid_index):
    stack, *layers = indexed_stack
    with pytest.raises(IndexError):
        stack.move_up(invalid_index)


def test_LayerStack_move_up_em_pilha_vazia():
    stack = LayerStack()
    with pytest.raises(IndexError):
        stack.move_up(0)


@pytest.mark.parametrize(
    "index, expected_ids",
    [(2, [10, 30, 20]), (1, [20, 10, 30]), (-1, [10, 30, 20])],
    ids=["topo_down", "meio_down", "neg_topo_down"]
)
def test_LayerStack_move_down_caminhos_felizes(indexed_stack, index, expected_ids):
    stack, *layers = indexed_stack
    stack.move_down(index)
    assert [e.id for e in stack] == expected_ids


@pytest.mark.parametrize("index", [0, -3])
def test_LayerStack_move_down_noop_na_base(indexed_stack, index):
    stack, *layers = indexed_stack
    original_ids = [e.id for e in stack]
    stack.move_down(index)
    assert [e.id for e in stack] == original_ids


@pytest.mark.parametrize("invalid_index", [99, -5])
def test_LayerStack_move_down_indices_invalidos(indexed_stack, invalid_index):
    stack, *layers = indexed_stack
    with pytest.raises(IndexError):
        stack.move_down(invalid_index)


def test_LayerStack_move_down_em_pilha_vazia():
    stack = LayerStack()
    with pytest.raises(IndexError):
        stack.move_down(0)


@pytest.mark.parametrize(
    "index, expected_ids",
    [(0, [20, 30, 10]), (1, [10, 30, 20]), (-3, [20, 30, 10])],
    ids=["base_front", "meio_front", "neg_base_front"]
)
def test_LayerStack_move_to_front_caminhos_felizes(indexed_stack, index, expected_ids):
    stack, *layers = indexed_stack
    stack.move_to_front(index)
    assert [e.id for e in stack] == expected_ids


@pytest.mark.parametrize("index", [2, -1])
def test_LayerStack_move_to_front_noop_no_topo(indexed_stack, index):
    stack, *layers = indexed_stack
    original_ids = [e.id for e in stack]
    stack.move_to_front(index)
    assert [e.id for e in stack] == original_ids


@pytest.mark.parametrize("invalid_index", [99])
def test_LayerStack_move_to_front_indices_invalidos(indexed_stack, invalid_index):
    stack, *layers = indexed_stack
    with pytest.raises(IndexError):
        stack.move_to_front(invalid_index)


@pytest.mark.parametrize(
    "index, expected_ids",
    [(2, [30, 10, 20]), (-1, [30, 10, 20]),
     (1, [20, 10, 30]), (-2, [20, 10, 30])],
    ids=["topo_back", "neg_topo_back", "meio_back", "neg_meio_back"]
)
def test_LayerStack_move_to_back_caminhos_felizes(indexed_stack, index, expected_ids):
    stack, *layers = indexed_stack
    stack.move_to_back(index)
    assert [e.id for e in stack] == expected_ids


@pytest.mark.parametrize("index", [0, -3])
def test_LayerStack_move_to_back_noop_na_base(indexed_stack, index):
    stack, *layers = indexed_stack
    original_ids = [e.id for e in stack]
    stack.move_to_back(index)
    assert [e.id for e in stack] == original_ids


@pytest.mark.parametrize("invalid_index", [99])
def test_LayerStack_move_to_back_indices_invalidos(indexed_stack, invalid_index):
    stack, *layers = indexed_stack
    with pytest.raises(IndexError):
        stack.move_to_back(invalid_index)


@pytest.mark.skip()
@patch("anicrop.layer_stack.Render")
@patch("anicrop.layer_stack.EditLayer")
def test_LayerStack_merge_down_topo_para_meio(mock_edit_layer, mock_render_class, indexed_stack):
    """Cenário 1: Merge do Topo para o Meio (Caminho Feliz)"""
    stack, l10, l20, l30 = indexed_stack

    # Setup do Mock do Render: retorna imagem 5x5
    mock_render_inst = mock_render_class.return_value
    fake_img = Image(np.zeros((5, 5, 3), dtype=np.uint8), ImageFormat.RGB)
    mock_render_inst.render.return_value = fake_img

    # Setup do Mock do EditLayer resultante
    merged_mock = MagicMock()
    merged_mock.id = "merged_layer"
    mock_edit_layer.return_value = merged_mock

    # Ação: Merge do Topo (2) com o Meio (1)
    result = stack.merge_down(2)

    # Validação
    assert result is merged_mock
    assert len(stack) == 2
    assert stack.get(1) is merged_mock
    assert stack.get(0) is l10
    # Garante que renderizou as duas camadas envolvidas
    assert mock_render_inst.render.call_count == 2


@pytest.mark.skip()
@patch("anicrop.layer_stack.Render")
@patch("anicrop.layer_stack.EditLayer")
def test_LayerStack_merge_down_meio_para_base(mock_edit_layer, mock_render_class, indexed_stack):
    """Cenário 2: Merge do Meio para a Base (Caminho Feliz)"""
    stack, l10, l20, l30 = indexed_stack

    mock_render_inst = mock_render_class.return_value
    fake_img = Image(np.zeros((5, 5, 3), dtype=np.uint8), ImageFormat.RGB)
    mock_render_inst.render.return_value = fake_img

    merged_mock = MagicMock()
    merged_mock.id = "merged_base"
    mock_edit_layer.return_value = merged_mock

    # Ação: Merge do Meio (1) com a Base (0)
    result = stack.merge_down(1)

    assert result is merged_mock
    assert len(stack) == 2
    assert stack.get(0) is merged_mock
    assert stack.get(1) is l30


@pytest.mark.skip()
def test_LayerStack_merge_down_na_base_rejeita(indexed_stack):
    """Cenário 3: Tentativa de Merge na Base (Operação Inválida)"""
    stack, *layers = indexed_stack

    with pytest.raises(ValueError, match="Cannot merge down from the bottom layer"):
        stack.merge_down(0)

    assert len(stack) == 3


@pytest.mark.skip()
@patch("anicrop.layer_stack.Render")
@patch("anicrop.layer_stack.EditLayer")
def test_LayerStack_merge_down_indice_negativo(mock_edit_layer, mock_render_class, indexed_stack):
    """Cenário 4: Uso de Índice Negativo Válido (Topo)"""
    stack, l10, l20, l30 = indexed_stack

    merged_mock = MagicMock()
    mock_edit_layer.return_value = merged_mock
    fake_img = Image(np.zeros((5, 5, 3), dtype=np.uint8), ImageFormat.RGB)
    mock_render_class.return_value.render.return_value = fake_img

    # -1 refere-se ao topo (índice 2)
    stack.merge_down(-1)

    assert len(stack) == 2
    assert stack.get(1) is merged_mock


@pytest.mark.skip()
def test_LayerStack_merge_down_fora_dos_limites(indexed_stack):
    """Cenário 5: Índice Fora dos Limites"""
    stack, *layers = indexed_stack
    with pytest.raises(IndexError):
        stack.merge_down(99)


@pytest.mark.skip()
def test_LayerStack_merge_down_pilha_insuficiente():
    """Cenário 6: Merge em Pilha Vazia ou com 1 Camada"""
    stack_vazia = LayerStack()
    with pytest.raises((IndexError, ValueError)):
        stack_vazia.merge_down(0)

    stack_uma = LayerStack()
    stack_uma.add(DummyLayer(1))
    with pytest.raises(ValueError, match="Cannot merge down"):
        stack_uma.merge_down(0)
