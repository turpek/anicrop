from unittest.mock import MagicMock
import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import (
    Container,
    GroupLayer,
    GroupLayoutStrategy,
    LayerStack,
    _NULL_CONTAINER,
    walk_nodes,
)
from anicrop.image import Image, ImageFormat
from anicrop.layout import Layout
from anicrop.layer import Layer
from anicrop.spatial import Region, Span


from anicrop.transform import (
    TransformAbs,
    TransformRel,
)


@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
@pytest.mark.parametrize("parent", [Container, GroupLayer])
@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_adicionar_item_em_group_layer(mocker, count, container_cls, item_cls, parent):
    group = container_cls()

    for _ in range(count):
        item = mocker.MagicMock(spec=item_cls)
        old_parent = mocker.MagicMock(spec=parent)
        item.parent = old_parent

        group.append(item)

        old_parent.remove.assert_called_once_with(item)
        assert item.parent == group

    assert len(group) == count


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_remover_item_inexistente_deve_lancar_value_error(
    mocker, container_cls, item_cls
):
    group = container_cls()
    item = mocker.MagicMock(spec=item_cls)

    with pytest.raises(ValueError) as exc_info:
        group.remove(item)

    assert (
        str(exc_info.value) == f"Item {item} is not in this {group.__class__.__name__}"
    )


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_remover_item_existente_reseta_parent_len_e_matriz(
    mocker, container_cls, item_cls
):
    parent_group = container_cls()
    item = mocker.MagicMock(spec=item_cls)
    item.parent = _NULL_CONTAINER

    # 1. Anexa o item
    parent_group.append(item)
    assert len(parent_group) == 1
    assert item in parent_group._children
    assert item.parent == parent_group

    # 2. Remove o item
    parent_group.remove(item)

    # 3. Verifica se os estados foram resetados corretamente
    assert len(parent_group) == 0
    assert item not in parent_group._children
    assert item.parent == _NULL_CONTAINER


def test_remover_subgrupo_reseta_matriz_global_do_filho():
    parent_group = GroupLayer()
    parent_group.transform.rotate(90, 0.5, 0.5)

    child_group = GroupLayer()
    # child_group.region = Region.from_size(50, 50)

    # Anexa o filho no pai rotacionado
    parent_group.append(child_group)
    assert len(parent_group) == 1

    # Remove o filho
    parent_group.remove(child_group)

    # O filho deve ser removido, seu len do pai zera e a matriz do filho volta a ser a identidade
    assert len(parent_group) == 0
    assert child_group not in parent_group._children
    assert child_group.parent == _NULL_CONTAINER

    expected_identity = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(child_group.matrix, expected_identity, atol=1e-4)


def test_group_layer_matrix_multiplica_matriz_do_parent(mocker):

    group = GroupLayer()

    mock_parent = mocker.MagicMock(spec=Container)
    mock_parent.matrix = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]], dtype=float
    )

    group.parent = mock_parent

    expected_matrix = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]], dtype=float
    )

    assert np.allclose(group.matrix, expected_matrix)


def test_group_layer_matrix_hierarquia_real():

    parent_group = GroupLayer()
    child_group = GroupLayer()

    parent_group.append(child_group)

    expected_matrix = np.eye(3, dtype=float)

    assert np.allclose(child_group.matrix, expected_matrix)


@pytest.mark.parametrize(
    "regions, expected_region",
    [
        [[], Region.from_size(1, 1)],
        [[Region(Span(10, 100), Span(10, 100))], Region(Span(10, 100), Span(10, 100))],
        [
            [Region(Span(-10, 50), Span(10, 20)), Region(Span(60, 40), Span(15, 20))],
            Region(Span(-10, 110), Span(10, 25)),
        ],
    ],
    ids=["vazio", "1_layer", "2_layers"],
)
def test_group_layer_region_so_com_layers(mocker, regions, expected_region):

    group = GroupLayer()

    for region in regions:
        layer = mocker.MagicMock(spec=Layer)
        layer.region = region
        layer.parent = _NULL_CONTAINER
        group.append(layer)

    assert group.region == expected_region


@pytest.mark.parametrize(
    "direct_regions, subgroup_regions",
    [
        [[], [Region(Span(10, 100), Span(10, 100))]],
        [[Region(Span(-10, 50), Span(10, 20))], [Region(Span(60, 40), Span(15, 20))]],
    ],
    ids=["1_group", "1_layer_e_1_group"],
)
def test_group_layer_region_com_subgrupo(mocker, direct_regions, subgroup_regions):
    group = GroupLayer()

    for region in direct_regions:
        layer = mocker.MagicMock(spec=Layer)
        layer.region = region
        layer.parent = _NULL_CONTAINER
        group.append(layer)

    if subgroup_regions:
        subgroup = GroupLayer()
        for region in subgroup_regions:
            sublayer = mocker.MagicMock(spec=Layer)
            sublayer.region = region
            sublayer.parent = _NULL_CONTAINER
            subgroup.append(sublayer)
        group.append(subgroup)

    all_regions = direct_regions + subgroup_regions
    expected_region = all_regions[0]
    for reg in all_regions[1:]:
        expected_region |= reg

    assert group.region == expected_region


def test_group_layer_set_transform_substitui_transformacao():
    group = GroupLayer()

    t1 = TransformRel().translate(10, 20)
    group.set_transform(t1)

    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(group.matrix @ pt_origem, [10, 20, 1], atol=1e-4)

    t2 = TransformRel().translate(40, 50)
    group.set_transform(t2)

    np.testing.assert_allclose(group.matrix @ pt_origem, [40, 50, 1], atol=1e-4)


def test_group_layer_transform_clear():
    group = GroupLayer()

    t = TransformRel().translate(10, 20)
    group.set_transform(t)

    group.transform_clear()

    expected_identity = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(group.matrix, expected_identity, atol=1e-4)


def test_group_layer_set_transform_com_transform_abs():
    group = GroupLayer()

    t_abs = TransformAbs().rotate(90, px=50, py=50)
    group.set_transform(t_abs)

    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(group.matrix @ pt_origem, [100, 0, 1], atol=1e-4)


def test_group_layer_set_transform_com_referencia_layer_e_canvas(mocker):
    group = GroupLayer()

    t_rel = TransformRel().rotate(90, 0.5, 0.5)

    ref_layer = mocker.MagicMock(spec=Layer)
    type(ref_layer).region = mocker.PropertyMock(
        return_value=Region(Span(0, 1000), Span(0, 1000))
    )
    group.set_transform(t_rel, reference=ref_layer)

    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(group.matrix @ pt_origem, [1000, 0, 1], atol=1e-4)

    canvas_obj = Canvas.from_size(500, 500)
    group.set_transform(t_rel, reference=canvas_obj)
    np.testing.assert_allclose(group.matrix @ pt_origem, [500, 0, 1], atol=1e-4)


def test_group_layer_transform_dinamica_lazy_recalcula_pivo_apos_expansao(mocker):
    group = GroupLayer()

    mock_layer = mocker.MagicMock(spec=Layer)
    mock_layer.parent = _NULL_CONTAINER
    mock_layer._region = Region(Span(0, 100), Span(0, 100))
    type(mock_layer).region = property(lambda self: self._region)

    group.append(mock_layer)

    t_rel = TransformRel().rotate(90, 0.5, 0.5)
    group.set_transform(t_rel)

    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(group.matrix @ pt_origem, [100, 0, 1], atol=1e-4)


def test_group_layer_hierarquia_acumula_matrizes_pai_e_filho(mocker):
    parent_group = GroupLayer()

    child_group = GroupLayer()
    mocker.patch.object(
        type(child_group.base),
        "region",
        new_callable=mocker.PropertyMock,
        return_value=Region.from_size(50, 50),
    )
    parent_group.append(child_group)

    # 1. Aplica rotação no filho (pivô em 25, 25) e verifica a matriz do filho
    t_child = TransformRel().rotate(90, 0.5, 0.5)
    child_group.set_transform(t_child)

    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(child_group.matrix @ pt_origem, [50, 0, 1], atol=1e-4)

    # 2. Aplica rotação no pai (pivô em 25, 25)
    t_parent = TransformRel().rotate(90, 0.5, 0.5)
    parent_group.set_transform(t_parent)

    np.testing.assert_allclose(child_group.matrix @ pt_origem, [50, 50, 1], atol=1e-4)


@pytest.mark.parametrize("container_cls", [GroupLayer])
@pytest.mark.parametrize("method_name", ["append", "insert"])
def test_adicionar_grupo_a_si_mesmo_deve_lancar_value_error(container_cls, method_name):
    group = container_cls()

    with pytest.raises(ValueError) as exc_info:
        if method_name == "append":
            group.append(group)
        else:
            group.insert(0, group)

    assert str(exc_info.value) == f"Cannot add a {group.__class__.__name__} to itself"


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
@pytest.mark.parametrize("method_name", ["append", "insert"])
def test_adicionar_item_ja_existente_deve_lancar_value_error(
    mocker, container_cls, item_cls, method_name
):
    group = container_cls()
    item = mocker.MagicMock(spec=item_cls)
    item.parent = _NULL_CONTAINER

    group.append(item)

    with pytest.raises(ValueError) as exc_info:
        if method_name == "append":
            group.append(item)
        else:
            group.insert(0, item)

    assert (
        str(exc_info.value)
        == f"Item {item} is already in this {group.__class__.__name__}"
    )


@pytest.mark.parametrize("container_cls", [GroupLayer])
@pytest.mark.parametrize("method_name", ["append", "insert"])
def test_adicionar_ancestral_no_filho_deve_lancar_value_error(
    container_cls, method_name
):
    grandparent_group = container_cls()
    parent_group = container_cls()
    child_group = container_cls()

    grandparent_group.append(parent_group)
    parent_group.append(child_group)

    # 1. Tenta adicionar o pai direto no filho
    with pytest.raises(ValueError) as exc_info:
        if method_name == "append":
            child_group.append(parent_group)
        else:
            child_group.insert(0, parent_group)
    assert str(exc_info.value) == "Cannot add an ancestor container to a child container"

    # 2. Tenta adicionar o avô (ancestral de múltiplos níveis) no neto
    with pytest.raises(ValueError) as exc_info:
        if method_name == "append":
            child_group.append(grandparent_group)
        else:
            child_group.insert(0, grandparent_group)
    assert str(exc_info.value) == "Cannot add an ancestor container to a child container"


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("method_name", ["append", "insert"])
def test_adicionar_layer_stack_como_item_lanca_type_error(container_cls, method_name):
    container = container_cls()
    stack = LayerStack()

    with pytest.raises(TypeError) as exc_info:
        if method_name == "append":
            container.append(stack)
        else:
            container.insert(0, stack)

    assert (
        str(exc_info.value)
        == "A LayerStack is a Root object and cannot be added as a child."
    )


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_inserir_item_em_index_especifico_respeita_ordem(
    mocker, container_cls, item_cls
):
    container = container_cls()

    item1 = mocker.MagicMock(spec=item_cls)
    item2 = mocker.MagicMock(spec=item_cls)
    item_novo = mocker.MagicMock(spec=item_cls)

    item1.parent = _NULL_CONTAINER
    item2.parent = _NULL_CONTAINER

    # Adiciona num parent antigo para testar a desvinculação automática
    old_parent = mocker.MagicMock(spec=GroupLayer)
    item_novo.parent = old_parent

    container.append(item1)
    container.append(item2)

    # O container tem [item1, item2]
    # Vamos inserir o item_novo no index 1, ficando [item1, item_novo, item2]
    container.insert(1, item_novo)

    # Asserts
    assert len(container) == 3
    assert container._children == [item1, item_novo, item2]
    assert item_novo.parent == container
    old_parent.remove.assert_called_once_with(item_novo)


def test_group_layer_iter_retorna_children():
    group = GroupLayer()
    item1 = GroupLayer()
    item2 = GroupLayer()
    group.append(item1)
    group.append(item2)

    assert list(group) == [item1, item2]


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_mover_item_existente_muda_ordem_sem_chamar_parent_remove(
    mocker, container_cls, item_cls
):
    container = container_cls()
    item1 = mocker.MagicMock(spec=item_cls)
    item2 = mocker.MagicMock(spec=item_cls)
    item3 = mocker.MagicMock(spec=item_cls)

    item1.parent = _NULL_CONTAINER
    item2.parent = _NULL_CONTAINER
    item3.parent = _NULL_CONTAINER

    container.append(item1)
    container.append(item2)
    container.append(item3)

    # Usar spy para certificar de que não tentamos desvincular via remove()
    spy_remove = mocker.spy(container, "remove")

    # move item3 do index 2 para o index 0
    container.move(item3, 0)

    assert container._children == [item3, item1, item2]
    # Certificar de que não tentamos desvincular chamando remove
    spy_remove.assert_not_called()


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_mover_item_inexistente_deve_lancar_value_error(mocker, container_cls, item_cls):
    container = container_cls()
    item_invalido = mocker.MagicMock(spec=item_cls)
    item_invalido.parent = _NULL_CONTAINER

    with pytest.raises(ValueError) as exc_info:
        container.move(item_invalido, 0)

    assert (
        str(exc_info.value)
        == f"Item {item_invalido} is not in this {container.__class__.__name__}"
    )


def make_mock_image(size: tuple[int, int] = (100, 100)) -> Image:
    """Cria uma Image real cujo buffer interno (_data) é um MagicMock."""
    w, h = size
    mock_data = MagicMock(spec=np.ndarray)
    mock_data.ndim = 3
    mock_data.shape = (h, w, 4)
    mock_data.dtype = np.uint8
    return Image(mock_data, ImageFormat.RGBA)


def test_group_layer_fit_preserva_pivo_com_camada_filha_deslocada():
    """
    Valida se o cálculo do pivô relativo (0.5, 0.5) do Composer no GroupLayer
    utiliza a moldura ativa do Layout (layout.region) em vez da base.region dos filhos.
    Se a base.region (40x40 em 50, 50) fosse usada, a rotação de 90° de um grupo
    ajustado para (0, 0, 100, 100) calcularia o pivô em (70, 70), deslocando a global_region incorretamente.
    """
    group = GroupLayer()
    mock_child_img = make_mock_image(size=(40, 40))
    child = Layer(mock_child_img)
    child.region += (50, 50)
    group.append(child)

    layout = Layout()
    layout.fit(group, Region.from_rect(0, 0, 100, 100))
    assert group.global_region == Region.from_rect(0, 0, 100, 100)

    # Rotação de 90° no centro (0.5, 0.5) da moldura de 100x100
    group.transform.rotate(90)

    # A global_region deve permanecer perfeitamente em (0, 0, 100, 100)
    assert group.global_region == Region.from_rect(0, 0, 100, 100)


def test_walk_nodes_com_camadas_e_grupos_aninhados():
    """Valida se walk_nodes realiza a travessia DFS completa de uma árvore contendo grupos e camadas."""
    root = GroupLayer(name="root_group")
    mock_img1 = make_mock_image(size=(10, 10))
    child_layer1 = Layer(mock_img1, name="layer1")

    child_group = GroupLayer(name="sub_group")
    mock_img2 = make_mock_image(size=(10, 10))
    sub_child_layer = Layer(mock_img2, name="sub_layer")

    child_group.append(sub_child_layer)
    root.append(child_layer1)
    root.append(child_group)

    result = list(walk_nodes(root))

    assert result == [root, child_layer1, child_group, sub_child_layer]


def test_group_layer_layout_bound_api():
    """Valida as operações de layout diretamente via group.layout."""
    group = GroupLayer(name="test_group")
    mock_img = make_mock_image(size=(100, 100))
    child = Layer(mock_img)
    group.append(child)

    assert isinstance(group.layout, GroupLayoutStrategy)

    assert group.layout.fit(Region.from_rect(10, 10, 200, 200)) is True
    assert group.global_region == Region.from_rect(10, 10, 200, 200)

    target_ref = Region.from_rect(0, 0, 1000, 1000)
    assert group.layout.align(target_ref, 0.5, 0.5) is True
    assert group.global_region == Region.from_rect(400, 400, 200, 200)

    assert group.layout.resize_bounds(300, 300, 0.5, 0.5) is True
    assert group.global_region.size == (300, 300)


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize(
    "initial_index,steps,expected_index",
    [
        (1, 1, 2),
        (1, 2, 3),
        (2, -1, 1),
        (2, -2, 0),
        (1, 0, 1),
        (2, 10, 3),
        (1, -10, 0),
    ],
    ids=["up_1", "up_2", "down_1", "down_2", "zero", "clamp_top", "clamp_bottom"],
)
def test_container_move_relative(container_cls, initial_index, steps, expected_index):
    """Valida movimentacao relativa de itens no container com clamping nos limites."""
    container = container_cls()
    layers = [Layer(make_mock_image(), name=f"L{i}") for i in range(4)]
    for l in layers:
        container.append(l)

    target = layers[initial_index]
    container.move_relative(target, steps)

    assert container._children.index(target) == expected_index


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_move_relative_item_inexistente_lanca_erro(container_cls):
    """Valida que move_relative com item fora do container lanca ValueError."""
    container = container_cls()
    l1 = Layer(make_mock_image(), name="L1")
    l_fora = Layer(make_mock_image(), name="LFora")
    container.append(l1)

    with pytest.raises(ValueError):
        container.move_relative(l_fora, 1)


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("start_idx", [0, 1, 2])
def test_container_move_to_front(container_cls, start_idx):
    """Valida envio de item para o topo absoluto do container."""
    container = container_cls()
    layers = [Layer(make_mock_image(), name=f"L{i}") for i in range(3)]
    for l in layers:
        container.append(l)

    target = layers[start_idx]
    container.move_to_front(target)

    assert container._children[-1] is target
    assert len(container) == 3


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_move_to_front_item_inexistente_lanca_erro(container_cls):
    """Valida que move_to_front com item fora do container lanca ValueError."""
    container = container_cls()
    l_fora = Layer(make_mock_image(), name="LFora")

    with pytest.raises(ValueError):
        container.move_to_front(l_fora)


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("start_idx", [0, 1, 2])
def test_container_move_to_back(container_cls, start_idx):
    """Valida envio de item para a base absoluta do container."""
    container = container_cls()
    layers = [Layer(make_mock_image(), name=f"L{i}") for i in range(3)]
    for l in layers:
        container.append(l)

    target = layers[start_idx]
    container.move_to_back(target)

    assert container._children[0] is target
    assert len(container) == 3


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_move_to_back_item_inexistente_lanca_erro(container_cls):
    """Valida que move_to_back com item fora do container lanca ValueError."""
    container = container_cls()
    l_fora = Layer(make_mock_image(), name="LFora")

    with pytest.raises(ValueError):
        container.move_to_back(l_fora)


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_swap_itens(container_cls):
    """Valida troca de posicao entre dois itens do container."""
    container = container_cls()
    l0 = Layer(make_mock_image(), name="L0")
    l1 = Layer(make_mock_image(), name="L1")
    l2 = Layer(make_mock_image(), name="L2")
    for l in (l0, l1, l2):
        container.append(l)

    container.swap(l0, l2)

    assert container._children == [l2, l1, l0]

    container.swap(l1, l1)
    assert container._children == [l2, l1, l0]


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_swap_item_inexistente_lanca_erro(container_cls):
    """Valida que swap com item fora do container lanca ValueError."""
    container = container_cls()
    l0 = Layer(make_mock_image(), name="L0")
    l_fora = Layer(make_mock_image(), name="LFora")
    container.append(l0)

    with pytest.raises(ValueError):
        container.swap(l0, l_fora)

    with pytest.raises(ValueError):
        container.swap(l_fora, l0)


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_reverse_shallow(container_cls):
    """Valida inversao de ordem de filhos no nivel atual com recursive=False."""
    container = container_cls()
    l0 = Layer(make_mock_image(), name="L0")
    l1 = Layer(make_mock_image(), name="L1")
    l2 = Layer(make_mock_image(), name="L2")
    for l in (l0, l1, l2):
        container.append(l)

    container.reverse()

    assert container._children == [l2, l1, l0]


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_reverse_shallow_preserves_subgroup_order(container_cls):
    """Valida que reverse com recursive=False nao altera a ordem interna dos subgrupos."""
    container = container_cls()
    l0 = Layer(make_mock_image(), name="L0")
    sub0 = Layer(make_mock_image(), name="Sub0")
    sub1 = Layer(make_mock_image(), name="Sub1")
    group = GroupLayer(name="Group")
    group.append(sub0)
    group.append(sub1)
    l2 = Layer(make_mock_image(), name="L2")

    container.append(l0)
    container.append(group)
    container.append(l2)

    container.reverse(recursive=False)

    assert container._children == [l2, group, l0]
    assert group._children == [sub0, sub1]


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_reverse_recursive(container_cls):
    """Valida inversao profunda com recursive=True descendo para subgrupos."""
    container = container_cls()
    l0 = Layer(make_mock_image(), name="L0")
    sub0 = Layer(make_mock_image(), name="Sub0")
    sub1 = Layer(make_mock_image(), name="Sub1")
    group = GroupLayer(name="Group")
    group.append(sub0)
    group.append(sub1)
    l2 = Layer(make_mock_image(), name="L2")

    container.append(l0)
    container.append(group)
    container.append(l2)

    container.reverse(recursive=True)

    assert container._children == [l2, group, l0]
    assert group._children == [sub1, sub0]


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
def test_container_reverse_empty_and_single(container_cls):
    """Valida que reverse em container vazio ou unitario e seguro."""
    container = container_cls()
    container.reverse(recursive=True)
    assert len(container) == 0

    l0 = Layer(make_mock_image(), name="L0")
    container.append(l0)
    container.reverse(recursive=True)
    assert container._children == [l0]
