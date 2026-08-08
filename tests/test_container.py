import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import Container, GroupLayer, LayerStack, _NULL_CONTAINER
from anicrop.image import Image, ImageFormat
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
def test_remover_item_inexistente_deve_lancar_value_error(mocker, container_cls, item_cls):
    group = container_cls()
    item = mocker.MagicMock(spec=item_cls)

    with pytest.raises(ValueError) as exc_info:
        group.remove(item)

    assert str(
        exc_info.value) == f"Item {item} is not in this {group.__class__.__name__}"


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_remover_item_existente_reseta_parent_len_e_matriz(mocker, container_cls, item_cls):
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
    mock_parent.matrix = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 20.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    group.parent = mock_parent

    expected_matrix = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 20.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

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
            Region(Span(-10, 110), Span(10, 25))
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
        [
            [Region(Span(-10, 50), Span(10, 20))],
            [Region(Span(60, 40), Span(15, 20))]
        ],
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

    canvas_obj = Canvas(500, 500)
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
        type(child_group.base), 'region',
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
def test_adicionar_item_ja_existente_deve_lancar_value_error(mocker, container_cls, item_cls, method_name):
    group = container_cls()
    item = mocker.MagicMock(spec=item_cls)
    item.parent = _NULL_CONTAINER

    group.append(item)

    with pytest.raises(ValueError) as exc_info:
        if method_name == "append":
            group.append(item)
        else:
            group.insert(0, item)

    assert str(
        exc_info.value) == f"Item {item} is already in this {group.__class__.__name__}"


@pytest.mark.parametrize("container_cls", [GroupLayer])
@pytest.mark.parametrize("method_name", ["append", "insert"])
def test_adicionar_ancestral_no_filho_deve_lancar_value_error(container_cls, method_name):
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

    assert str(
        exc_info.value) == "A LayerStack is a Root object and cannot be added as a child."


@pytest.mark.parametrize("container_cls", [LayerStack, GroupLayer])
@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_inserir_item_em_index_especifico_respeita_ordem(mocker, container_cls, item_cls):
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
def test_mover_item_existente_muda_ordem_sem_chamar_parent_remove(mocker, container_cls, item_cls):
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
    spy_remove = mocker.spy(container, 'remove')

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

    assert str(
        exc_info.value) == f"Item {item_invalido} is not in this {container.__class__.__name__}"


def test_group_layer_render_recursivo_com_culling(mocker):

    group_raiz = GroupLayer()
    sub_grupo = GroupLayer()

    img1 = Image.new((100, 100), ImageFormat.RGBA)
    img2 = Image.new((100, 100), ImageFormat.RGBA)
    layer1 = Layer(img1)
    layer2 = Layer(img2)

    # Configura máscaras de opacidade (layer1 cobre 100% da miniview 32x32 com opacidade 255)
    layer1._opacity_mask = np.full((32, 32), 255, dtype=np.uint8)
    layer2._opacity_mask = np.full((32, 32), 255, dtype=np.uint8)

    sub_grupo.append(layer1)
    group_raiz.append(sub_grupo)
    group_raiz.append(layer2)

    mock_img = Image.new((100, 100), ImageFormat.RGBA)
    mock_plan = mocker.MagicMock()
    mock_plan.dst_region = Region.from_size(100, 100)

    mock_renderer = mocker.MagicMock(return_value=mock_img)
    mock_plan_cls = mocker.MagicMock(return_value=mock_plan)
    mock_surface = mocker.MagicMock()
    mock_surface.size = (100, 100)
    miniview = np.zeros((32, 32), dtype=np.uint8)

    # Executa o render recursivo do grupo raiz
    result, images_gp = group_raiz.render(
        mock_renderer, mock_plan_cls, mock_surface, miniview)

    # Como o layer1 no sub_grupo cobriu 100% da miniview (255), result deve ser True
    assert result is True
    # E a execução do loop foi interrompida antes de renderizar o layer2!
    assert len(images_gp) == 1
    assert images_gp[0][0] == group_raiz
    assert mock_renderer.call_count == 1
    assert np.all(miniview == 255)


@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_group_layer_render_ignora_itens_invisiveis(mocker, item_cls):
    # 1. Cria o item via TDD estrito usando spec
    item = mocker.MagicMock(spec=item_cls)

    # 2. Espiona o acesso ao atributo 'visible' forçando um PropertyMock apenas no teste
    mock_visible = mocker.PropertyMock(return_value=False)
    type(item).visible = mock_visible
    item.parent = mocker.Mock()

    # 3. Adiciona ao grupo
    group = GroupLayer()
    group.append(item)

    # 4. Chama o render com mocks básicos
    renderer = mocker.Mock()
    plan_cls = mocker.Mock()
    surface = mocker.Mock()
    surface.size = (10, 10)
    miniview = mocker.Mock()

    result = group.render(renderer, plan_cls, surface, miniview)

    # 5. Prova que o GroupLayer DE FATO leu o atributo visible do item
    mock_visible.assert_called()

    # 6. Prova que ele abortou e retornou a tupla vazia corretamente
    assert result == (False, [])


def test_group_layer_render_ignora_tudo_se_raiz_for_invisivel(mocker):
    # 1. Torna a classe GroupLayer inteira invisível (para o root)
    root = GroupLayer()
    root.visible = False

    # 2. Cria o GroupLayer filho via Mock (TDD) e o faz visível
    child = mocker.MagicMock(spec=GroupLayer)
    type(child).visible = mocker.PropertyMock(return_value=True)
    child.parent = mocker.Mock()

    root.append(child)

    # Argumentos mockados
    renderer = mocker.Mock()
    plan_cls = mocker.Mock()
    surface = mocker.Mock()
    miniview = mocker.Mock()

    # 3. Executa o render na raiz invisível
    result = root.render(renderer, plan_cls, surface, miniview)

    # 4. Verifica que o filho não foi sequer processado/renderizado
    child.render.assert_not_called()

    # 5. O retorno final da raiz deve ser a tupla vazia
    assert result == (False, [])
