import numpy as np
import pytest
from anicrop.command import BaseLayerCommand, LayerImageCommand, ReparentCommand
from anicrop.container import GroupLayer, LayerStack, _NULL_CONTAINER
from anicrop.enums import BlendMode
from anicrop.history import GlobalHistory
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.proxy import GroupProxy, LayerStackProxy, ProxyLayer
from anicrop.transform import Transform


def make_img(w=10, h=10):
    return Image(np.zeros((h, w, 4), dtype=np.uint8), ImageFormat.RGBA)


# ============================================================================
# Testes de Cobertura do ReparentCommand
# ============================================================================

@pytest.mark.parametrize("cls, proxy_cls", [(GroupLayer, GroupProxy), (LayerStack, LayerStackProxy)])
def test_reparent_between_different_groups(cls, proxy_cls):
    """1. Mover um Layer entre Grupos Diferentes (Grupo A -> Grupo B)"""
    history = GlobalHistory()
    group_a = proxy_cls(cls(), history)
    group_b = proxy_cls(cls(), history)
    layer = ProxyLayer(Layer(make_img(), name="Layer 1"), history)

    group_a.append(layer)

    cmd = ReparentCommand("append", group_b, layer)
    group_b.append(layer)
    cmd.seal()

    # Estado modificado
    assert layer in group_b
    assert layer not in group_a
    assert layer.parent is group_b

    # UNDO
    cmd.undo()
    assert layer in group_a
    assert layer not in group_b
    assert layer.parent is group_a
    assert group_a[0] == layer

    # REDO
    cmd.execute()
    assert layer in group_b
    assert layer not in group_a
    assert layer.parent is group_b
    assert group_b[0] == layer


@pytest.mark.parametrize("cls, proxy_cls", [(GroupLayer, GroupProxy), (LayerStack, LayerStackProxy)])
def test_reparent_reorder_same_container(cls, proxy_cls):
    """2. Reordenação no Mesmo Container (move)"""
    history = GlobalHistory()
    group = proxy_cls(cls(), history)
    l1 = ProxyLayer(Layer(make_img(), name="L1"), history)
    l2 = ProxyLayer(Layer(make_img(), name="L2"), history)
    l3 = ProxyLayer(Layer(make_img(), name="L3"), history)

    group.append(l1)
    group.append(l2)
    group.append(l3)

    # Move L1 do índice 0 para o índice 2
    cmd = ReparentCommand("move", group, l1)
    group.move(l1, 2)
    cmd.seal()

    assert list(group._target) == [l2, l3, l1]

    # UNDO restaura ordem original [L1, L2, L3]
    cmd.undo()
    assert list(group._target) == [l1, l2, l3]

    # REDO aplica [L2, L3, L1]
    cmd.execute()
    assert list(group._target) == [l2, l3, l1]


@pytest.mark.parametrize("cls, proxy_cls", [(GroupLayer, GroupProxy), (LayerStack, LayerStackProxy)])
def test_reparent_remove_and_restore(cls, proxy_cls):
    """3. Remoção de um Elemento (remove / pop)"""
    history = GlobalHistory()
    group = proxy_cls(cls(), history)
    layer = ProxyLayer(Layer(make_img(), name="Layer 1"), history)

    group.append(layer)

    cmd = ReparentCommand("remove", group, layer)
    group.remove(layer)
    cmd.seal()

    assert layer not in group
    assert layer.parent is _NULL_CONTAINER

    # UNDO devolve o layer para o grupo
    cmd.undo()
    assert layer in group
    assert layer.parent is group
    assert group[0] == layer

    # REDO remove novamente
    cmd.execute()
    assert layer not in group
    assert layer.parent is _NULL_CONTAINER


def test_reparent_nested_group_in_group():
    """4. Movimentação Aninhada de Grupos (GroupLayer dentro de GroupLayer)"""
    history = GlobalHistory()
    root_group = GroupProxy(GroupLayer(name="Root Group"), history)
    sub_group = GroupProxy(GroupLayer(name="Sub Group"), history)
    target_group = GroupProxy(GroupLayer(name="Target Group"), history)

    root_group.append(sub_group)

    cmd = ReparentCommand("append", target_group, sub_group)
    target_group.append(sub_group)
    cmd.seal()

    assert sub_group in target_group
    assert sub_group not in root_group
    assert sub_group.parent is target_group

    # UNDO traz sub_group de volta para root_group
    cmd.undo()
    assert sub_group in root_group
    assert sub_group not in target_group
    assert sub_group.parent is root_group

    # REDO
    cmd.execute()
    assert sub_group in target_group


def test_reparent_restores_parent_inverse_matrix():
    """5. Restauração da Matriz Espacial Inversa (_parent_inverse)"""
    history = GlobalHistory()
    group_a = GroupProxy(GroupLayer(name="Grupo A"), history)
    group_b = GroupProxy(GroupLayer(name="Grupo B"), history)
    layer = ProxyLayer(Layer(make_img(), name="Layer 1"), history)

    # Aplica transformações espaciais diferentes nos dois grupos
    group_a.set_transform(Transform.relative().translate(100, 100))
    group_b.set_transform(Transform.relative().translate(500, 500))

    group_a.append(layer)
    inverse_a_initial = np.copy(layer._parent_inverse)

    cmd = ReparentCommand("append", group_b, layer)
    group_b.append(layer)
    cmd.seal()

    inverse_b_after = np.copy(layer._parent_inverse)
    assert not np.array_equal(inverse_a_initial, inverse_b_after)

    # UNDO restaura a matriz de inversão espacial do grupo A
    cmd.undo()
    assert np.array_equal(layer._parent_inverse, inverse_a_initial)

    # REDO restaura a matriz do grupo B
    cmd.execute()
    assert np.array_equal(layer._parent_inverse, inverse_b_after)


@pytest.mark.parametrize("cls, proxy_cls", [(GroupLayer, GroupProxy), (LayerStack, LayerStackProxy)])
def test_reparent_has_changes(cls, proxy_cls):
    """6. Verificação do método has_changes()"""
    history = GlobalHistory()
    group = proxy_cls(cls(), history)
    layer = ProxyLayer(Layer(make_img(), name="Layer"), history)
    group.append(layer)

    # Tenta mover sem alterar posição real
    cmd = ReparentCommand("move", group, layer)
    cmd.seal()

    # Como parent e index não mudaram, não há alterações
    assert cmd.has_changes() is False


def test_reparent_layerstack_as_child_raises_error():
    """7. TDD: O comando não deve aceitar um LayerStack como alvo da operação (value)"""
    history = GlobalHistory()
    root = LayerStackProxy(LayerStack(), history)
    grupo = GroupProxy(GroupLayer(name="Grupo Destino"), history)

    expected_msg = "A LayerStack is a Root object and cannot be added as a child."

    # O comando DEVE estourar o erro logo na sua instanciação para proteger o sistema!
    with pytest.raises(ValueError, match=expected_msg):
        ReparentCommand("append", grupo, root)


# ============================================================================
# Testes de Cobertura do BaseLayerCommand
# ============================================================================


def test_baselayer_command_opacity_and_visibility():
    """1. Testa alteração de opacidade e visibilidade no BaseLayerCommand"""
    history = GlobalHistory()
    layer = ProxyLayer(Layer(make_img(), name="Layer 1"), history)
    layer.opacity = 1.0
    layer.visible = True

    cmd = BaseLayerCommand("state_change", layer)
    layer.opacity = 0.5
    layer.visible = False
    cmd.seal()

    assert layer.opacity == 0.5
    assert layer.visible is False
    assert cmd.has_changes() is True

    cmd.undo()
    assert layer.opacity == 1.0
    assert layer.visible is True

    cmd.execute()
    assert layer.opacity == 0.5
    assert layer.visible is False


def test_baselayer_command_blend_mode():
    """2. Testa alteração de BlendMode"""
    history = GlobalHistory()
    layer = ProxyLayer(Layer(make_img(), name="Layer 1"), history)
    layer.blend_mode = BlendMode.NORMAL

    cmd = BaseLayerCommand("state_change", layer)
    layer.blend_mode = BlendMode.MULTIPLY
    cmd.seal()

    assert layer.blend_mode == BlendMode.MULTIPLY

    cmd.undo()
    assert layer.blend_mode == BlendMode.NORMAL


def test_baselayer_command_transform():
    """3. Testa captura e restauração de matrizes no transform"""
    history = GlobalHistory()
    layer = ProxyLayer(Layer(make_img(), name="Layer 1"), history)
    matrix_before = np.copy(layer.transform.matrix)

    cmd = BaseLayerCommand("state_change", layer)
    layer.transform.translate(50, 50)
    matrix_after = np.copy(layer.transform.matrix)
    cmd.seal()

    assert not np.array_equal(matrix_before, matrix_after)
    assert np.array_equal(layer.transform.matrix, matrix_after)

    cmd.undo()
    assert np.array_equal(layer.transform.matrix, matrix_before)

    cmd.execute()
    assert np.array_equal(layer.transform.matrix, matrix_after)


def test_baselayer_command_no_changes():
    """4. Testa has_changes() quando não houver mutação"""
    history = GlobalHistory()
    layer = ProxyLayer(Layer(make_img(), name="Layer 1"), history)
    layer.opacity = 1.0

    cmd = BaseLayerCommand("state_change", layer)
    layer.opacity = 1.0  # Mesma coisa
    cmd.seal()

    assert cmd.has_changes() is False


# ============================================================================
# Testes de Cobertura do LayerImageCommand
# ============================================================================


def test_layerimage_command_add_edit():
    """1. Testa alteração na fila de edições (Layer._edits)"""
    history = GlobalHistory()
    img1 = make_img()
    layer = ProxyLayer(Layer(img1, name="Layer 1"), history)
    # A camada nasce com 1 edit (a própria imagem base)
    assert len(layer._edits) == 1

    cmd = LayerImageCommand("add_edit", layer)

    img2 = make_img()
    layer.add_edit(img2, layer.region)
    assert len(layer._edits) == 2
    cmd.seal()

    assert cmd.has_changes() is True

    cmd.undo()
    assert len(layer._edits) == 1

    cmd.execute()
    assert len(layer._edits) == 2


def test_layerimage_command_opacity_mask():
    """2. Testa aplicação de opacity_mask (Numpy array)"""
    history = GlobalHistory()
    layer = ProxyLayer(Layer(make_img(), name="Layer"), history)
    assert layer._opacity_mask is None

    cmd = LayerImageCommand("mask", layer)
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    layer._opacity_mask = mask
    cmd.seal()

    assert cmd.has_changes() is True
    assert np.array_equal(layer._opacity_mask, mask)

    cmd.undo()
    assert layer._opacity_mask is None

    cmd.execute()
    assert np.array_equal(layer._opacity_mask, mask)


def test_layerimage_command_no_changes():
    """3. Testa has_changes() retornando False para ausência de mutação"""
    history = GlobalHistory()
    layer = ProxyLayer(Layer(make_img(), name="Layer"), history)

    cmd = LayerImageCommand("mask", layer)
    # Nenhuma mutação real
    cmd.seal()

    assert cmd.has_changes() is False
