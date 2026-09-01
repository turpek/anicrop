from __future__ import annotations

from functools import reduce
from operator import or_

import numpy as np
import pytest

from anicrop.composition import (
    LayerComposition,
    clone_group,
    clone_layer,
    clone_node,
    flatten,
    merge,
)
from anicrop.container import GroupLayer
from anicrop.document import Document
from anicrop.effect import BoundEffect
from anicrop.enums import BlendMode, ImageFormat
from anicrop.filter import BlurFilter
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import CanvasRender
from anicrop.spatial import Region


def make_layer(
    color: tuple[int, ...], size: tuple[int, int] = (100, 100), name: str = "Layer"
) -> Layer:
    """Cria uma camada de teste com cor solida e tamanho especificado."""
    img = Image.new(size, ImageFormat.RGBA, color=color)
    return Layer(img, name=name)


def test_clone_layer_basic_properties_and_id_uniqueness():
    """Valida clonagem de atributos basicos e unicidade do Id da camada."""
    layer = make_layer((255, 0, 0, 255), (80, 60), name="Original")
    layer.opacity = 0.65
    layer.blend_mode = BlendMode.MULTIPLY
    layer.visible = False

    cloned = clone_layer(layer)

    assert cloned is not layer
    assert cloned.name == "Original"
    assert cloned.opacity == 0.65
    assert cloned.blend_mode == BlendMode.MULTIPLY
    assert cloned.visible is False
    assert cloned.region == layer.region
    assert cloned.format == layer.format
    assert cloned._id != layer._id
    assert len(cloned.edits) == len(layer.edits)
    assert cloned.edits[0].image is layer.edits[0].image


def test_clone_layer_transform_mutation_isolation():
    """Valida que mutacoes nas transformacoes do clone nao afetam a camada original."""
    layer = make_layer((0, 255, 0, 255), (100, 100), name="TransformLayer")
    layer.transform.rotate(30).translate(25, 40)
    orig_matrix = layer.matrix.copy()

    cloned = clone_layer(layer)
    assert np.allclose(cloned.matrix, orig_matrix)

    cloned.transform.rotate(45).translate(100, 200)

    assert np.allclose(layer.matrix, orig_matrix)
    assert not np.allclose(cloned.matrix, orig_matrix)


def test_clone_layer_mask_mutation_isolation():
    """Valida que mutacoes in-place no buffer da mascara do clone nao afetam a mascara original."""
    layer = make_layer((0, 0, 255, 255), (50, 50), name="MaskLayer")
    mask_data = np.full((50, 50), 255, dtype=np.uint8)
    mask_img = Image(mask_data, ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(50, 50), invert=True, name="OriginalMask")

    cloned = clone_layer(layer)

    assert cloned.mask is not None
    assert cloned.mask.invert is True
    assert cloned.mask.name == "OriginalMask"
    assert cloned.mask.image is not layer.mask.image

    cloned.mask[0:10, 0:10] = 0

    assert np.all(layer.mask[0:10, 0:10] == 255)
    assert np.all(cloned.mask[0:10, 0:10] == 0)


def test_clone_layer_edits_collection_isolation():
    """Valida que adicionar ou remover edits no clone nao afeta a camada original."""
    layer = make_layer((255, 255, 0, 255), (100, 100), name="EditsLayer")
    cloned = clone_layer(layer)

    patch_img = Image.new((30, 30), ImageFormat.RGBA, color=(255, 0, 255, 255))
    cloned.add_edit(patch_img, Region.from_rect(10, 10, 30, 30))

    assert len(layer.edits) == 1
    assert len(cloned.edits) == 2


def test_clone_layer_effects_preservation():
    """Valida que BoundEffect e filtros sao clonados com novas matrizes e mascaras."""
    layer = make_layer((200, 100, 50, 255), (80, 80), name="EffectLayer")
    flt = BlurFilter(5.0)
    layer.bind_effect(flt)

    cloned = clone_layer(layer)

    assert len(cloned.effects) == 1
    assert isinstance(cloned.effects[0], BoundEffect)
    assert cloned.effects[0] is not layer.effects[0]
    assert np.allclose(cloned.effects[0].matrix, layer.effects[0].matrix)


def test_clone_layer_layout_frame_preservation():
    """Valida que enquadramentos de layout ativos (FitGeometry) sao preservados no clone."""
    layer = make_layer((100, 100, 100, 255), (50, 50), name="LayoutLayer")
    layer.layout.fit(Region.from_rect(0, 0, 150, 150))

    cloned = clone_layer(layer)

    assert cloned.global_region == Region.from_rect(0, 0, 150, 150)
    assert cloned.base.region == Region.from_size(50, 50)


def test_clone_layer_rendering_equivalence():
    """Valida que a renderizacao da camada clonada produz resultado identico a original."""
    layer = make_layer((128, 64, 32, 255), (60, 60), name="RenderLayer")
    layer.transform.rotate(15).translate(10, 10)

    cloned = clone_layer(layer)

    render = CanvasRender()
    rendered_orig = render.render_layer(layer)
    rendered_clone = render.render_layer(cloned)

    assert rendered_orig is not None
    assert rendered_clone is not None
    assert np.array_equal(rendered_orig[...], rendered_clone[...])


def test_clone_group_with_nested_hierarchy():
    """Valida clonagem profunda de GroupLayer com sub-grupos e camadas filhas."""
    group = GroupLayer(name="RootGroup")
    group.transform.translate(50, 50)

    l1 = make_layer((255, 0, 0, 255), (40, 40), name="Child1")
    l1.transform.rotate(45)
    group.append(l1)

    sub_group = GroupLayer(name="SubGroup")
    l2 = make_layer((0, 255, 0, 255), (30, 30), name="Child2")
    sub_group.append(l2)
    group.append(sub_group)

    cloned_group = clone_group(group)

    assert cloned_group is not group
    assert cloned_group.name == "RootGroup"
    assert len(cloned_group) == 2
    assert cloned_group[0] is not l1
    assert cloned_group[0].name == "Child1"
    assert np.allclose(cloned_group[0].matrix, l1.matrix)

    assert cloned_group[1] is not sub_group
    assert cloned_group[1][0] is not l2
    assert cloned_group[1][0].name == "Child2"


def test_clone_node_polymorphic_dispatch():
    """Valida clonagem polimorfica via clone_node."""
    layer = make_layer((255, 0, 0, 255), (20, 20))
    group = GroupLayer()

    assert isinstance(clone_node(layer), Layer)
    assert isinstance(clone_node(group), GroupLayer)

    with pytest.raises(TypeError):
        clone_node(object())  # type: ignore[arg-type]


def test_merge_layers_creates_isolated_group():
    """Valida que merge cria um GroupLayer independente com cópias das camadas."""
    l1 = make_layer((255, 0, 0, 255), (50, 50), name="L1")
    l1.transform.translate(10, 10)

    l2 = make_layer((0, 255, 0, 255), (60, 60), name="L2")
    l2.transform.translate(100, 80)

    group = merge([l1, l2], name="MergedComposition")

    assert isinstance(group, GroupLayer)
    assert group.name == "MergedComposition"
    assert len(group) == 2
    assert group[0] is not l1
    assert group[1] is not l2

    expected_union = reduce(or_, (l1.global_region, l2.global_region))
    assert group.global_region == expected_union

    group[0].transform.translate(500, 500)
    assert not np.allclose(group[0].matrix, l1.matrix)


def test_merge_facade_layer_composition_and_empty_validation():
    """Valida chamada atraves da classe LayerComposition e validacao de lista vazia."""
    l1 = make_layer((255, 0, 0, 255), (50, 50), name="L1")

    res = LayerComposition.merge([l1], name="FacadeGroup")
    assert isinstance(res, GroupLayer)
    assert res.name == "FacadeGroup"

    cloned_node = LayerComposition.clone(l1)
    assert isinstance(cloned_node, Layer)
    assert cloned_node is not l1

    with pytest.raises(ValueError):
        merge([], name="Empty")


def test_flatten_layers_renders_single_rasterized_layer():
    """Valida que flatten renderiza as camadas e retorna um unico Layer com a imagem final e regiao delimitadora."""
    l1 = make_layer((255, 0, 0, 255), (60, 60), name="BottomRed")
    l1.transform.translate(10, 10)

    l2 = make_layer((0, 255, 0, 128), (40, 40), name="TopGreenAlpha")
    l2.transform.rotate(45).translate(30, 30)

    flat = LayerComposition.flatten([l1, l2], name="RasterizedResult")

    assert isinstance(flat, Layer)
    assert flat.name == "RasterizedResult"
    assert len(flat.edits) == 1

    expected_roi = reduce(or_, (l1.global_region, l2.global_region))
    assert flat.global_region == expected_roi

    # Compara a renderizacao direta do container com a renderizacao do layer achatado
    renderer = CanvasRender()
    expected_image = renderer.render_container([l1, l2])
    rendered_flat = renderer.render_layer(flat)

    assert expected_image is not None
    assert rendered_flat is not None
    assert np.array_equal(rendered_flat[...], expected_image[...])


def test_flatten_with_filters_and_masks_bakes_effects():
    """Valida que flatten processa e incorpora filtros e mascaras diretamente no buffer do Layer final."""
    l1 = make_layer((200, 50, 50, 255), (50, 50), name="Blurred")
    l1.bind_effect(BlurFilter(2.0))

    l2 = make_layer((0, 0, 255, 255), (50, 50), name="Masked")
    mask_data = np.full((50, 50), 128, dtype=np.uint8)
    l2.set_mask(Image(mask_data, ImageFormat.GRAY), Region.from_size(50, 50))

    flat = LayerComposition.flatten([l1, l2], name="BakedEffects")

    assert isinstance(flat, Layer)
    assert len(flat.effects) == 0  # Os efeitos ja foram assados no buffer
    assert flat.mask is None  # A mascara ja foi assada no buffer


def test_flatten_empty_and_non_renderable_validations():
    """Valida lancamento de excecoes ao tentar executar flatten em listas vazias ou sem nos renderizaveis."""
    with pytest.raises(ValueError):
        LayerComposition.flatten([], name="Empty")

    l_invisible = make_layer((255, 0, 0, 255), (10, 10))
    l_invisible.visible = False
    with pytest.raises(ValueError):
        LayerComposition.flatten([l_invisible], name="Invisible")


def test_merge_default_name():
    """Valida se merge utiliza 'Group' como nome padrao quando nao especificado."""
    l1 = make_layer((255, 0, 0, 255), (50, 50))
    group = merge([l1])
    assert group.name == "Group"


def test_flatten_default_name():
    """Valida se flatten utiliza 'Layer' como nome padrao quando nao especificado."""
    l1 = make_layer((255, 0, 0, 255), (50, 50))
    flat = flatten([l1])
    assert flat.name == "Layer"


def test_doc_combine_merge_basic_and_order():
    """Valida merge descendente basico no Document e substituicao na posicao correta da stack."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l1 = make_layer((0, 255, 0, 255), name="L1")
    l2 = make_layer((0, 0, 255, 255), name="L2")
    l3 = make_layer((255, 255, 0, 255), name="L3")
    for l in (l0, l1, l2, l3):
        doc.add(l)

    group = doc.combine.merge("L3", name="MergedGroup", count=1)

    assert isinstance(group, GroupLayer)
    assert group.name == "MergedGroup"
    assert len(doc.stack) == 3
    assert [child.name for child in doc.stack] == ["L0", "L1", "MergedGroup"]
    assert len(group) == 2
    assert [child.name for child in group] == ["L2", "L3"]


def test_doc_combine_merge_skips_invisible_layers():
    """Valida que o merge descendente pula camadas com visible=False."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l1 = make_layer((0, 255, 0, 255), name="L1")
    l2 = make_layer((0, 0, 255, 255), name="L2")
    l3 = make_layer((255, 255, 0, 255), name="L3")
    l2.visible = False
    for l in (l0, l1, l2, l3):
        doc.add(l)

    group = doc.combine.merge(l3, name="MergedGroup", count=1)

    assert [child.name for child in group] == ["L1", "L3"]
    assert [child.name for child in doc.stack] == ["L0", "MergedGroup", "L2"]


def test_doc_combine_merge_count_exceeds_available_consumes_all():
    """Valida que count maior que o disponivel consome todas as camadas visiveis abaixo."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l1 = make_layer((0, 255, 0, 255), name="L1")
    l2 = make_layer((0, 0, 255, 255), name="L2")
    for l in (l0, l1, l2):
        doc.add(l)

    group = doc.combine.merge("L2", name="MergedAll", count=10)

    assert len(doc.stack) == 1
    assert doc.stack[0] is group
    assert [child.name for child in group] == ["L0", "L1", "L2"]


def test_doc_combine_merge_nested_inside_group():
    """Valida merge descendente aplicado sobre camadas filhas dentro de um GroupLayer."""
    doc = Document("TestDoc", 200, 200, history=False)
    parent_group = doc.add_group(name="ParentGroup")

    sub0 = make_layer((255, 0, 0, 255), name="Sub0")
    sub1 = make_layer((0, 255, 0, 255), name="Sub1")
    sub2 = make_layer((0, 0, 255, 255), name="Sub2")
    for sub in (sub0, sub1, sub2):
        parent_group.append(sub)

    res = doc.combine.merge("Sub2", name="InnerGroup", count=1)

    assert len(parent_group) == 2
    assert [child.name for child in parent_group] == ["Sub0", "InnerGroup"]
    assert [child.name for child in res] == ["Sub1", "Sub2"]


def test_doc_combine_merge_remove_source_false():
    """Valida que remove_source=False gera o grupo sem alterar a pilha original."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l1 = make_layer((0, 255, 0, 255), name="L1")
    l2 = make_layer((0, 0, 255, 255), name="L2")
    for l in (l0, l1, l2):
        doc.add(l)

    group = doc.combine.merge("L2", name="UnattachedGroup", count=1, remove_source=False)

    assert len(doc.stack) == 3
    assert [child.name for child in doc.stack] == ["L0", "L1", "L2"]
    assert len(group) == 2
    assert [child.name for child in group] == ["L1", "L2"]


def test_doc_combine_name_validation_and_reuse():
    """Valida que e permitido reutilizar nomes das camadas sendo removidas e bloqueia colisoes externas."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l1 = make_layer((0, 255, 0, 255), name="L1")
    l2 = make_layer((0, 0, 255, 255), name="L2")
    l3 = make_layer((255, 255, 0, 255), name="L3")
    for l in (l0, l1, l2, l3):
        doc.add(l)

    # Reutilizar nome de camada que sera removida (L3) deve funcionar
    group = doc.combine.merge("L3", name="L3", count=1, remove_source=True)
    assert group.name == "L3"

    # Colidir com camada existente nao envolvida (L0) deve lancar ValueError
    with pytest.raises(ValueError):
        doc.combine.merge("L3", name="L0", count=1, remove_source=True)


def test_doc_combine_flatten_inherits_target_format():
    """Valida que flatten herda o ImageFormat da camada alvo do topo."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = Layer(Image.new((100, 100), ImageFormat.RGBA), name="L0")
    l1 = Layer(Image.new((100, 100), ImageFormat.RGB), name="L1")
    doc.add(l0)
    doc.add(l1)

    flat = doc.combine.flatten("L1", name="FlatRGB", count=1)

    assert isinstance(flat, Layer)
    assert flat.format == ImageFormat.RGB
    assert len(doc.stack) == 1
    assert doc.stack[0].name == "FlatRGB"


def test_doc_combine_errors_on_invalid_count_and_bottom_layer():
    """Valida lancamento de excecoes em contagem invalida ou sem camadas abaixo."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    doc.add(l0)

    with pytest.raises(ValueError):
        doc.combine.merge("L0", name="G", count=0)

    with pytest.raises(ValueError):
        doc.combine.merge("L0", name="G", count=1)


def test_doc_combine_bake_group_in_place():
    """Valida bake de GroupLayer substituindo o grupo por um Layer no mesmo indice."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l3 = make_layer((0, 255, 0, 255), name="L3")
    doc.add(l0)

    group = doc.add_group(name="CharGroup")
    sub1 = make_layer((0, 0, 255, 255), name="Sub1")
    sub2 = make_layer((255, 255, 0, 255), name="Sub2")
    group.append(sub1)
    group.append(sub2)

    doc.add(l3)

    baked = doc.combine.bake("CharGroup")

    assert isinstance(baked, Layer)
    assert baked.name == "CharGroup"
    assert len(doc.stack) == 3
    assert [child.name for child in doc.stack] == ["L0", "CharGroup", "L3"]
    assert doc.stack[1] is baked


def test_doc_combine_bake_stack():
    """Valida bake da stack completa do documento em uma unica camada plana."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l1 = make_layer((0, 255, 0, 255), name="L1")
    doc.add(l0)
    doc.add(l1)

    baked = doc.combine.bake_stack(name="FullScene")

    assert isinstance(baked, Layer)
    assert len(doc.stack) == 1
    assert doc.stack[0] is baked
    assert doc.stack[0].name == "FullScene"


def test_doc_combine_bake_invalid_target_errors():
    """Valida que bake lanca erro quando target nao e GroupLayer ou nao pertence a um container."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    doc.add(l0)

    with pytest.raises(TypeError):
        doc.combine.bake("L0")

    unattached_group = GroupLayer(name="Unattached")
    with pytest.raises(ValueError):
        doc.combine.bake(unattached_group)


def test_flatten_infers_format_from_top_layer_and_inherits_base_blend():
    """Valida inferencia automatica de formato do topo e heranca de blend_mode da base no flatten."""
    l0 = Layer(Image.new((100, 100), ImageFormat.RGBA), name="L0")
    l0.blend_mode = BlendMode.MULTIPLY
    l1 = Layer(Image.new((100, 100), ImageFormat.RGB), name="L1")
    l1.visible = True

    flat = flatten([l0, l1], name="Flat")

    assert flat.format == ImageFormat.RGB
    assert flat.blend_mode == BlendMode.MULTIPLY
    assert flat.visible is True


def test_doc_combine_flatten_inherits_base_blend_and_target_visibility():
    """Valida que Combine.flatten herda blend_mode da camada base e visibilidade do target."""
    doc = Document("TestDoc", 200, 200, history=False)
    l0 = make_layer((255, 0, 0, 255), name="L0")
    l0.blend_mode = BlendMode.SOLID_FILL
    l1 = make_layer((0, 255, 0, 255), name="L1")
    l1.visible = True
    doc.add(l0)
    doc.add(l1)

    flat = doc.combine.flatten("L1", name="FlatResult", count=1)

    assert flat.blend_mode == BlendMode.SOLID_FILL
    assert flat.visible is True


def test_doc_combine_bake_inherits_group_blend_opacity_and_visibility():
    """Valida que Combine.bake herda blend_mode, opacity e visibility do GroupLayer original."""
    doc = Document("TestDoc", 200, 200, history=False)
    group = doc.add_group(name="EffectGroup")
    group.blend_mode = BlendMode.MULTIPLY
    group.opacity = 0.75
    group.visible = True

    sub1 = make_layer((0, 0, 255, 255), name="Sub1")
    group.append(sub1)

    baked = doc.combine.bake("EffectGroup")

    assert baked.blend_mode == BlendMode.MULTIPLY
    assert baked.opacity == 0.75
    assert baked.visible is True
