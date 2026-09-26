from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from anicrop.cache import (
    LayerCache,
    LayerFrameState,
    wrap_add_edit,
    wrap_add_effect,
    wrap_background,
)
from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.effect import DynamicEffect, Effect
from anicrop.enums import ImageFormat
from anicrop.filter import BlurFilter
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import CanvasRender
from anicrop.spatial import Region


def make_img(
    w: int = 40,
    h: int = 40,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
    fmt: ImageFormat = ImageFormat.RGBA,
) -> Image:
    img_data = np.zeros((h, w, fmt.channels), dtype=np.uint8)
    img_data[:] = color
    return Image(img_data, fmt)


def test_wrap_add_edit_registra_edit_no_status():
    """Valida se wrap_add_edit chama a funcao original e adiciona o retorno na lista edits do status."""
    status = LayerFrameState()
    mock_edit = MagicMock()
    mock_orig = MagicMock(return_value=mock_edit)

    wrapped = wrap_add_edit(status, mock_orig)
    result = wrapped("arg1", key="val")

    assert result is mock_edit
    assert status.orig_add_edit is mock_orig
    mock_orig.assert_called_once_with("arg1", key="val")
    assert status.edits == [mock_edit]


def test_wrap_add_effect_registra_efeito_no_status():
    """Valida se wrap_add_effect chama a funcao original e adiciona o retorno na lista effects do status."""
    status = LayerFrameState()
    mock_effect = MagicMock()
    mock_orig = MagicMock(return_value=mock_effect)

    wrapped = wrap_add_effect(status, mock_orig)
    result = wrapped(mock_effect)

    assert result is mock_effect
    assert status.orig_add_effect is mock_orig
    mock_orig.assert_called_once_with(mock_effect)
    assert status.effects == [mock_effect]


def test_wrap_background_retorna_baked_warp_e_incrementa_contador():
    """Valida se wrap_background entrega baked_warp e incrementa background_calls quando o bake existe."""
    status = LayerFrameState()
    baked = Image.new((20, 20), ImageFormat.RGBA)
    status.baked_warp = baked

    mock_orig = MagicMock()
    wrapped = wrap_background(status, mock_orig)
    result = wrapped((20, 20), ImageFormat.RGBA)

    assert result is baked
    assert status.orig_background is mock_orig
    assert status.background_calls == 1
    mock_orig.assert_not_called()


def test_wrap_background_delega_para_original_quando_sem_bake():
    """Valida se wrap_background cria e armazena baked_warp via funcao original quando for None."""
    status = LayerFrameState()
    fallback_img = Image.new((30, 30), ImageFormat.RGBA)
    mock_orig = MagicMock(return_value=fallback_img)

    wrapped = wrap_background(status, mock_orig)
    result = wrapped((30, 30), ImageFormat.RGBA, dtype=np.uint8)

    assert result is fallback_img
    assert status.baked_warp is fallback_img
    assert status.background_calls == 1
    mock_orig.assert_called_once_with((30, 30), ImageFormat.RGBA, dtype=np.uint8)


def test_layer_cache_register_and_unregister():
    """Valida se register decora os metodos da camada e unregister restaura as referencias originais."""
    layer = Layer(make_img(30, 30))
    orig_add_edit = layer.add_edit
    orig_add_effect = layer.add_effect
    orig_bg = layer.background
    cache = LayerCache()

    cache.register(layer)
    assert layer.add_edit != orig_add_edit
    assert layer.add_effect != orig_add_effect
    assert layer.background != orig_bg

    cache.unregister(layer)
    assert layer.add_edit == orig_add_edit
    assert layer.add_effect == orig_add_effect
    assert layer.background == orig_bg
    assert cache.get_state(layer) is None


def test_layer_cache_register_container_recursively():
    """Valida se register em um GroupLayer registra recursivamente todas as camadas filhas."""
    layer1 = Layer(make_img(20, 20))
    layer2 = Layer(make_img(20, 20))
    group = GroupLayer()
    group.append(layer1)
    group.append(layer2)
    cache = LayerCache()

    cache.register(group)
    assert cache.get_state(layer1) is not None
    assert cache.get_state(layer2) is not None

    cache.unregister(group)
    assert cache.get_state(layer1) is None
    assert cache.get_state(layer2) is None


def test_layer_cache_is_dirty_states():
    """Valida o ciclo de dirty quando a camada nao esta assada ou quando a matriz e alterada."""
    layer = Layer(make_img(30, 30))
    cache = LayerCache()

    assert cache.is_dirty(layer) is True

    cache.register(layer)
    assert cache.is_dirty(layer) is True

    cache.set_baked(layer, make_img(30, 30))
    assert cache.is_dirty(layer) is False

    layer.transform.translate(10, 5)
    assert cache.is_dirty(layer) is True


def test_layer_cache_set_baked_injection():
    """Valida a injecao direta de buffer pré-assado com sincronizacao de matriz."""
    layer = Layer(make_img(30, 30))
    cache = LayerCache()
    baked = make_img(30, 30, (0, 255, 0, 255))

    cache.set_baked(layer, baked)
    status = cache.get_state(layer)

    assert status is not None
    assert status.baked_warp is baked
    assert status.baked_effects is None
    assert status.matrix is not None
    assert np.allclose(status.matrix, layer.matrix)


def test_layer_cache_scope_swaps_and_restores_deltas():
    """Valida se o escopo troca os edits da camada por deltas durante o render e restaura ao sair."""
    layer = Layer(make_img(40, 40))
    cache = LayerCache()
    cache.register(layer)

    renderer = CanvasRender()
    canvas = Canvas(layer.global_region)

    # 1. Render inicial: estabelece baked_warp
    renderer.render_scene([layer], canvas, cache=cache)
    status = cache.get_state(layer)
    assert status is not None
    assert status.baked_warp is not None
    assert len(layer.edits) == 1

    # 2. Adiciona um novo retalho (delta)
    patch_img = make_img(20, 20, (0, 0, 255, 255))
    layer.add_edit(patch_img, Region.from_size(20, 20))
    assert len(layer.edits) == 2
    assert len(status.edits) == 1

    # 3. Durante o contexto with cache([layer]), a camada deve enxergar apenas o delta
    with cache([layer]):
        assert len(layer.edits) == 1

    # 4. Apos o contexto, a camada restaura a visao total de edits e limpa deltas
    assert len(layer.edits) == 2
    assert len(status.edits) == 0


def test_layer_cache_render_incremental_matches_clean_render():
    """Valida se a renderizacao incremental com cache produz o mesmo resultado de uma renderizacao direta."""
    layer_cached = Layer(make_img(40, 40, (10, 10, 10, 255)))
    layer_clean = Layer(make_img(40, 40, (10, 10, 10, 255)))
    cache = LayerCache()
    cache.register(layer_cached)

    renderer = CanvasRender()
    canvas_cached = Canvas(layer_cached.global_region)
    canvas_clean = Canvas(layer_clean.global_region)

    # Frame 1: Base renderizada
    renderer.render_scene([layer_cached], canvas_cached, cache=cache)
    renderer.render_scene([layer_clean], canvas_clean)

    # Frame 2: Adiciona edit incremental
    patch = make_img(15, 15, (200, 50, 50, 255))
    layer_cached.add_edit(patch, Region.from_size(15, 15))
    layer_clean.add_edit(patch, Region.from_size(15, 15))

    out_cached = renderer.render_scene([layer_cached], canvas_cached, cache=cache)
    out_clean = renderer.render_scene([layer_clean], canvas_clean)

    status = cache.get_state(layer_cached)
    assert status is not None
    assert status.background_calls >= 1
    assert np.array_equal(out_cached[...], out_clean[...])


def test_layer_cache_with_effects_and_mask_preserves_baked_warp():
    """Valida se o baked_warp permanece intacto quando ha efeitos e modulacao por mascara."""
    layer = Layer(make_img(40, 40, (255, 255, 255, 255)))
    blur = BlurFilter(radius=2.0)
    layer.add_effect(blur)

    mask_img = make_img(40, 40, (128, 128, 128, 255))
    layer.set_mask(mask_img, Region.from_size(40, 40))

    cache = LayerCache()
    cache.register(layer)

    renderer = CanvasRender()
    canvas = Canvas(layer.global_region)

    # Renderiza com cache
    renderer.render_scene([layer], canvas, cache=cache)
    status = cache.get_state(layer)

    assert status is not None
    assert status.baked_warp is not None
    assert status.baked_effects is not None
    assert np.all(status.baked_warp[..., 3] == 255)
    assert np.all(status.baked_effects[..., 3] == 255)


class CountingDynamicEffect(DynamicEffect):
    def __init__(self) -> None:
        super().__init__(name="CountingDynamicEffect")
        self.apply_count = 0

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        self.apply_count += 1
        return image


class CountingStaticEffect(Effect):
    def __init__(self) -> None:
        super().__init__(name="CountingStaticEffect")
        self.apply_count = 0

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        self.apply_count += 1
        return image


def test_layer_cache_runs_dynamic_effect_every_frame():
    """Valida se DynamicEffect eh reexecutado a cada frame enquanto baked_warp permanece em cache."""
    layer = Layer(make_img(40, 40))
    dyn_effect = CountingDynamicEffect()
    layer.add_effect(dyn_effect)

    cache = LayerCache()
    cache.register(layer)

    renderer = CanvasRender()
    canvas = Canvas(layer.global_region)

    renderer.render_scene([layer], canvas, cache=cache)
    assert dyn_effect.apply_count == 1
    status = cache.get_state(layer)
    assert status is not None
    assert status.baked_warp is not None
    assert status.baked_effects is None

    renderer.render_scene([layer], canvas, cache=cache)
    assert dyn_effect.apply_count == 2
    assert status.baked_warp is not None
    assert status.baked_effects is None


def test_layer_cache_bakes_static_effects_and_executes_dynamic_effects_incrementally():
    """Valida se efeitos estaticos sao assados e apenas os dinamicos rodam nos frames subsequentes."""
    layer = Layer(make_img(40, 40))
    static_effect = CountingStaticEffect()
    dyn_effect = CountingDynamicEffect()
    layer.add_effect(static_effect)
    layer.add_effect(dyn_effect)

    cache = LayerCache()
    cache.register(layer)

    renderer = CanvasRender()
    canvas = Canvas(layer.global_region)

    renderer.render_scene([layer], canvas, cache=cache)
    assert static_effect.apply_count == 1
    assert dyn_effect.apply_count == 1
    status = cache.get_state(layer)
    assert status is not None
    assert status.baked_effects is not None

    renderer.render_scene([layer], canvas, cache=cache)
    assert static_effect.apply_count == 1
    assert dyn_effect.apply_count == 2


def test_layer_cache_recognizes_bound_dynamic_effect():
    """Valida se DynamicEffect envelopado por BoundEffect eh reconhecido como dinamico."""
    layer = Layer(make_img(40, 40))
    dyn_effect = CountingDynamicEffect()
    layer.bind_effect(dyn_effect)

    cache = LayerCache()
    cache.register(layer)

    renderer = CanvasRender()
    canvas = Canvas(layer.global_region)

    renderer.render_scene([layer], canvas, cache=cache)
    assert dyn_effect.apply_count == 1

    renderer.render_scene([layer], canvas, cache=cache)
    assert dyn_effect.apply_count == 2


def test_layer_cache_invalidates_baked_effects_when_static_effect_visibility_changes():
    """Valida se alteracao na visibilidade de um efeito estatico invalida o baked_effects."""
    layer = Layer(make_img(40, 40))
    static_effect = CountingStaticEffect()
    layer.add_effect(static_effect)

    cache = LayerCache()
    cache.register(layer)

    renderer = CanvasRender()
    canvas = Canvas(layer.global_region)

    renderer.render_scene([layer], canvas, cache=cache)
    assert static_effect.apply_count == 1
    status = cache.get_state(layer)
    assert status is not None
    assert status.baked_effects is not None

    static_effect.visible = False
    renderer.render_scene([layer], canvas, cache=cache)
    assert static_effect.apply_count == 1
    assert status.baked_effects is None


def test_layer_cache_invalidates_baked_warp_when_base_edit_visibility_changes():
    """Valida se alteracao na visibilidade de um edit base invalida o baked_warp."""
    layer = Layer(make_img(40, 40))
    cache = LayerCache()
    cache.register(layer)

    renderer = CanvasRender()
    canvas = Canvas(layer.global_region)

    renderer.render_scene([layer], canvas, cache=cache)
    status = cache.get_state(layer)
    assert status is not None
    assert status.baked_warp is not None

    orig_warp = status.baked_warp
    layer.edits[0].visible = False
    renderer.render_scene([layer], canvas, cache=cache)
    assert status.baked_warp is not orig_warp
