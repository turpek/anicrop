from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.effect import BoundEffect, DynamicEffect, Effect
from anicrop.interfaces.cache import AbstractLayerCache
from anicrop.layer import Layer

if TYPE_CHECKING:
    from anicrop.edit_layer import EditLayer
    from anicrop.enums import ImageFormat
    from anicrop.image import Image


def _unwrap_effect(effect: Effect) -> Effect:
    """Desembrulha o efeito se ele estiver encapsulado por um BoundEffect."""
    return effect.effect if isinstance(effect, BoundEffect) else effect


def _is_dynamic_effect(effect: Effect) -> bool:
    """Verifica se o efeito é uma instância de DynamicEffect."""
    return isinstance(_unwrap_effect(effect), DynamicEffect)


class LayerFrameState:
    """Estado e metadados de cache de uma camada registrada."""

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.edits: list[EditLayer] = []
        self.effects: list[Effect] = []
        self.baked_warp: Image | None = None
        self.baked_effects: Image | None = None
        self.background_calls: int = 0

        self.baked_edits_count: int = 0
        self.edits_visibility: tuple[bool, ...] = ()
        self.baked_effects_count: int = 0
        self.effects_visibility: tuple[bool, ...] = ()

        self.orig_add_edit: Any = None
        self.orig_add_effect: Any = None
        self.orig_bind_effect: Any = None
        self.orig_background: Any = None

        self.saved_edits: deque[EditLayer] | None = None
        self.saved_effects: list[Effect] | None = None


def wrap_add_edit(
    status: LayerFrameState, original_func: Callable[..., Any]
) -> Callable[..., Any]:
    """Empacota add_edit para registrar novos edits na lista de deltas do cache."""
    status.orig_add_edit = original_func

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        edit = original_func(*args, **kwargs)
        status.edits.append(edit)
        return edit

    return wrapper


def wrap_add_effect(
    status: LayerFrameState, original_func: Callable[..., Any]
) -> Callable[..., Any]:
    """Empacota add_effect para registrar novos efeitos na lista de deltas do cache."""
    status.orig_add_effect = original_func

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        effect = original_func(*args, **kwargs)
        status.effects.append(effect)
        return effect

    return wrapper


def wrap_bind_effect(
    status: LayerFrameState, original_func: Callable[..., Any]
) -> Callable[..., Any]:
    """Empacota bind_effect para registrar novos efeitos na lista de deltas do cache."""
    status.orig_bind_effect = original_func

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = original_func(*args, **kwargs)
        status.effects.append(bound)
        return bound

    return wrapper


def wrap_background(
    status: LayerFrameState, original_func: Callable[..., Any]
) -> Callable[..., Any]:
    """Empacota background para retornar o buffer pré-assado quando disponível."""
    status.orig_background = original_func

    def wrapper(size: Any, format: ImageFormat, dtype: Any = np.uint8) -> Image:
        if status.baked_warp is None:
            status.baked_warp = original_func(size, format, dtype=dtype)
        status.background_calls += 1
        return status.baked_warp

    return wrapper


class CacheEffect(Effect):
    """Efeito guardião de cache posicionado no início da cadeia de efeitos."""

    def __init__(
        self,
        status: LayerFrameState,
        layer: Layer,
        has_dynamic_following: bool = False,
    ) -> None:
        super().__init__(visible=True, name="CacheEffect")
        self.status = status
        self.layer = layer
        self.has_dynamic_following = has_dynamic_following

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        has_mask = self.layer.mask is not None and self.layer.mask.visible

        # 1. Se o status.baked_warp ainda for None (ex: executou via Fast-Path 1/2 sem chamar background):
        if self.status.baked_warp is None:
            self.status.baked_warp = image.crop()
            image = self.status.baked_warp

        # 2. Se já temos o bake dos efeitos prontos:
        if self.status.baked_effects is not None:
            return (
                self.status.baked_effects.crop()
                if (has_mask or self.has_dynamic_following)
                else self.status.baked_effects
            )

        # 3. Se não temos o bake dos efeitos:
        has_user_effects = bool(
            self.status.saved_effects and any(e.visible for e in self.status.saved_effects)
        )
        if has_mask or has_user_effects:
            return image.crop()

        return image


class CaptureBakedEffectsEffect(Effect):
    """Efeito de captura que armazena a imagem resultante após a execução dos efeitos de usuário estáticos."""

    def __init__(
        self,
        status: LayerFrameState,
        layer: Layer,
        has_dynamic_following: bool = False,
    ) -> None:
        super().__init__(visible=True, name="CaptureBakedEffectsEffect")
        self.status = status
        self.layer = layer
        self.has_dynamic_following = has_dynamic_following

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        has_mask = self.layer.mask is not None and self.layer.mask.visible
        self.status.baked_effects = image
        return (
            image.crop()
            if (has_mask or self.has_dynamic_following)
            else image
        )


def _collect_layers(container: Sequence[BaseLayer] | Container) -> list[Layer]:
    """Coleta recursivamente todas as instâncias de Layer dentro de contêineres e grupos."""
    layers: list[Layer] = []
    for item in container:
        if isinstance(item, Layer):
            layers.append(item)
        elif isinstance(item, (GroupLayer, Container)):
            layers.extend(_collect_layers(item))
    return layers


class LayerCacheScope:
    """Gerenciador de contexto temporário que ativa a orquestração do cache durante o render."""

    def __init__(
        self,
        cache: LayerCache,
        container: Sequence[BaseLayer] | Container,
    ) -> None:
        self._cache = cache
        self._container = container
        self._active_layers: list[Layer] = []

    def __enter__(self) -> LayerCacheScope:
        layers = _collect_layers(self._container)
        for layer in layers:
            if layer in self._cache._states:
                status = self._cache._states[layer]
                self._activate_layer(layer, status)
                self._active_layers.append(layer)
        return self

    def _activate_layer(self, layer: Layer, status: LayerFrameState) -> None:
        status.background_calls = 0

        # 1. Distorção da matriz (rotação e escala 2x2): se mudou, o warp é inválido
        matrix_changed = (
            status.matrix is None
            or not np.allclose(
                layer.matrix[:2, :2], status.matrix[:2, :2], atol=1e-5
            )
        )

        # 2. Verificar visibilidade dos edits que compõem o baked_warp
        edits_vis_changed = False
        if status.baked_warp is not None:
            if len(layer.edits) < status.baked_edits_count:
                edits_vis_changed = True
            else:
                current_base_vis = tuple(
                    layer.edits[i].visible for i in range(status.baked_edits_count)
                )
                if current_base_vis != status.edits_visibility:
                    edits_vis_changed = True

        # 3. Verificar visibilidade dos efeitos estáticos que compõem o baked_effects
        effects_vis_changed = False
        if status.baked_effects is not None:
            if len(layer.effects) < status.baked_effects_count:
                effects_vis_changed = True
            else:
                current_static_vis = tuple(
                    layer.effects[i].visible for i in range(status.baked_effects_count)
                )
                if current_static_vis != status.effects_visibility:
                    effects_vis_changed = True

        # 4. Invalidação de bakes
        if matrix_changed or edits_vis_changed:
            status.baked_warp = None
            status.baked_effects = None
            status.baked_edits_count = 0
            status.edits_visibility = ()
            status.baked_effects_count = 0
            status.effects_visibility = ()
        elif effects_vis_changed or status.edits or status.effects:
            status.baked_effects = None
            status.baked_effects_count = 0
            status.effects_visibility = ()

        # 5. Preparar os edits
        status.saved_edits = layer._edits
        if status.baked_warp is None:
            layer._edits = deque(layer._edits)
            status.baked_edits_count = len(layer.edits)
            status.edits_visibility = tuple(e.visible for e in layer.edits)
        else:
            layer._edits = deque(status.edits)

        # 6. Preparar os efeitos (particionando entre estáticos e dinâmicos)
        status.saved_effects = layer._effects
        visible_user_effects = [e for e in status.saved_effects if e.visible]

        first_dynamic_idx = -1
        for i, eff in enumerate(visible_user_effects):
            if _is_dynamic_effect(eff):
                first_dynamic_idx = i
                break

        if first_dynamic_idx == -1:
            static_effects = visible_user_effects
            dynamic_effects: list[Effect] = []
        elif first_dynamic_idx == 0:
            static_effects = []
            dynamic_effects = visible_user_effects
        else:
            static_effects = visible_user_effects[:first_dynamic_idx]
            dynamic_effects = visible_user_effects[first_dynamic_idx:]

        has_dynamic = bool(dynamic_effects)

        if status.baked_effects is not None:
            layer._effects = [
                CacheEffect(status, layer, has_dynamic_following=has_dynamic),
                *dynamic_effects,
            ]
        else:
            if static_effects:
                status.baked_effects_count = len(static_effects)
                status.effects_visibility = tuple(e.visible for e in static_effects)
                layer._effects = [
                    CacheEffect(status, layer, has_dynamic_following=True),
                    *static_effects,
                    CaptureBakedEffectsEffect(
                        status, layer, has_dynamic_following=has_dynamic
                    ),
                    *dynamic_effects,
                ]
            elif dynamic_effects:
                layer._effects = [
                    CacheEffect(status, layer, has_dynamic_following=True),
                    *dynamic_effects,
                ]
            else:
                layer._effects = [CacheEffect(status, layer)]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        for layer in self._active_layers:
            status = self._cache._states[layer]
            self._deactivate_layer(layer, status)
        self._active_layers.clear()

    def _deactivate_layer(self, layer: Layer, status: LayerFrameState) -> None:
        if status.saved_edits is not None:
            layer._edits = status.saved_edits
            status.saved_edits = None
        if status.saved_effects is not None:
            layer._effects = status.saved_effects
            status.saved_effects = None

        status.matrix = layer.matrix.copy()
        status.edits.clear()
        status.effects.clear()


class LayerCache(AbstractLayerCache):
    """Gerenciador central de cache incremental e decorators para renderização."""

    def __init__(self) -> None:
        self._states: dict[Layer, LayerFrameState] = {}

    def register(self, item: Layer | Container) -> None:
        """Registra a camada ou contêiner (recursivo) para gerenciamento de cache."""
        if isinstance(item, Container):
            for child in item:
                if isinstance(child, (Layer, Container)):
                    self.register(child)
            return

        if not isinstance(item, Layer):
            return

        layer = item
        if layer in self._states:
            return

        status = LayerFrameState()
        layer.add_edit = wrap_add_edit(status, layer.add_edit)  # type: ignore[method-assign]
        layer.add_effect = wrap_add_effect(status, layer.add_effect)  # type: ignore[method-assign]
        layer.bind_effect = wrap_bind_effect(status, layer.bind_effect)  # type: ignore[method-assign]
        layer.background = wrap_background(status, layer.background)  # type: ignore[method-assign]
        self._states[layer] = status

    def unregister(self, item: Layer | Container) -> None:
        """Remove a camada ou contêiner do gerenciamento de cache e restaura seu estado original."""
        if isinstance(item, Container):
            for child in item:
                if isinstance(child, (Layer, Container)):
                    self.unregister(child)
            return

        if not isinstance(item, Layer):
            return

        layer = item
        status = self._states.pop(layer, None)
        if status is not None:
            if status.orig_add_edit is not None:
                layer.add_edit = status.orig_add_edit  # type: ignore[method-assign]
            if status.orig_add_effect is not None:
                layer.add_effect = status.orig_add_effect  # type: ignore[method-assign]
            if status.orig_bind_effect is not None:
                layer.bind_effect = status.orig_bind_effect  # type: ignore[method-assign]
            if status.orig_background is not None:
                layer.background = status.orig_background  # type: ignore[method-assign]

    def is_dirty(self, layer: Layer) -> bool:
        """Verifica se a camada precisa ser renderizada do zero."""
        if layer not in self._states:
            return True
        status = self._states[layer]
        if status.baked_warp is None or status.matrix is None:
            return True
        return not np.allclose(
            layer.matrix[:2, :2], status.matrix[:2, :2], atol=1e-5
        )

    def set_baked(
        self,
        layer: Layer,
        image: Image,
        matrix: np.ndarray | None = None,
    ) -> None:
        """Injeta externamente um buffer pré-assado na camada (ex: Anifuse 2-pass)."""
        if layer not in self._states:
            self.register(layer)
        status = self._states[layer]
        status.baked_warp = image
        status.baked_effects = None
        status.matrix = layer.matrix.copy() if matrix is None else matrix.copy()
        status.edits.clear()
        status.effects.clear()
        status.baked_edits_count = len(layer.edits)
        status.edits_visibility = tuple(e.visible for e in layer.edits)
        status.baked_effects_count = 0
        status.effects_visibility = ()

    def get_state(self, layer: Layer) -> LayerFrameState | None:
        """Retorna o estado de cache da camada, se registrada."""
        return self._states.get(layer)

    def __call__(
        self, container: Sequence[BaseLayer] | Container
    ) -> LayerCacheScope:
        """Cria um escopo de contexto para ativação de cache durante a renderização."""
        return LayerCacheScope(self, container)
