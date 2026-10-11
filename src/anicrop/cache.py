from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence
from weakref import WeakKeyDictionary, ref

import numpy as np

from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.edit_layer import EditStack
from anicrop.effect import BoundEffect, Effect, EffectStack
from anicrop.interfaces.cache import AbstractLayerCache, Cacheable
from anicrop.layer import Layer
from anicrop.spatial import Region

if TYPE_CHECKING:
    from anicrop.edit_layer import EditLayer
    from anicrop.enums import ImageFormat
    from anicrop.image import Image


def is_cacheable_effect(effect: Effect) -> bool:
    """Verifica se o efeito é estático e elegível para cache."""
    target = effect
    while isinstance(target, BoundEffect):
        target = target.effect
    return isinstance(target, Cacheable)


def snapshot_effect(effect: Effect) -> tuple[ref[Effect], dict[str, Any]]:
    """Gera uma tupla imutável com weakref e dicionário de estado observável do efeito."""
    state = effect.cache_state() if isinstance(effect, Cacheable) else {}
    return (ref(effect), state)


def snapshot_edit(edit: EditLayer) -> tuple[ref[EditLayer], dict[str, Any]]:
    """Retorna tupla com weakref e dicionário de estado observável da edição."""
    state = (
        edit.cache_state()
        if isinstance(edit, Cacheable)
        else {"visible": edit.visible, "blend_mode": edit.blend_mode}
    )
    return (ref(edit), state)


class LayerFrameState:
    """Estado e metadados de cache de uma camada registrada."""

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.matrix_2x2_bytes: bytes | None = None
        self.baked_warp: Image | None = None
        self.baked_effects: Image | None = None
        self.background_calls: int = 0

        self.baked_edits_snapshot: tuple[tuple[ref[EditLayer], dict[str, Any]], ...] = ()
        self.baked_effects_snapshot: tuple[tuple[ref[Effect], dict[str, Any]], ...] = ()

        self.saved_edits: list[EditLayer] | None = None
        self.saved_effects: list[Effect] | None = None


def wrap_background(
    status: LayerFrameState, original_func: Callable[..., Any]
) -> Callable[..., Any]:
    """Empacota background para retornar o buffer pré-assado quando disponível."""
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

        # 1. Se já temos baked_warp e o renderizador usou Fast-Path (background_calls == 0):
        if self.status.baked_warp is not None and self.status.background_calls == 0:
            bg = self.layer.background(image.size, self.layer.format, dtype=image.dtype)
            active_edit = next((e for e in self.layer.edits if e.visible), None)
            if active_edit is not None:
                active_edit.blend_into(
                    bg, image, Region.from_size(*image.size)
                )
            image = bg
        elif self.status.baked_warp is None:
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
        static_effects: list[Effect] | None = None,
        has_dynamic_following: bool = False,
    ) -> None:
        super().__init__(visible=True, name="CaptureBakedEffectsEffect")
        self.status = status
        self.layer = layer
        self.static_effects = static_effects or []
        self.has_dynamic_following = has_dynamic_following

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        has_mask = self.layer.mask is not None and self.layer.mask.visible
        self.status.baked_effects = image
        self.status.baked_effects_snapshot = tuple(
            snapshot_effect(e) for e in self.static_effects
        )
        return (
            image.crop()
            if (has_mask or self.has_dynamic_following)
            else image
        )


def _collect_layers(container: Sequence[BaseLayer] | Container) -> list[Layer]:
    """Coleta recursivamente todas as instâncias de Layer dentro de contêineres e grupos."""
    layers: list[Layer] = []
    for item in container:
        target = getattr(item, "_target", item)
        if isinstance(target, Layer):
            layers.append(target)
        elif isinstance(target, (GroupLayer, Container)):
            layers.extend(_collect_layers(target))
    return layers


class LayerCacheScope:
    """Gerenciador de contexto temporário que ativa a orquestração do cache durante o render."""

    def __init__(
        self,
        cache: LayerCache,
        container: Sequence[BaseLayer] | Container,
        effective_region: Region | None = None,
    ) -> None:
        self._cache = cache
        self._container = container
        self._effective_region = effective_region
        self._active_layers: list[Layer] = []

    def __enter__(self) -> LayerCacheScope:
        try:
            layers = _collect_layers(self._container)
            for layer in layers:
                if layer in self._cache._states:
                    if self._effective_region is not None:
                        if not self._effective_region.overlaps(layer.global_region):
                            continue
                        if (
                            layer.global_region & self._effective_region
                        ).size != layer.global_region.size:
                            continue

                    status = self._cache._states[layer]
                    self._active_layers.append(layer)
                    self._activate_layer(layer, status)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def _activate_layer(self, layer: Layer, status: LayerFrameState) -> None:
        status.background_calls = 0
        layer.background = wrap_background(status, layer.background)  # type: ignore[method-assign]

        # 1. Distorção da matriz (rotação e escala 2x2): se mudou, o warp é inválido
        curr_2x2_bytes = layer.matrix[:2, :2].tobytes()
        matrix_changed = (
            status.matrix_2x2_bytes is None
            or curr_2x2_bytes != status.matrix_2x2_bytes
        )

        # 2. Verificar integridade dos edits que compõem o baked_warp
        edits_changed = False
        if status.baked_warp is not None:
            if len(layer._edits) < len(status.baked_edits_snapshot):
                edits_changed = True
            else:
                for (saved_ref, saved_state), current in zip(
                    status.baked_edits_snapshot, layer._edits
                ):
                    curr_state = (
                        current.cache_state()
                        if isinstance(current, Cacheable)
                        else {"visible": current.visible, "blend_mode": current.blend_mode}
                    )
                    if saved_ref() is not current or saved_state != curr_state:
                        edits_changed = True
                        break

        # 3. Particionar efeitos entre estáticos e dinâmicos e verificar snapshot
        status.saved_effects = layer._effects
        visible_user_effects = [e for e in status.saved_effects if e.visible]

        first_dynamic_idx = -1
        for i, eff in enumerate(visible_user_effects):
            if not is_cacheable_effect(eff):
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

        current_static_snapshot = tuple(snapshot_effect(e) for e in static_effects)
        effects_changed = False
        if status.baked_effects is not None:
            if current_static_snapshot != status.baked_effects_snapshot:
                effects_changed = True

        # 4. Invalidação de bakes
        has_new_edits = (
            status.baked_warp is not None
            and len(layer._edits) > len(status.baked_edits_snapshot)
        )

        if matrix_changed or edits_changed:
            status.baked_warp = None
            status.baked_effects = None
            status.baked_edits_snapshot = ()
            status.baked_effects_snapshot = ()
        elif effects_changed or has_new_edits:
            status.baked_effects = None
            status.baked_effects_snapshot = ()

        # 5. Preparar os edits
        status.saved_edits = layer._edits
        if status.baked_warp is None:
            status.baked_edits_snapshot = tuple(snapshot_edit(e) for e in layer._edits)
            layer._edits = EditStack(layer._edits)
        else:
            layer._edits = EditStack(layer._edits[len(status.baked_edits_snapshot):])

        # 6. Preparar os efeitos
        if status.baked_effects is not None:
            layer._effects = EffectStack([
                CacheEffect(status, layer, has_dynamic_following=has_dynamic),
                *dynamic_effects,
            ])
        else:
            if static_effects:
                layer._effects = EffectStack([
                    CacheEffect(status, layer, has_dynamic_following=True),
                    *static_effects,
                    CaptureBakedEffectsEffect(
                        status,
                        layer,
                        static_effects=static_effects,
                        has_dynamic_following=has_dynamic,
                    ),
                    *dynamic_effects,
                ])
            elif dynamic_effects:
                layer._effects = EffectStack([
                    CacheEffect(status, layer, has_dynamic_following=True),
                    *dynamic_effects,
                ])
            else:
                layer._effects = EffectStack([CacheEffect(status, layer)])

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
        layer.__dict__.pop("background", None)

        status.matrix = layer.matrix.copy()
        status.matrix_2x2_bytes = layer.matrix[:2, :2].tobytes()
        if status.baked_warp is not None:
            status.baked_edits_snapshot = tuple(snapshot_edit(e) for e in layer._edits)


class LayerCache(AbstractLayerCache):
    """Gerenciador central de cache incremental e decorators para renderização."""

    def __init__(self) -> None:
        self._states: WeakKeyDictionary[Layer, LayerFrameState] = WeakKeyDictionary()

    def register(self, item: Layer | Container) -> None:
        """Registra a camada ou contêiner (recursivo) para gerenciamento de cache."""
        target = getattr(item, "_target", item)
        if isinstance(target, Container):
            for child in target:
                if isinstance(child, (Layer, Container)):
                    self.register(child)
            return

        if not isinstance(target, Layer):
            return

        layer = target
        if layer in self._states:
            return

        status = LayerFrameState()
        self._states[layer] = status

    def unregister(self, item: Layer | Container) -> None:
        """Remove a camada ou contêiner do gerenciamento de cache e restaura seu estado original."""
        target = getattr(item, "_target", item)
        if isinstance(target, Container):
            for child in target:
                if isinstance(child, (Layer, Container)):
                    self.unregister(child)
            return

        if not isinstance(target, Layer):
            return

        layer = target
        self._states.pop(layer, None)
        layer.__dict__.pop("background", None)

    def is_dirty(self, layer: Layer) -> bool:
        """Verifica se a camada precisa ser renderizada do zero."""
        target = getattr(layer, "_target", layer)
        if target not in self._states:
            return True
        status = self._states[target]
        if status.baked_warp is None or status.matrix_2x2_bytes is None:
            return True
        if target.matrix[:2, :2].tobytes() != status.matrix_2x2_bytes:
            return True
        if len(target._edits) != len(status.baked_edits_snapshot):
            return True
        for (saved_ref, saved_state), current in zip(
            status.baked_edits_snapshot, target._edits
        ):
            curr_state = (
                current.cache_state()
                if isinstance(current, Cacheable)
                else {"visible": current.visible, "blend_mode": current.blend_mode}
            )
            if saved_ref() is not current or saved_state != curr_state:
                return True
        if status.baked_effects is not None:
            visible_user_effects = [e for e in target._effects if e.visible]
            static_effects = []
            for eff in visible_user_effects:
                if not is_cacheable_effect(eff):
                    break
                static_effects.append(eff)
            current_static_snapshot = tuple(snapshot_effect(e) for e in static_effects)
            if current_static_snapshot != status.baked_effects_snapshot:
                return True
        return False

    def set_baked(
        self,
        layer: Layer,
        image: Image,
        matrix: np.ndarray | None = None,
    ) -> None:
        """Injeta externamente um buffer pré-assado na camada (ex: Anifuse 2-pass)."""
        target = getattr(layer, "_target", layer)
        if target not in self._states:
            self.register(target)
        status = self._states[target]
        status.baked_warp = image
        status.baked_effects = None
        status.matrix = target.matrix.copy() if matrix is None else matrix.copy()
        status.matrix_2x2_bytes = status.matrix[:2, :2].tobytes()
        status.baked_edits_snapshot = tuple(snapshot_edit(e) for e in target._edits)
        status.baked_effects_snapshot = ()

    def get_state(self, layer: Layer) -> LayerFrameState | None:
        """Retorna o estado de cache da camada, se registrada."""
        target = getattr(layer, "_target", layer)
        return self._states.get(target)

    def __call__(
        self,
        container: Sequence[BaseLayer] | Container,
        effective_region: Region | None = None,
    ) -> LayerCacheScope:
        """Cria um escopo de contexto para ativação de cache durante a renderização."""
        return LayerCacheScope(self, container, effective_region=effective_region)
