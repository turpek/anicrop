from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.effect import Effect
from anicrop.interfaces.cache import AbstractLayerCache
from anicrop.layer import Layer

if TYPE_CHECKING:
    from anicrop.edit_layer import EditLayer
    from anicrop.enums import ImageFormat
    from anicrop.image import Image


class LayerFrameState:
    """Estado e metadados de cache de uma camada registrada."""

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.edits: list[EditLayer] = []
        self.effects: list[Effect] = []
        self.baked_warp: Image | None = None
        self.baked_effects: Image | None = None
        self.background_calls: int = 0

        self.orig_add_edit: Any = None
        self.orig_add_effect: Any = None
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

    def __init__(self, status: LayerFrameState, layer: Layer) -> None:
        super().__init__(visible=True, name="CacheEffect")
        self.status = status
        self.layer = layer

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
                if has_mask
                else self.status.baked_effects
            )

        # 3. Se não temos o bake dos efeitos:
        # Se outros efeitos de usuário vão rodar a seguir, ou se há máscara ativa,
        # isola com crop para que o status.baked_warp nunca seja mutado in-place:
        has_user_effects = bool(
            self.status.saved_effects and any(e.visible for e in self.status.saved_effects)
        )
        if has_mask or has_user_effects:
            return image.crop()

        return image


class CaptureBakedEffectsEffect(Effect):
    """Efeito de captura que armazena a imagem resultante após a execução dos efeitos de usuário."""

    def __init__(self, status: LayerFrameState, layer: Layer) -> None:
        super().__init__(visible=True, name="CaptureBakedEffectsEffect")
        self.status = status
        self.layer = layer

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        return None

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        has_mask = self.layer.mask is not None and self.layer.mask.visible
        self.status.baked_effects = image
        return image.crop() if has_mask else image


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
        # 1. Verificar matriz
        matrix_changed = (
            status.matrix is None
            or not np.allclose(layer.matrix, status.matrix, atol=1e-5)
        )

        # 2. Limpar os bakes se necessário
        if matrix_changed:
            status.baked_warp = None
            status.baked_effects = None
        elif status.edits or status.effects:
            status.baked_effects = None

        # 3. Preparar os edits
        status.saved_edits = layer._edits
        if status.baked_warp is None:
            layer._edits = deque(layer._edits)
        else:
            layer._edits = deque(status.edits)

        # 4. Preparar os efeitos (CacheEffect sempre roda primeiro)
        status.saved_effects = layer._effects
        if status.baked_effects is not None:
            layer._effects = [CacheEffect(status, layer)]
        else:
            visible_user_effects = [e for e in status.saved_effects if e.visible]
            if visible_user_effects:
                layer._effects = [
                    CacheEffect(status, layer),
                    *status.saved_effects,
                    CaptureBakedEffectsEffect(status, layer),
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
            if status.orig_background is not None:
                layer.background = status.orig_background  # type: ignore[method-assign]

    def is_dirty(self, layer: Layer) -> bool:
        """Verifica se a camada precisa ser renderizada do zero."""
        if layer not in self._states:
            return True
        status = self._states[layer]
        if status.baked_warp is None or status.matrix is None:
            return True
        return not np.allclose(layer.matrix, status.matrix, atol=1e-5)

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

    def get_state(self, layer: Layer) -> LayerFrameState | None:
        """Retorna o estado de cache da camada, se registrada."""
        return self._states.get(layer)

    def __call__(
        self, container: Sequence[BaseLayer] | Container
    ) -> LayerCacheScope:
        """Cria um escopo de contexto para ativação de cache durante a renderização."""
        return LayerCacheScope(self, container)
