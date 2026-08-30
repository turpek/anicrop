from __future__ import annotations
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any, Callable, Optional, Protocol, runtime_checkable, TYPE_CHECKING


from anicrop.interfaces.container import AbstractContainer, AbstractGroupLayer
from anicrop.interfaces.layer import AbstractBaseLayer
from anicrop.layout import GroupLayoutStrategy
from anicrop.effect import Effect, BoundEffect
from anicrop.enums import BlendMode, ImageFormat
from anicrop.geometry import GeometryStrategy, GroupGeometry, GeometryController
from anicrop.mask import Mask
from anicrop.spatial import Region
from anicrop.transform import (
    mat_global,
    mat_inverse,
    Composer,
    ComposerRel,
    Transform,
)

import numpy as np

if TYPE_CHECKING:
    from anicrop.canvas import Canvas
    from anicrop.layer import Layer
    from anicrop.image import Image


@runtime_checkable
class NodeContainerProtocol(Protocol):
    """Protocolo formal para qualquer elemento da árvore espacial/hierárquica."""
    parent: Container
    _parent_inverse: np.ndarray


class NullContainer:

    def __init__(self):
        self.parent: AbstractContainer | NullContainer = self
        self._parent_inverse: np.ndarray = np.identity(3, dtype=np.float32)
        self.__matrix = np.identity(3, dtype=np.float32)
        self._inner_children = []

    @property
    def _children(self) -> list:
        return []

    @_children.setter
    def _children(self, value: list) -> None:
        ...

    @property
    def matrix(self) -> np.ndarray:
        return self.__matrix

    def remove(self, item: BaseLayer) -> None:
        ...


_NULL_CONTAINER = NullContainer()


class Container(NullContainer, AbstractContainer):
    def __init__(self):
        super().__init__()
        self.parent = _NULL_CONTAINER
        self._parent_inverse = np.identity(3, dtype=np.float32)

    @property
    def _children(self) -> list:
        return self._inner_children

    @_children.setter
    def _children(self, value: list) -> None:
        self._inner_children = value

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self):
        return iter(self._children)

    def __reversed__(self):
        return reversed(self._children)

    def __getitem__(self, index: int) -> BaseLayer:
        return self._children[index]

    def clear(self) -> None:
        while self._children:
            self.remove(self._children[-1])

    def pop(self, index: int = -1) -> BaseLayer:
        item = self._children[index]
        self.remove(item)
        return item

    def _check_and_remove_item(self, item: BaseLayer):
        if isinstance(item, LayerStack):
            raise TypeError(
                "A LayerStack is a Root object and cannot be added as a child.")
        if self is item:
            raise ValueError(f"Cannot add a {self.__class__.__name__} to itself")
        elif item in self._children:
            raise ValueError(f"Item {item} is already in this {self.__class__.__name__}")
        elif self.parent is item:
            raise ValueError("Cannot add an ancestor container to a child container")

        item.parent.remove(item)
        item._parent_inverse = mat_inverse(self.matrix)

    def append(self, item: BaseLayer) -> None:
        self._check_and_remove_item(item)
        self._children.append(item)
        item.parent = self

    def insert(self, index: int, item: BaseLayer) -> None:
        self._check_and_remove_item(item)
        self._children.insert(index, item)
        item.parent = self

    def remove(self, item: BaseLayer) -> None:
        if item not in self._children:
            raise ValueError(f"Item {item} is not in this {self.__class__.__name__}")
        self._children.remove(item)
        item.parent = _NULL_CONTAINER
        item._parent_inverse = mat_inverse(item.parent.matrix)

    def move(self, item: BaseLayer, new_index: int) -> None:
        if item not in self._children:
            raise ValueError(f"Item {item} is not in this {self.__class__.__name__}")
        old_index = self._children.index(item)
        if old_index == new_index:
            return
        self._children.pop(old_index)
        self._children.insert(new_index, item)

    def move_relative(self, item: BaseLayer, steps: int) -> None:
        """Move an item by a relative number of steps towards top (positive) or bottom (negative)."""
        if item not in self._children:
            raise ValueError(f"Item {item} is not in this {self.__class__.__name__}")
        if steps == 0:
            return
        old_index = self._children.index(item)
        new_index = max(0, min(len(self._children) - 1, old_index + steps))
        self.move(item, new_index)

    def move_to_front(self, item: BaseLayer) -> None:
        """Move an item to the very top (highest index) of the container."""
        if item not in self._children:
            raise ValueError(f"Item {item} is not in this {self.__class__.__name__}")
        self.move(item, len(self._children) - 1)

    def move_to_back(self, item: BaseLayer) -> None:
        """Move an item to the very bottom (index 0) of the container."""
        if item not in self._children:
            raise ValueError(f"Item {item} is not in this {self.__class__.__name__}")
        self.move(item, 0)

    def swap(self, item_a: BaseLayer, item_b: BaseLayer) -> None:
        """Swap the positions of two items in the container."""
        if item_a not in self._children:
            raise ValueError(f"Item {item_a} is not in this {self.__class__.__name__}")
        if item_b not in self._children:
            raise ValueError(f"Item {item_b} is not in this {self.__class__.__name__}")
        idx_a = self._children.index(item_a)
        idx_b = self._children.index(item_b)
        if idx_a != idx_b:
            self._children[idx_a], self._children[idx_b] = self._children[idx_b], self._children[idx_a]

    def reverse(self, recursive: bool = False) -> None:
        """Reverse the order of children in this container in place."""
        self._children.reverse()
        if recursive:
            for child in self._children:
                if isinstance(child, Container):
                    child.reverse(recursive=True)

    @property
    def matrix(self) -> np.ndarray:
        return np.identity(3, dtype=np.float32)


class BaseLayer(AbstractBaseLayer):

    def __init__(
        self,
        parent: NullContainer | Container | AbstractContainer | Any,
        geometry_cls: Callable[[Any, Region], GeometryStrategy],
        region: Region,

        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'BaseLayer',
        format: ImageFormat = ImageFormat.RGBA,
    ):
        self.parent = parent
        self._transform: Composer = ComposerRel(region.size)
        self._parent_inverse = mat_inverse(parent.matrix)

        self.opacity = opacity
        self.visible = True
        self.blend_mode = blend_mode
        self.name = name
        self._format = format
        self._effects: list[Effect] = []
        self._mask: Mask | None = None

        base = geometry_cls(self, region)
        frame = geometry_cls(self, region)
        self.control = GeometryController(base, frame)

    @property
    def format(self) -> ImageFormat:
        """Formato de cor da camada."""
        return self._format

    @format.setter
    def format(self, value: ImageFormat) -> None:
        if not isinstance(value, ImageFormat):
            raise TypeError(f"Expected ImageFormat, got {type(value).__name__}")
        self._format = value

    @property
    def effects(self) -> tuple[Effect, ...]:
        """Fila de efeitos de pós-processamento aplicados sobre a camada."""
        return tuple(self._effects)

    @property
    def mask(self) -> Mask | None:
        """Retorna a máscara da camada ou None se não houver."""
        return self._mask

    def set_mask(
        self,
        image: Image,
        region: Region,
        invert: bool = False,
        visible: bool = True,
        name: str = "Mask",
    ) -> Mask:
        """Cria e atribui a máscara da camada, vinculando a matriz inversa."""
        matrix = mat_inverse(mat_global(self))
        self._mask = Mask(image, region, matrix, invert=invert, visible=visible, name=name)
        return self._mask

    def remove_mask(self) -> None:
        """Remove a máscara da camada."""
        self._mask = None

    def clear_mask(self) -> None:
        """Alias para remove_mask."""
        self.remove_mask()

    def add_effect(self, effect: Effect) -> Effect:
        """Adiciona um efeito diretamente à fila de pós-processamento da camada."""
        self._effects.append(effect)
        return effect

    def bind_effect(
        self,
        effect: Effect,
        mask: Mask | None = None,
        visible: bool = True,
    ) -> BoundEffect:
        """Cria e adiciona um BoundEffect ancorado à matriz inversa da camada."""
        inv_matrix = mat_inverse(mat_global(self))
        bound = BoundEffect(effect, matrix=inv_matrix, mask=mask, visible=visible)
        self._effects.append(bound)
        return bound

    def remove_effect(self, effect: Effect) -> None:
        """Remove um efeito da camada."""
        self._effects = [e for e in self._effects if e is not effect and getattr(e, "effect", None) is not effect]

    def clear_effects(self) -> None:
        """Remove todos os efeitos de pós-processamento da camada."""
        self._effects.clear()

    def get_effects_padding(self) -> tuple[int, int, int, int]:
        """Calcula o padding total somado/máximo de todos os efeitos ativos e visíveis."""
        top, right, bottom, left = 0, 0, 0, 0
        for effect in self._effects:
            pt, pr, pb, pl = effect.get_padding()
            top = max(top, pt)
            right = max(right, pr)
            bottom = max(bottom, pb)
            left = max(left, pl)
        return top, right, bottom, left

    @property
    def is_renderable(self) -> bool:
        """Indica se a camada deve ser processada no pipeline de renderização."""
        if not (self.visible and self.opacity > 0.0):
            return False
        if self._mask is not None and self._mask.visible:
            m_global = mat_global(self)
            if not self.global_region.overlaps(self._mask.projected_region(m_global)):
                return False
        return True

    @property
    def region(self) -> Region:
        return self.control.frame.region

    @property
    def global_region(self) -> Region:
        return self.control.frame.global_region

    @property
    def matrix(self) -> np.ndarray:
        return self.control.content_matrix

    @property
    def content_matrix(self) -> np.ndarray:
        return self.control.content_matrix

    @property
    def frame_matrix(self) -> np.ndarray:
        return self.control.frame_matrix

    @property
    def base(self) -> GeometryStrategy:
        return self.control.base

    @property
    def frame(self) -> GeometryStrategy:
        return self.control.frame

    @frame.setter
    def frame(self, strategy: GeometryStrategy) -> None:
        self.control.set_strategy(strategy)

    @property
    def transform(self) -> Composer:
        self._transform.sync_region(self.region)
        return self._transform

    def transform_clear(self) -> None:
        self._transform = ComposerRel(self.region.size)

    def set_transform(
        self,
        transform: Transform,
        reference: Optional[Canvas | Layer] = None,
    ) -> None:
        self._transform = transform.create_composer(self.region.size)
        ref_size = reference.region.size if reference is not None else self.region.size
        self._transform.add_transform(transform, reference_size=ref_size)


class LayerStack(Container):
    ...


class GroupLayer(Container, BaseLayer, AbstractGroupLayer):

    def __init__(
        self,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'BaseLayer',
        format: ImageFormat = ImageFormat.RGBA,
    ):
        region = Region.from_size(1, 1)
        Container.__init__(self)
        BaseLayer.__init__(
            self, self.parent, GroupGeometry, region, opacity, blend_mode, name, format=format,
        )
        self._layout = GroupLayoutStrategy(self)

    def __repr__(self):
        return f'GroupLayer(name="{self.name}")'

    def _check_ancestor(self, item: BaseLayer):
        # Verifica ciclos (evita adicionar um pai/avô como filho)
        curr = self.parent
        while curr is not _NULL_CONTAINER:
            if curr == item:
                raise ValueError("Cannot add an ancestor container to a child container")
            curr = curr.parent

    def append(self, item: BaseLayer) -> None:
        self._check_ancestor(item)
        super().append(item)

    def insert(self, index: int, item: BaseLayer) -> None:
        self._check_ancestor(item)
        super().insert(index, item)

    @property
    def matrix(self) -> np.ndarray:
        return self.control.base.matrix

    @property
    def layout(self) -> GroupLayoutStrategy:
        """Estratégia de layout da moldura do grupo."""
        return self._layout


def walk_nodes(root: BaseLayer | Container | Iterable[BaseLayer]) -> Generator[BaseLayer, None, None]:
    """Gera uma travessia preguiçosa de todos os nós a partir de uma raiz ou coleção (DFS)."""
    if isinstance(root, BaseLayer):
        yield root

    if isinstance(root, (Container, Iterable)) and not isinstance(root, (str, bytes)):
        for child in root:
            yield from walk_nodes(child)


@contextmanager
def freeze_geometry(container: Container | Iterable[BaseLayer]) -> Generator[None, None, None]:
    """Congela temporariamente o cálculo de matrizes e regiões com snapshot sob demanda para todos os nós."""
    def _toggle_freeze(enable: bool) -> None:
        for node in walk_nodes(container):
            for strategy in (node.control.base, node.control.frame):
                strategy._cached_matrix = None
                strategy._cached_region = None
                strategy._cached_global_region = None
                strategy._resolve_matrix = strategy._lazy_matrix if enable else strategy._direct_matrix
                strategy._resolve_region = strategy._lazy_region if enable else strategy._direct_region
                strategy._resolve_global_region = strategy._lazy_global_region if enable else strategy._direct_global_region

    _toggle_freeze(True)
    try:
        yield
    finally:
        _toggle_freeze(False)
