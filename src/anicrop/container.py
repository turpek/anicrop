from __future__ import annotations
from abc import ABC
from functools import reduce
from typing import Optional, Callable, Any, Protocol, runtime_checkable, TYPE_CHECKING
from operator import or_

from anicrop.canvas import Canvas
from anicrop.blend import blend_rendered_images
from anicrop.enums import ImageFormat, BlendMode, InterpolationOption
from anicrop.image import Image
from anicrop.spatial import Region
from anicrop.viewport import Viewport
from anicrop.transform import (
    mat_inverse,
    Composer,
    ComposerRel,
    Transform,
)

import numpy as np

if TYPE_CHECKING:
    from anicrop.layer import Layer
    from anicrop.render import BaseRenderPlan


@runtime_checkable
class NodeContainerProtocol(Protocol):
    """Protocolo formal para qualquer elemento da árvore espacial/hierárquica."""
    parent: Container
    _parent_inverse: np.ndarray


class NullContainer(ABC):

    def __init__(self):
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

    def remove(self, item: Container | Layer) -> None:
        ...


_NULL_CONTAINER = NullContainer()


class Container(NullContainer):
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

    def __getitem__(self, index: int) -> Container | Layer:
        return self._children[index]

    def clear(self) -> None:
        while self._children:
            self.remove(self._children[-1])

    def pop(self, index: int = -1) -> Container | Layer:
        item = self._children[index]
        self.remove(item)
        return item

    def _check_and_remove_item(self, item: Container | Layer):
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

    def append(self, item: Container | Layer) -> None:
        self._check_and_remove_item(item)
        self._children.append(item)
        item.parent = self

    def insert(self, index: int, item: Container | Layer) -> None:
        self._check_and_remove_item(item)
        self._children.insert(index, item)
        item.parent = self

    def remove(self, item: Container | Layer) -> None:
        if item not in self._children:
            raise ValueError(f"Item {item} is not in this {self.__class__.__name__}")
        self._children.remove(item)
        item.parent = _NULL_CONTAINER
        item._parent_inverse = mat_inverse(item.parent.matrix)

    def move(self, item: Container | Layer, new_index: int) -> None:
        if item not in self._children:
            raise ValueError(f"Item {item} is not in this {self.__class__.__name__}")
        old_index = self._children.index(item)
        if old_index == new_index:
            return
        self._children.pop(old_index)
        self._children.insert(new_index, item)

    @property
    def matrix(self) -> np.ndarray:
        return np.identity(3, dtype=np.float32)


class BaseLayer(ABC):

    def __init__(
        self,
        parent: NullContainer,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'BaseLayer',
    ):
        self._region = Region.from_size(1, 1)
        self._transform = ComposerRel(self.region.size)
        self._parent_inverse = mat_inverse(parent.matrix)

        self.opacity = opacity
        self.visible = True
        self.blend_mode = blend_mode
        self.name = name

    @property
    def region(self) -> Region:
        if len(self):
            return reduce(or_, [c.region for c in self._children])
        return self._region

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.region.size

    @property
    def transform(self) -> Composer:
        self._transform._region = self.region
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


class GroupLayer(Container, BaseLayer):

    def __init__(
        self,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'BaseLayer',
    ):
        Container.__init__(self)
        BaseLayer.__init__(self, self.parent, opacity, blend_mode, name)

    def __repr__(self):
        return f'GroupLayer(name="{self.name}")'

    def _check_ancestor(self, item: "GroupLayer | Layer"):
        # Verifica ciclos (evita adicionar um pai/avô como filho)
        curr = self.parent
        while curr is not _NULL_CONTAINER:
            if curr == item:
                raise ValueError("Cannot add an ancestor container to a child container")
            curr = curr.parent

    def append(self, item: "GroupLayer | Layer") -> None:
        self._check_ancestor(item)
        super().append(item)

    def insert(self, index: int, item: GroupLayer | Layer) -> None:
        self._check_ancestor(item)
        super().insert(index, item)

    @property
    def matrix(self) -> np.ndarray:
        return self.parent.matrix @ self._parent_inverse @ self.transform.matrix

    def render(
        self,
        renderer: Callable,
        plan_cls: BaseRenderPlan,
        surface: Canvas | Viewport,
        miniview: np.ndarray,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
    ) -> tuple[bool, list[tuple[Any, Any, Any]]]:

        if not self.visible:
            return False, []

        result = False
        images_gp = []

        for container in self:
            if not container.visible:
                continue

            elif isinstance(container, GroupLayer):
                result, images = container.render(
                    renderer, plan_cls, surface, miniview, interp)
                images_gp.extend(images)

                if result:
                    break
            else:
                layer = container
                plan = plan_cls(layer, surface)
                image = renderer(layer, plan, interp=interp)

                if image:
                    images_gp.append((layer, image, plan))

                    if layer._opacity_mask is not None:
                        np.maximum(miniview, layer._opacity_mask, out=miniview)
                    if np.all(miniview == 255):
                        result = True

        if images_gp:

            # Podemos fazer o blend das imagens aqui
            buffer = Image.new(surface.size, ImageFormat.RGBA)
            image = blend_rendered_images(images_gp, buffer)
            group_plan = plan_cls(self, surface)
            return result, [(self, image, group_plan)]

        return False, []
