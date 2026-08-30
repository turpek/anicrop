from __future__ import annotations
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Sequence
import numpy as np

from anicrop.container import BaseLayer, Container, GroupLayer, LayerStack, _NULL_CONTAINER
from anicrop.edit_layer import EditLayer, EDIT_LAYER_MAP
from anicrop.effect import BoundEffect
from anicrop.enums import ImageFormat, InterpMode
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import CanvasRender

if TYPE_CHECKING:
    from anicrop.document import Document


def clone_layer(layer: Layer) -> Layer:
    """Creates a fully isolated and decoupled copy of a layer (zero-copy of heavy pixel buffers)."""
    cloned = Layer(
        layer.base.region,
        opacity=layer.opacity,
        blend_mode=layer.blend_mode,
        name=layer.name,
        format=layer.format,
    )
    cloned.visible = layer.visible
    cloned.transform.copy_from(layer.transform)

    # 1. Clone each EditLayer with new affine 3x3 matrix
    for edit in layer.edits:
        edit_cls = EDIT_LAYER_MAP.get(edit.blend_mode, EditLayer)
        cloned._edits.append(edit_cls(
            image=edit.image,
            region=edit.region,
            matrix=edit.matrix.copy(),
            blend_mode=edit.blend_mode,
            name=edit.name,
            visible=edit.visible,
        ))

    # 2. Clone mask with isolated NumPy buffer
    if layer.mask is not None:
        mask_copy = Image(layer.mask.image[...].copy(), layer.mask.image.format)
        cloned.set_mask(
            image=mask_copy,
            region=layer.mask.region,
            invert=layer.mask.invert,
            visible=layer.mask.visible,
            name=layer.mask.name,
        )

    # 3. Clone effects with new matrices
    for effect in layer.effects:
        if isinstance(effect, BoundEffect):
            cloned_mask = None
            if effect.mask is not None:
                cloned_mask = Image(effect.mask.image[...].copy(), effect.mask.image.format)
            bound = BoundEffect(
                effect.effect,
                matrix=effect.matrix.copy(),
                mask=cloned_mask,  # type: ignore[arg-type]
                visible=effect.visible,
            )
            cloned.add_effect(bound)
        else:
            cloned.add_effect(effect)

    # 4. Preserve active Layout frame if fitted
    if layer.control.frame.region != layer.base.region:
        cloned.layout.fit(layer.control.frame.region)

    return cloned


def clone_group(group: GroupLayer) -> GroupLayer:
    """Creates a deep, decoupled copy of a GroupLayer and its children."""
    cloned = GroupLayer(
        opacity=group.opacity,
        blend_mode=group.blend_mode,
        name=group.name,
        format=group.format,
    )
    cloned.visible = group.visible
    cloned.transform.copy_from(group.transform)

    for child in group:
        if isinstance(child, Layer):
            cloned.append(clone_layer(child))
        elif isinstance(child, GroupLayer):
            cloned.append(clone_group(child))

    if group.mask is not None:
        mask_copy = Image(group.mask.image[...].copy(), group.mask.image.format)
        cloned.set_mask(
            image=mask_copy,
            region=group.mask.region,
            invert=group.mask.invert,
            visible=group.mask.visible,
            name=group.mask.name,
        )

    for effect in group.effects:
        if isinstance(effect, BoundEffect):
            cloned_mask = None
            if effect.mask is not None:
                cloned_mask = Image(effect.mask.image[...].copy(), effect.mask.image.format)
            bound = BoundEffect(
                effect.effect,
                matrix=effect.matrix.copy(),
                mask=cloned_mask,  # type: ignore[arg-type]
                visible=effect.visible,
            )
            cloned.add_effect(bound)
        else:
            cloned.add_effect(effect)

    if group.control.frame.region != group.base.region:
        cloned.layout.fit(group.control.frame.region)

    return cloned


def clone_node(node: BaseLayer) -> BaseLayer:
    """Polymorphically clones any graphical node (Layer or GroupLayer)."""
    if isinstance(node, Layer):
        return clone_layer(node)
    elif isinstance(node, GroupLayer):
        return clone_group(node)
    raise TypeError(f"Unsupported node type for cloning: {type(node).__name__}")


def merge(
    layers: Sequence[BaseLayer] | Container,
    name: str = "Group",
) -> GroupLayer:
    """Creates a non-destructive grouped composition (GroupLayer) containing decoupled
    copies of the given layers.
    """
    if not layers:
        raise ValueError("No layers provided for merge.")

    group = GroupLayer(name=name)

    for item in layers:
        group.append(clone_node(item))

    return group


def flatten(
    layers: Sequence[BaseLayer] | Container,
    name: str = "Layer",
    format: ImageFormat = ImageFormat.RGBA,
    interp: InterpMode = InterpMode.LANCZOS,
    bg_color: tuple[int, ...] | None = None,
) -> Layer:
    """Faithfully renders the layer composition and returns a single rasterized flat Layer."""
    if not layers:
        raise ValueError("No layers provided for flatten.")

    renderable_nodes = [item for item in layers if item.is_renderable]
    if not renderable_nodes:
        raise ValueError("No renderable layers found for flatten.")

    roi = reduce(or_, (item.global_region for item in renderable_nodes))

    rendered_image = CanvasRender().render_container(
        layers,
        format=format,
        interp=interp,
        bg_color=bg_color,
    )
    if rendered_image is None:
        raise ValueError("Failed to render layers for flatten.")

    flat_layer = Layer(rendered_image, name=name)
    flat_layer.region = roi
    return flat_layer


class LayerComposition:
    """Pure operations for layer blending, grouping, and composition."""

    @staticmethod
    def merge(
        layers: Sequence[BaseLayer] | Container,
        name: str = "Group",
    ) -> GroupLayer:
        return merge(layers, name=name)

    @staticmethod
    def flatten(
        layers: Sequence[BaseLayer] | Container,
        name: str = "Layer",
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
        bg_color: tuple[int, ...] | None = None,
    ) -> Layer:
        return flatten(layers, name=name, format=format, interp=interp, bg_color=bg_color)

    @staticmethod
    def clone(node: BaseLayer) -> BaseLayer:
        return clone_node(node)


class Combine:
    """Stateful service attached to a Document for merging/flattening layers in the document tree."""

    def __init__(self, doc: Document) -> None:
        self._doc = doc

    def _resolve_target_and_sequence(
        self,
        target: BaseLayer | str,
        count: int,
    ) -> tuple[Container, list[BaseLayer], int]:
        if count < 1:
            raise ValueError("count must be at least 1.")

        if isinstance(target, str):
            resolved_target = self._doc[target]
        elif isinstance(target, BaseLayer):
            resolved_target = target
        else:
            raise TypeError(f"Invalid target type {type(target).__name__}. Expected BaseLayer or str.")

        parent = resolved_target.parent
        if parent is _NULL_CONTAINER or not isinstance(parent, Container):
            raise ValueError(f"Layer '{resolved_target.name}' does not belong to a valid container in the document.")

        if resolved_target not in parent._children:
            raise ValueError(f"Layer '{resolved_target.name}' is not in its parent container.")

        target_idx = parent._children.index(resolved_target)

        # Collect up to `count` visible sibling layers downwards
        collected: list[BaseLayer] = []
        for i in range(target_idx - 1, -1, -1):
            sibling = parent._children[i]
            if sibling.visible:
                collected.append(sibling)
                if len(collected) == count:
                    break

        if not collected:
            raise ValueError(f"No visible layers found below '{resolved_target.name}' in the same container to merge.")

        # Order from bottom to top for composition
        sequence: list[BaseLayer] = list(reversed(collected)) + [resolved_target]
        lowest_index = parent._children.index(sequence[0])

        return parent, sequence, lowest_index

    def _validate_name(self, name: str, sequence: list[BaseLayer]) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("A valid non-empty string name is required.")
        found = self._doc.find(name, recursive=True)
        if found is not None and found not in sequence:
            raise ValueError(f"A layer named '{name}' already exists in the document.")

    def merge(
        self,
        target: BaseLayer | str,
        name: str,
        count: int = 1,
        remove_source: bool = True,
    ) -> GroupLayer:
        """Merges the target layer with up to 'count' visible layers below it into a GroupLayer."""
        parent, sequence, lowest_index = self._resolve_target_and_sequence(target, count)
        self._validate_name(name, sequence if remove_source else [])

        merged_group = merge(sequence, name=name)

        if remove_source:
            for layer in sequence:
                parent.remove(layer)
            parent.insert(lowest_index, merged_group)

        return merged_group

    def flatten(
        self,
        target: BaseLayer | str,
        name: str,
        count: int = 1,
        format: ImageFormat | None = None,
        interp: InterpMode = InterpMode.LANCZOS,
        bg_color: tuple[int, ...] | None = None,
        remove_source: bool = True,
    ) -> Layer:
        """Flattens the target layer with up to 'count' visible layers below it into a single rasterized Layer."""
        parent, sequence, lowest_index = self._resolve_target_and_sequence(target, count)
        self._validate_name(name, sequence if remove_source else [])

        resolved_format = format if format is not None else sequence[-1].format

        flat_layer = flatten(
            sequence,
            name=name,
            format=resolved_format,
            interp=interp,
            bg_color=bg_color,
        )

        if remove_source:
            for layer in sequence:
                parent.remove(layer)
            parent.insert(lowest_index, flat_layer)

        return flat_layer

    def bake(
        self,
        target: GroupLayer | str,
        name: str | None = None,
        format: ImageFormat | None = None,
        interp: InterpMode = InterpMode.LANCZOS,
        bg_color: tuple[int, ...] | None = None,
    ) -> Layer:
        """Bakes a GroupLayer and replaces it with a flat Layer in its parent container."""
        group = self._doc[target] if isinstance(target, str) else target
        if not isinstance(group, GroupLayer):
            raise TypeError(f"Expected GroupLayer, got {type(group).__name__}.")

        parent = group.parent
        if parent is _NULL_CONTAINER or not isinstance(parent, Container):
            raise ValueError(f"Group '{group.name}' is not attached to a valid container.")

        resolved_name = name or group.name
        resolved_format = format or group.format

        self._validate_name(resolved_name, [group])

        flat_layer = flatten(group, name=resolved_name, format=resolved_format, interp=interp, bg_color=bg_color)

        idx = parent._children.index(group)
        parent.remove(group)
        parent.insert(idx, flat_layer)

        return flat_layer

    def bake_stack(
        self,
        name: str = "Layer",
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
        bg_color: tuple[int, ...] | None = None,
    ) -> Layer:
        """Bakes the entire document's LayerStack into a single flat Layer."""
        self._validate_name(name, list(self._doc.stack))

        flat_layer = flatten(self._doc.stack, name=name, format=format, interp=interp, bg_color=bg_color)

        self._doc.stack.clear()
        self._doc.stack.append(flat_layer)

        return flat_layer
