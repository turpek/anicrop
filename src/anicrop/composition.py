from __future__ import annotations
from functools import reduce
from operator import or_
from typing import Sequence
import numpy as np

from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.edit_layer import EditLayer, EDIT_LAYER_MAP
from anicrop.effect import BoundEffect
from anicrop.enums import ImageFormat, InterpMode
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import CanvasRender


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
