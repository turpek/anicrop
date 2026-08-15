from anicrop.blend import BLEND_MODE
from anicrop.enums import BlendMode, InterpolationOption as IO
from anicrop.layer import Layer
from anicrop.image import Image
from anicrop.render import CanvasRender


def merge_down(layer_up: Layer, layer_down: Layer) -> Layer:
    region_up, region_down = layer_up.global_region, layer_down.global_region
    canvas_region = region_up | region_down
    view_up = canvas_region.overlap_with(region_up)
    view_down = canvas_region.overlap_with(region_down)

    canvas = Image.new(canvas_region.size, layer_up.format)

    render = CanvasRender()
    image_up = render.render_layer(layer_up)
    image_down = render.render_layer(layer_down)

    blend = BLEND_MODE.get(BlendMode.HARD_MASKING)
    blend(canvas.view(view_down), image_down, layer_down.opacity)

    blend = BLEND_MODE.get(layer_up.blend_mode)
    blend(canvas.view(view_up), image_up, layer_up.opacity)

    layer = Layer(canvas, blend_mode=layer_down.blend_mode)
    layer.region = canvas_region
    return layer
