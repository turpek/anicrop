import pytest
from anicrop.canvas import Canvas, CanvasLayoutStrategy
from anicrop.spatial import Region, SpanError


def test_canvas_initialization():
    """Garante que o Canvas inicializa com uma Region válida."""
    region = Region.from_size(800, 600)
    canvas = Canvas(region)
    assert canvas.region == region
    assert canvas.size == (800, 600)


@pytest.mark.parametrize(
    "factory_name, dimensions",
    [
        pytest.param("from_size", (800, 600), id="from_size"),
        pytest.param("from_rect", (100, -55, 800, 600), id="from_rect"),
    ],
)
def test_canvas_alternative_constructors(factory_name, dimensions):
    """Garante que construtores alternativos instanciam o Canvas corretamente."""
    canvas = getattr(Canvas, factory_name)(*dimensions)
    region_expect = getattr(Region, factory_name)(*dimensions)
    assert canvas.region == region_expect


@pytest.mark.parametrize(
    "width, height",
    [
        (0, 600),
        (800, 0),
        (-100, 600),
        (800, -100),
    ],
    ids=['size=(0,600)', 'size=(800,0)', 'size=(-100,600)', 'size=(800,-100)']
)
def test_canvas_invalid_dimensions(width, height):
    """Garante que o Canvas valida dimensões maiores que zero."""
    with pytest.raises((ValueError, SpanError)):
        Canvas.from_size(width, height)


def test_canvas_region_setter():
    """Garante que a propriedade region pode ser atualizada através do setter."""
    canvas = Canvas.from_size(800, 600)
    new_region = Region.from_rect(50, 100, 400, 300)

    canvas.region = new_region
    assert canvas.region == new_region


def test_canvas_region_setter_invalid_type():
    """Garante que o setter de region rejeita tipos inválidos com TypeError e mensagem descritiva."""
    canvas = Canvas.from_size(800, 600)
    with pytest.raises(TypeError, match="Expected Region, got int"):
        canvas.region = 123


def test_canvas_region_shift_addition():
    """Garante que a região do Canvas pode ser transladada usando a operação de soma in-place (+=)."""
    canvas = Canvas.from_size(800, 600)
    canvas.region += (50, 100)
    assert canvas.region == Region.from_rect(50, 100, 800, 600)


def test_canvas_layout_bound_api():
    """Valida as operações de layout diretamente via canvas.layout."""
    canvas = Canvas.from_size(800, 600)
    assert isinstance(canvas.layout, CanvasLayoutStrategy)

    assert canvas.layout.fit((100, 100, 400, 300)) is True
    assert canvas.region == Region.from_rect(100, 100, 400, 300)

    target_ref = Region.from_rect(0, 0, 1000, 1000)
    assert canvas.layout.align(target_ref, 0.5, 0.5) is True
    assert canvas.region == Region.from_rect(300, 350, 400, 300)

    assert canvas.layout.resize_bounds(500, 500, 0.5, 0.5) is True
    assert canvas.size == (500, 500)
