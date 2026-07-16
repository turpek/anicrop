import pytest
from anicrop.canvas import Canvas


def test_canvas_initialization():
    """Garante que o Canvas inicializa com largura e altura válidas."""
    canvas = Canvas(800, 600)
    assert canvas.size == (800, 600)


def test_canvas_invalid_dimensions():
    """Garante que o Canvas valida dimensões maiores que zero."""
    with pytest.raises(ValueError):
        Canvas(0, 600)
    with pytest.raises(ValueError):
        Canvas(800, 0)
    with pytest.raises(ValueError):
        Canvas(-100, 600)
