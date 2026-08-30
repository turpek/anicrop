import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.content import Content, FitContext, GroupContentStrategy
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.spatial import Region


def make_layer(
    w: int = 100,
    h: int = 100,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
    x: int = 0,
    y: int = 0,
    name: str = "Layer",
) -> Layer:
    """Cria uma camada de teste com imagem sólida e região definida."""
    data = np.full((h, w, 4), color, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA), name=name)
    layer.region = Region.from_rect(x, y, w, h)
    return layer


def make_sample_group() -> GroupLayer:
    """Cria um grupo com duas camadas posicionadas lado a lado."""
    group = GroupLayer(name="Group")
    l1 = make_layer(100, 100, (255, 0, 0, 255), x=0, y=0, name="L1")
    l2 = make_layer(100, 100, (0, 255, 0, 255), x=100, y=0, name="L2")
    group.append(l1)
    group.append(l2)
    return group


def test_group_content_instance():
    """Valida se a propriedade content do GroupLayer é uma instância de GroupContentStrategy."""
    group = GroupLayer()
    assert isinstance(group.content, GroupContentStrategy)


def test_group_content_resize():
    """Valida o redimensionamento de um grupo e a propagação de escala para sua região global."""
    group = make_sample_group()
    assert group.global_region.size == (200, 100)

    result = group.content.resize(400, 200)

    assert result is True
    assert group.global_region.size == (400, 200)


def test_group_content_resize_same_size():
    """Valida que o resize para o mesmo tamanho atual retorna False."""
    group = make_sample_group()
    result = group.content.resize(200, 100)
    assert result is False


@pytest.mark.parametrize(
    "w,h",
    [
        (0, 100),
        (100, 0),
        (-50, 100),
        (100, -50),
    ],
    ids=["width_zero", "height_zero", "width_negative", "height_negative"],
)
def test_group_content_resize_invalid_dimensions(w, h):
    """Valida que dimensões menores ou iguais a zero disparam ValueError."""
    group = make_sample_group()
    with pytest.raises(ValueError, match="Dimensões inválidas para resize"):
        group.content.resize(w, h)


def test_group_content_resize_empty_group():
    """Valida que redimensionar um grupo vazio retorna False de forma segura."""
    group = GroupLayer()
    result = group.content.resize(100, 100)
    assert result is False


def test_group_content_nested_empty_group():
    """Valida que operações em grupo contendo apenas subgrupos vazios retornam False."""
    parent_group = GroupLayer(name="Parent")
    child_group = GroupLayer(name="Child")
    parent_group.append(child_group)

    assert parent_group.content.resize(100, 100) is False
    assert parent_group.content.fit(Region.from_rect(0, 0, 50, 50)) is False
    assert parent_group.content.crop(Region.from_rect(0, 0, 50, 50)) is False


def test_group_content_fit():
    """Valida o enquadramento de um grupo dentro de uma região alvo mantendo posição e escala."""
    group = make_sample_group()
    target_region = Region.from_rect(50, 50, 400, 300)

    result = group.content.fit(target_region)

    assert result is True
    assert group.global_region == target_region


def test_group_content_fit_same_region():
    """Valida que o fit para a mesma região global atual retorna False."""
    group = make_sample_group()
    result = group.content.fit(group.global_region)
    assert result is False


def test_group_content_flip_x():
    """Valida o espelhamento horizontal do grupo via inversão da matriz afim."""
    group = make_sample_group()
    result = group.content.flip_x()

    assert result is True
    assert group.matrix[0, 0] < 0


def test_group_content_flip_y():
    """Valida o espelhamento vertical do grupo via inversão da matriz afim."""
    group = make_sample_group()
    result = group.content.flip_y()

    assert result is True
    assert group.matrix[1, 1] < 0


def test_group_content_crop():
    """Valida se o crop no grupo ajusta sua moldura e vincula uma máscara retangular ao grupo."""
    group = make_sample_group()
    crop_box = Region.from_rect(20, 20, 80, 60)

    result = group.content.crop(crop_box)

    assert result is True
    assert group.mask is not None
    assert group.mask.image.format == ImageFormat.GRAY
    assert group.mask.image.size == (80, 60)
    assert group.global_region == crop_box


def test_group_content_facade_operations():
    """Valida a execução de operações de conteúdo em grupo através da fachada Content."""
    group = make_sample_group()
    content = Content()

    assert content.resize(group, 300, 150) is True
    assert group.global_region.size == (300, 150)

    assert content.fit(group, Region.from_rect(10, 10, 100, 50)) is True
    assert group.global_region == Region.from_rect(10, 10, 100, 50)

    assert content.flip_x(group) is True
    assert group.matrix[0, 0] < 0

    assert content.flip_y(group) is True
    assert group.matrix[1, 1] < 0

    assert content.crop(group, Region.from_rect(0, 0, 50, 50)) is True
    assert group.mask is not None


def test_group_content_fit_context():
    """Valida o uso de FitContext com GroupLayer aplicando fit_contain e fit_cover."""
    group = make_sample_group()
    canvas = Canvas(Region.from_size(500, 500))
    content = Content()

    ctx = FitContext(group, canvas)
    result = content.fit(ctx.fit_contain(x_factor=0.5, y_factor=0.5))

    assert result is True
    assert group.global_region.size == (500, 250)
    assert group.global_region.top_left == (0, 125)
