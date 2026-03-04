import pytest
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.transform import Transform, mat_translation
from anicrop.transform import TRotate, TScale
from anicrop.transform import calculate_new_bbox
from anicrop.spatial import Region, Span


def make_canvas(w=10, h=10):
    return Image.new((h, w), ImageFormat.RGBA)


@pytest.fixture
def canvas():
    return make_canvas()


@pytest.mark.parametrize("transform_intent, expected_bbox", [
    (lambda t: t, (0, 0, 10, 10)),  # Identidade
    (lambda t: t.translate(10, 20), (10, 20, 10, 10)),  # Translação
    (lambda t: t.rotate(90, 0.5, 0.5), (0, 0, 10, 10)),  # Rotação centro
])
def test_layer_set_transform_cenario_sem_transformacoes(canvas, transform_intent, expected_bbox):
    layer = Layer(canvas)
    t = transform_intent(Transform())
    layer.set_transform(t)

    res_region = layer.canvas_region
    expected_region = Region(
        Span(expected_bbox[0], expected_bbox[2]),
        Span(expected_bbox[1], expected_bbox[3])
    )
    assert res_region == expected_region


def test_layer_set_transform_com_duas_transformacoes(canvas):
    layer = Layer(canvas)

    # translate(10, 10) -> rotate(90, 0.5, 0.5)
    # Como a translação vai para o final: M = T(10,10) @ R(90, pivo 5,5)
    # BBox resultante deve ser (10, 10, 10, 10)
    t = Transform().translate(10, 10).rotate(90, 0.5, 0.5)
    layer.set_transform(t)

    assert layer.canvas_region == Region(Span(10, 10), Span(10, 10))


def test_layer_cenario_complexo_violao(canvas):
    layer = Layer(canvas)

    # Transform().translate(10, 10).rotate(45).scale(2,1,0).translate(5,5).rotate(-30)
    # Intenção:
    # 1. Translação total acumulada: 10 + 5 = 15
    # 2. Distorção acumulada: R(-30) @ S(2, 1, pivo 0) @ R(45, pivo 0.5)
    t = Transform().translate(10, 10).rotate(45).scale(
        2, 1, 0, 0).translate(5, 5).rotate(-30)
    layer.set_transform(t)

    # Calculando a matriz manualmente para validar o BBox
    # M_dist = R(-30, pivo 0.5) @ S(2, 1, pivo 0) @ R(45, pivo 0.5)
    # M_final = T(15, 15) @ M_dist
    m_dist = TRotate(-30).matrix((10, 10)) @ TScale(2, 1, 0,
                                                    0).matrix((10, 10)) @ TRotate(45).matrix((10, 10))
    m_expected = mat_translation(15, 15) @ m_dist

    x, y, w, h = calculate_new_bbox(m_expected, (10, 10))

    assert layer.canvas_region == Region(Span(x, w), Span(y, h))


def test_layer_set_transform_imutabilidade_aditiva(canvas):
    layer = Layer(canvas)
    t1 = Transform().translate(10, 10)

    # Aplica t1 (Layer agora tem offset 10,10)
    layer.set_transform(t1)

    # Cria t2 a partir de t1 (t2 tem as intenções: [trans 10,10, trans 40,40])
    t2 = t1.translate(40, 40)

    # Aplica t2 (O layer ADICIONA as intenções de t2 ao que já tinha)
    # Offset antigo (10) + Offset de t2 (50) = 60
    layer.set_transform(t2)

    assert layer.canvas_region.x.start == 60
    assert layer.canvas_region.y.start == 60
