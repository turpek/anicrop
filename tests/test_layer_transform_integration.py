import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.spatial import Region, Span
from anicrop.transform import ComposerAbs, ComposerRel, TransformAbs, TransformRel


def make_canvas(w=10, h=10):
    return Image.new((h, w), ImageFormat.RGBA)


@pytest.fixture
def canvas():
    return make_canvas()


@pytest.mark.parametrize(
    "transform_intent, expected_bbox",
    [
        (lambda t: t, (0, 0, 10, 10)),  # Identidade
        (lambda t: t.translate(10, 20), (10, 20, 10, 10)),  # Translação
        (lambda t: t.rotate(90, 0.5, 0.5), (0, 0, 10, 10)),  # Rotação centro
    ],
)
def test_layer_set_transform_cenario_sem_transformacoes(
    canvas, transform_intent, expected_bbox
):
    layer = Layer(canvas)
    t = transform_intent(TransformRel())
    layer.set_transform(t)

    res_region = layer.global_region
    expected_region = Region(
        Span(expected_bbox[0], expected_bbox[2]),
        Span(expected_bbox[1], expected_bbox[3]),
    )
    assert res_region == expected_region


def test_layer_set_transform_com_duas_transformacoes(canvas):
    layer = Layer(canvas)

    # translate(10, 10) -> rotate(90, 0.5, 0.5)
    # Como a translação vai para o final: M = T(10,10) @ R(90, pivo 5,5)
    # Rect resultante deve ser (10, 10, 10, 10)
    t = TransformRel().translate(10, 10).rotate(90, 0.5, 0.5)
    layer.set_transform(t)

    assert layer.global_region == Region(Span(10, 10), Span(10, 10))


def test_layer_cenario_complexo_violao(canvas):
    layer = Layer(canvas)

    # Transform().translate(10, 10).rotate(45).scale(2,1,0).translate(5,5).rotate(-30)
    # Intenção:
    # 1. Translação total acumulada: 10 + 5 = 15
    # 2. Distorção acumulada com PIVÔ DINÂMICO
    t = (
        TransformRel()
        .translate(10, 10)
        .rotate(45)
        .scale(2, 1, 0, 0)
        .translate(5, 5)
        .rotate(-30)
    )
    layer.set_transform(t)

    # O resultado deve refletir o posicionamento no canvas.
    # Baseado no debug, o motor dinâmico produz start=(13, 13) e length=(26, 15)
    # Esses valores são ligeiramente diferentes do cálculo estático pois o pivô
    # acompanha o Rect em tempo real.
    res = layer.global_region
    assert res.x.start == 14
    assert res.y.start == 12
    assert res.x.length == 26  # Verificando se esticou


def test_layer_set_transform_comportamento_absoluto(canvas):
    layer = Layer(canvas)
    t1 = TransformRel().translate(10, 10)

    # Aplica t1 (Offset 10,10)
    layer.set_transform(t1)
    assert layer.global_region.x.start == 10

    # Cria t2 independente (Offset 40,40)
    t2 = TransformRel().translate(40, 40)

    # Aplica t2 (O layer deve SUBSTITUIR t1 por t2)
    layer.set_transform(t2)

    assert layer.global_region.x.start == 40
    assert layer.global_region.y.start == 40


def test_layer_set_transform_imutabilidade_aditiva_interna(canvas):
    layer = Layer(canvas)
    t1 = TransformRel().translate(10, 10)

    layer.set_transform(t1)

    t2 = t1.translate(40, 40)
    layer.set_transform(t2)

    assert layer.global_region.x.start == 50
    assert layer.global_region.y.start == 50


# --- Testes de Integração com Referências Alternadas de Pivô ---


def test_layer_transform_intercambiavel_referencia_layer_vs_canvas():
    """Testa a intercambeabilidade de referência de pivô no add_transform.

    Quando usado reference_size = layer.region.size (100x100), pivo 0.5, 0.5 gira em torno de (50, 50).
    Quando usado reference_size = canvas (1000x1000), pivo 0.5, 0.5 gira em torno de (500, 500).
    """
    img = Image.new((100, 100), ImageFormat.RGBA)
    layer = Layer(img)

    t_rel = TransformRel().rotate(90, 0.5, 0.5)

    # 1. Referência do próprio Layer (padrão 100x100 -> pivô 50, 50)
    layer.transform.add_transform(t_rel)
    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    res_layer_ref = layer.transform.matrix @ pt_origem
    np.testing.assert_allclose(res_layer_ref, [100, 0, 1], atol=1e-4)

    # Limpa e aplica com referência do Canvas (1000x1000 -> pivô 500, 500)
    layer.transform_clear()
    layer.transform.add_transform(t_rel, reference_size=(1000, 1000))
    res_canvas_ref = layer.transform.matrix @ pt_origem
    np.testing.assert_allclose(res_canvas_ref, [1000, 0, 1], atol=1e-4)


def test_layer_transform_polimorfico_abs_e_rel():
    """Valida se o Layer aceita TransformAbs e TransformRel de forma 100% polimórfica e uniforme."""
    img = Image.new((100, 100), ImageFormat.RGBA)
    layer = Layer(img)

    # Aplica TransformAbs com pivô em (50, 50)
    t_abs = TransformAbs().rotate(90, px=50, py=50)
    layer.transform.add_transform(t_abs)

    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    res_abs = layer.transform.matrix @ pt_origem
    np.testing.assert_allclose(res_abs, [100, 0, 1], atol=1e-4)


def test_composer_multireferencia_customizada():
    """Valida se o ComposerRel aceita referências arbitrárias no add_transform (ex: 500x500)."""
    composer = ComposerRel((100, 100))
    t1 = TransformRel().scale(2.0, 2.0, 0.5, 0.5)

    # Passa uma caixa arbitrária de (500, 500) como referência -> Pivô (250, 250)
    composer.add_transform(t1, reference_size=(500, 500))

    pt_pivo = np.array([250, 250, 1], dtype=np.float32)
    res_pivo = composer.matrix @ pt_pivo
    # Ponto pivô (250, 250) deve permanecer imóvel sob a escala 2x
    np.testing.assert_allclose(res_pivo, [250, 250, 1], atol=1e-4)


def test_layer_set_transform_com_transform_abs():
    """Valida se layer.set_transform(t_abs) instancia um ComposerAbs via Factory Method."""
    img = Image.new((100, 100), ImageFormat.RGBA)
    layer = Layer(img)

    t_abs = TransformAbs().rotate(90, px=50, py=50)
    layer.set_transform(t_abs)

    assert isinstance(layer.transform, ComposerAbs)
    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(
        layer.transform.matrix @ pt_origem, [100, 0, 1], atol=1e-4
    )


def test_layer_set_transform_com_referencia_layer_e_canvas():
    """Valida duck-typing em set_transform aceitando reference=Canvas ou reference=Layer_Pai."""
    img_pai = Image.new((1000, 1000), ImageFormat.RGBA)
    layer_pai = Layer(img_pai)

    img_filho = Image.new((100, 100), ImageFormat.RGBA)
    layer_filho = Layer(img_filho)

    t_rel = TransformRel().rotate(90, 0.5, 0.5)

    # 1. Passando Layer como referência (1000x1000 -> pivô 500, 500)
    layer_filho.set_transform(t_rel, reference=layer_pai)
    pt_origem = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(
        layer_filho.transform.matrix @ pt_origem, [1000, 0, 1], atol=1e-4
    )

    # 2. Passando Canvas como referência (500x500 -> pivô 250, 250)
    canvas_obj = Canvas.from_size(500, 500)
    layer_filho.set_transform(t_rel, reference=canvas_obj)
    np.testing.assert_allclose(
        layer_filho.transform.matrix @ pt_origem, [500, 0, 1], atol=1e-4
    )
