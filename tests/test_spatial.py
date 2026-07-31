from anicrop.spatial import Span, SpanError, Region, Vector
from pytest import raises
import pytest
import re


def test_Span_com_start_padrao():
    assert Span(5) == Span(0, 5)


@pytest.mark.parametrize('value', [-5, 0, 5], ids=['-5', '0', '5'])
def test_Span_com_start_variado(value):
    span = Span(value, 5)
    assert span.start == value
    assert span.length == 5

    end_expect = value + 5
    assert span.end == end_expect


@pytest.mark.parametrize('value', [-1, 0], ids=['-1', '0'])
def test_Span_com_valores_de_length_invalidos(value):
    with raises(ValueError, match=re.escape(f'length must be greater than 0 (length={value})')):
        Span(value)


def test_Span_expand_com_valor_negativo():
    expect = (
        "Margin for expand() must be non-negative. To contract "
        "the span, use the shrink() method with a positive value."
    )
    with raises(ValueError) as excinfo:
        Span(0, 10).expand(-1)
    result = str(excinfo.value)
    assert result == expect


@pytest.mark.parametrize('expand', [0, 1, 5], ids=['expand=0', 'expand=1', 'expand=5'])
@pytest.mark.parametrize('start', [-1, 0, 1], ids=['start=-1', 'start=0', 'start=1'])
def test_Span_expand_com_varios_valores_de_expand_e_start(start, expand):
    span = Span(start, 10).expand(expand)
    assert span == Span(start - expand, 10 + 2 * expand)


def test_Span_expand_com_after():
    span = Span(1, 10).expand(after=1)
    assert span == Span(1, 11)


def test_Span_expand_com_before():
    span = Span(1, 10).expand(before=1)
    assert span == Span(0, 11)


def test_Span_expand_com_both():
    span = Span(1, 10).expand(both=1)
    assert span == Span(0, 12)


def test_Span_expand_com_both_e_after():
    span = Span(1, 10).expand(both=1, after=3)
    assert span == Span(0, 12)


def test_Span_shrink_com_valor_negativo():
    expect = ("Margin for shrink() must be non-negative. To contract "
              "To expand the span, use the expand() method with a positive margin.")
    with raises(ValueError) as excinfo:
        Span(0, 2).shrink(-1)
    result = str(excinfo.value)
    assert result == expect


@pytest.mark.parametrize('shrink', [0, 1, 4], ids=['shrink=0', 'shrink=1', 'shrink=4'])
@pytest.mark.parametrize('start', [-1, 0, 1], ids=['start=-1', 'start=0', 'start=1'])
def test_Span_shrink_com_varios_valores_de_both_e_start(shrink, start):
    span = Span(start, 10).shrink(shrink)
    assert span == Span(start + shrink, 10 - shrink * 2)


@pytest.mark.parametrize(
    'before, after, expect',
    [
        (2, 3, (2, 5)),
        (100, 0, (9, 1)),
        (0, 100, (0, 1)),
    ],
    ids=['(2,3)', '(100,0)', '(0,100)']
)
def test_Span_shrink_com_varios_valores_after_e_before(before, after, expect):
    span = Span(0, 10).shrink(before=before, after=after)
    assert span == Span(*expect)


@pytest.mark.parametrize(
    'before, after, expect',
    [
        (5, 5, (-45, 10)),
        (100, 100, (-31, 1)),
    ],
    ids=['(5,5)', '(100,100)']
)
def test_Span_shrink_com_varios_valores_after_e_before_e_start_negativo(before, after, expect):
    span = Span(-50, 20).shrink(before=before, after=after)
    assert span == Span(*expect)


def test_Span_deslocamento_para_direita():
    span = Span(3, 10) + 5
    assert span == Span(8, 10)


def test_Span_deslocamento_para_esquerda():
    span = Span(3, 10) - 2
    assert span == Span(1, 10)


def test_Span_deslocamento_para_esquerda_com_offset_maior_que_start():
    span = Span(3, 10) - 5
    assert span == Span(-2, 10)


def test_Span_deslocamento_para_direita_com_span():
    span = Span(3, 10) + Span(5, 8)
    assert span == Span(8, 10)


def test_Span_deslocamento_para_esquerda_com_span():
    span = Span(3, 10) - Span(2, 8)
    assert span == Span(1, 10)


def test_Span_deslocamento_para_esquerda_com_offset_maior_que_start_com_span():
    span = Span(3, 10) - Span(5, 15)
    assert span == Span(-2, 10)


def test_Span_uniao_de_dois_span():
    span = Span(6, 15) | Span(2, 10)
    assert span == Span(2, 19)


def test_Span_sobreposicao_de_dois_span():
    span = Span(6, 15) & Span(2, 10)
    assert span == Span(6, 6)


def test_Span_sobreposicao_de_dois_span_sem_sobreposicao():
    expect = "no overlap between spans."
    with raises(SpanError) as excinfo:
        Span(6, 9) & Span(15, 15)
    result = str(excinfo.value)
    assert result == expect


def test_Span_overlaps_de_dois_span():
    span = Span(6, 9)
    assert span.overlaps(Span(2, 8))


def test_Span_overlaps_de_dois_span_sem_sobreposicao():
    span = Span(6, 9)
    assert not span.overlaps(Span(15, 15))


def test_Span_offset_to_com_distancia_poisitiva():
    result = Span(2, 8).offset_to(Span(6, 9))
    assert result == 4


def test_Span_offset_to_com_distancia_negativa():
    result = Span(6, 9).offset_to(Span(2, 9))
    assert result == -4


def test_duas_Region_iguais():
    assert Region(Span(0, 10), Span(0, 5)) == Region(Span(0, 10), Span(0, 5))


def test_Region_deslocamento_para_direita_no_eixo_xy():
    region = Region.from_size(10, 5) + (2, 3)
    assert region == Region(Span(2, 10), Span(3, 5))


def test_Region_deslocamento_para_direita_no_eixo_x():
    region = Region.from_size(10, 5) + (2, 0)
    assert region == Region(Span(2, 10), Span(5))


def test_Region_deslocamento_para_direita_no_eixo_y():
    region = Region(Span(10), Span(5)) + Vector(0, 3)
    assert region == Region(Span(10), Span(3, 5))


def test_Region_deslocamento_para_esquerda_no_eixo_xy():
    region = Region(Span(5, 5), Span(10, 5)) - Vector(2, 3)
    assert region == Region(Span(3, 5), Span(7, 5))


def test_Region_deslocamento_para_esquerda_no_eixo_x():
    region = Region(Span(5, 5), Span(10, 5)) - (2, 0)
    assert region == Region(Span(3, 5), Span(10, 5))


def test_Region_deslocamento_para_esquerda_no_eixo_y():
    region = Region(Span(5, 5), Span(10, 5)) - (0, 3)
    assert region == Region(Span(5, 5), Span(7, 5))


def test_Region_deslocamento_para_direita_com_Region():
    region = Region(Span(10), Span(5)) + Region(Span(2, 8), Span(3, 12))
    assert region == Region(Span(2, 10), Span(3, 5))


def test_Region_deslocamento_positivo_nos_eixos_xy_com_int():
    region = Region.from_size(10, 5) + 2
    assert region == Region(Span(2, 10), Span(2, 5))


def test_Region_deslocamento_negativo_nos_eixos_xy_com_int():
    region = Region(Span(5, 5), Span(10, 5)) - 2
    assert region == Region(Span(3, 5), Span(8, 5))


def test_Region_deslocamento_positivo_nos_eixos_xy_com_tupla():
    region = Region.from_size(10, 5) + (2, 3)
    assert region == Region(Span(2, 10), Span(3, 5))


def test_Region_deslocamento_negativo_nos_eixos_xy_com_tupla():
    region = Region(Span(5, 5), Span(10, 5)) - (2, 3)
    assert region == Region(Span(3, 5), Span(7, 5))


def test_Region_deslocamento_positivo_com_tipo_errado():
    expect = r"offset must be an int, a \(x, y\) tuple, or a Vector instance \(got list\)"
    with raises(TypeError, match=expect):
        Region.from_size(10, 5) + [2, 3]


def test_Region_deslocamento_negativo_com_tipo_errado():
    expect = r"offset must be an int, a \(x, y\) tuple, or a Vector instance \(got list\)"
    with raises(TypeError, match=expect):
        Region(Span(2, 10), Span(5, 12)) - [2, 3]


def test_Region_uniao_de_duas_regioes():
    region = Region(Span(5, 5), Span(10, 5)) | Region(Span(7, 13), Span(12, 13))
    assert region == Region(Span(5, 15), Span(10, 15))


def test_Region_sobreposicao_de_duas_regioes():
    region = Region(Span(5, 5), Span(10, 5)) & Region(Span(7, 13), Span(12, 13))
    assert region == Region(Span(7, 3), Span(12, 3))


def test_Region_overlaps_no_eixo_xy():
    region = Region(Span(5, 5), Span(10, 5))
    assert region.overlaps(Region(Span(7, 13), Span(12, 13)))


def test_Region_overlaps_no_eixo_x():
    region = Region(Span(5, 5), Span(10, 5))
    assert not region.overlaps(Region(Span(15, 5), Span(12, 13)))


def test_Region_overlaps_no_eixo_y():
    region = Region(Span(5, 5), Span(10, 5))
    assert not region.overlaps(Region(Span(15, 5), Span(22, 3)))


def test_Region_area():
    region = Region(Span(5, 5), Span(10, 5))
    assert region.area == 25


def test_Region_width():
    region = Region(Span(5, 5), Span(10, 5))
    assert region.width == 5


def test_Region_height():
    region = Region(Span(5, 5), Span(10, 5))
    assert region.height == 5


def test_Region_expand_em_no_eixo_xy():
    region = Region(Span(5, 5), Span(10, 5)).expand(Vector(5, 5))
    assert region == Region(Span(0, 15), Span(5, 15))


def test_Region_expand_em_no_eixo_x():
    region = Region(Span(5, 5), Span(10, 5)).expand(Vector(5, 0))
    assert region == Region(Span(0, 15), Span(10, 5))


def test_Region_expand_em_no_eixo_y():
    region = Region(Span(5, 5), Span(10, 5)).expand(Vector(0, 5))
    assert region == Region(Span(5, 5), Span(5, 15))


def test_Region_expand_left_e_top():
    region = Region(Span(5, 5), Span(10, 5)).expand(left=5, top=5)
    assert region == Region(Span(0, 10), Span(5, 10))


def test_Region_expand_right_e_bottom():
    region = Region(Span(5, 5), Span(10, 5)).expand(right=5, bottom=5)
    assert region == Region(Span(5, 10), Span(10, 10))


def test_Region_shrink_em_no_eixo_xy():
    region = Region(Span(0, 15), Span(10, 15)).shrink(Vector(5, 5))
    assert region == Region(Span(5, 5), Span(15, 5))


def test_Region_shrink_em_no_eixo_x():
    region = Region(Span(0, 15), Span(10, 15)).shrink(Vector(5, 0))
    assert region == Region(Span(5, 5), Span(10, 15))


def test_Region_shrink_em_no_eixo_y():
    region = Region(Span(0, 15), Span(10, 15)).shrink(Vector(0, 5))
    assert region == Region(Span(0, 15), Span(15, 5))


def test_Region_shrink_left_e_top():
    region = Region(Span(0, 15), Span(10, 15)).shrink(left=5, top=5)
    assert region == Region(Span(5, 10), Span(15, 10))


def test_Region_shrink_right_e_bottom():
    region = Region(Span(0, 15), Span(10, 15)).shrink(right=5, bottom=5)
    assert region == Region(Span(0, 10), Span(10, 10))


def test_Region_offset_positivo():
    region = Region(Span(50, 110), Span(55, 135))
    result = Region(Span(0, 15), Span(10, 15)).offset_to(region)
    assert result == Vector(50, 45)


def test_Region_offset_negativo():
    region = Region(Span(0, 15), Span(10, 15))
    result = Region(Span(50, 110), Span(55, 135)).offset_to(region)
    assert result == Vector(-50, -45)


def test_Region_overlap_with_regionB_em_regionA():
    regionA = Region(Span(0, 1920), Span(437, 1080))
    regionB = Region(Span(236, 1970), Span(0, 1080))
    result = regionA.overlap_with(regionB)
    assert result == Region(Span(236, 1684), Span(0, 643))


def test_Region_overlap_with_regionA_em_regionB():
    regionA = Region(Span(0, 1920), Span(437, 1080))
    regionB = Region(Span(236, 1970), Span(0, 1080))
    result = regionB.overlap_with(regionA)
    assert result == Region(Span(0, 1684), Span(437, 643))


def test_Region_overlap_with_regionB_nao_sobreposto_regionA():
    with raises(ValueError, match="no overlap: 'other' out of bounds"):
        regionA = Region(Span(0, 1920), Span(437, 1080))
        regionB = Region(Span(2206, 1639), Span(1880, 1220))
        regionB.overlap_with(regionA)


def test_Span_offset_to_anchor_end():
    span1 = Span(-4, 58)  # end = 54
    span2 = Span(0, 54)    # end = 54
    assert span1.offset_to(span2, anchor_end=True) == 0

    span3 = Span(0, 50)    # end = 50
    assert span1.offset_to(span3, anchor_end=True) == -4


def test_Region_offset_to_anchor_end():
    region1 = Region(Span(-4, 58), Span(-4, 58))  # end = (54, 54)
    region2 = Region(Span(0, 50), Span(0, 50))    # end = (50, 50)
    result = region1.offset_to(region2, anchor_end=True)
    assert result == Vector(-4, -4)


def test_Span_slack():
    target = Span(10, 40)
    reference = Span(100, 200)

    # slack é o espaço extra que a referência tem em relação ao target
    assert target.slack(reference) == 160

    # se invertermos, o slack é negativo (a referência é menor)
    assert reference.slack(target) == -160


@pytest.mark.parametrize('factor, expect_start', [
    (0.0, 100),   # align left: ref.start + 160 * 0.0
    (0.5, 180),   # align center: ref.start + 160 * 0.5
    (1.0, 260),   # align right: ref.start + 160 * 1.0
])
def test_Span_align(factor, expect_start):
    target = Span(10, 40)
    reference = Span(100, 200)

    novo_span = target.align(reference, factor)

    # O método deve retornar um NOVO Span com o mesmo tamanho da origem,
    # mas com o start deslocado para o alinhamento.
    assert novo_span == Span(expect_start, target.length)


def test_Region_align():
    target = Region(Span(10, 40), Span(10, 30))
    reference = Region(Span(100, 200), Span(200, 300))

    # Alinhando no centro (0.5) em ambos os eixos
    aligned_region = target.align(reference, 0.5, 0.5)

    # Verifica se os tamanhos se mantiveram
    assert aligned_region.x.length == 40
    assert aligned_region.y.length == 30

    # Verifica a nova posição (os mesmos 180 e 335 do nosso exemplo!)
    assert aligned_region.x.start == 180
    assert aligned_region.y.start == 335
