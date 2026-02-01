from anicrop.spatial import Span, SpanError, Region, Vector
from pytest import raises


def test_Span_instanciacao_com_um_parametro():
    assert Span(5) == Span(0, 5)


def test_Span_com_start_igual_a_end():
    expect = "start must be < end (start=0, end=0)"
    with raises(ValueError) as excinfo:
        Span(0, 0)
    result = str(excinfo.value)
    assert result == expect


def test_Span_com_start_maior_que_end():
    expect = "start must be < end (start=1, end=0)"
    with raises(ValueError) as excinfo:
        Span(1, 0)
    result = str(excinfo.value)
    assert result == expect


def test_Span_com_start_negativo():
    expect = 'start cannot be less than 0 (start=-1)'
    with raises(ValueError) as excinfo:
        Span(-1, -10)
    result = str(excinfo.value)
    assert result == expect


def test_Span_com_end_negativo():
    expect = "start must be < end (start=0, end=-1)"
    with raises(ValueError) as excinfo:
        Span(0, -1)
    result = str(excinfo.value)
    assert result == expect


def test_Span_length_com_valores_positivos():
    span = Span(0, 1)
    result = span.length
    assert result == 1


def test_Span_expand_com_valor_negativo():
    expect = (
        "Margin for expand() must be non-negative. To contract "
        "the span, use the shrink() method with a positive margin."
    )
    with raises(ValueError) as excinfo:
        Span(0, 10).expand(-1)
    result = str(excinfo.value)
    assert result == expect


def test_Span_expand_com_zero():
    span = Span(0, 10).expand(0)
    assert span == Span(0, 10)


def test_Span_expand_com_start_zero():
    span = Span(0, 10).expand(1)
    assert span == Span(0, 11)


def test_Span_expand_com_start_maior_que_zero():
    span = Span(1, 10).expand(1)
    assert span == Span(0, 11)


def test_Span_shrink_com_valor_negativo():
    expect = ("Margin for shrink() must be non-negative, but got -1. "
              "To expand the span, use the expand() method with a positive margin.")
    with raises(ValueError) as excinfo:
        Span(0, 2).shrink(-1)
    result = str(excinfo.value)
    assert result == expect


def test_Span_shrink_com_valor_zero():
    span = Span(0, 10).shrink(0)
    assert span == Span(0, 10)


def test_Span_shrink_com_valor_positivo():
    span = Span(0, 10).shrink(1)
    assert span == Span(1, 9)


def test_Span_shrink_gerando_start_igual_a_end():
    expect = "start must be < end (start=1, end=1)"
    with raises(ValueError) as excinfo:
        Span(0, 2).shrink(1)
    result = str(excinfo.value)
    assert result == expect


def test_Span_shrink_gerando_start_maior_que_end():
    expect = "start must be < end (start=1, end=0)"
    with raises(ValueError) as excinfo:
        Span(0, 1).shrink(1)
    result = str(excinfo.value)
    assert result == expect


def test_Span_deslocamento_para_direita():
    span = Span(3, 10) + 5
    assert span == Span(8, 15)


def test_Span_deslocamento_para_esquerda():
    span = Span(3, 10) - 2
    assert span == Span(1, 8)


def test_Span_deslocamento_para_esquerda_com_offset_maior_que_start():
    span = Span(3, 10) - 5
    assert span == Span(0, 5)


def test_Span_uniao_de_dois_span():
    span = Span(6, 15) | Span(2, 10)
    assert span == Span(2, 15)


def test_Span_sobreposicao_de_dois_span():
    span = Span(6, 15) & Span(2, 10)
    assert span == Span(6, 10)


def test_Span_sobreposicao_de_dois_span_sem_sobreposicao():
    expect = "no overlap between spans."
    with raises(SpanError) as excinfo:
        Span(6, 15) & Span(15, 30)
    result = str(excinfo.value)
    assert result == expect


def test_Span_overlaps_de_dois_span():
    span = Span(6, 15)
    assert span.overlaps(Span(2, 10))


def test_Span_overlaps_de_dois_span_sem_sobreposicao():
    span = Span(6, 15)
    assert not span.overlaps(Span(15, 30))


def test_Span_offset_to_com_distancia_poisitiva():
    result = Span(2, 10).offset_to(Span(6, 15))
    assert result == 4


def test_Span_offset_to_com_distancia_negativa():
    result = Span(6, 15).offset_to(Span(2, 10))
    assert result == -4


def test_duas_Region_iguais():
    assert Region(Span(0, 10), Span(0, 5)) == Region(Span(0, 10), Span(0, 5))


def test_Region_deslocamento_para_direita_no_eixo_xy():
    region = Region(Span(0, 10), Span(0, 5)) + Vector(2, 3)
    assert region == Region(Span(2, 12), Span(3, 8))


def test_Region_deslocamento_para_direita_no_eixo_x():
    region = Region(Span(0, 10), Span(0, 5)) + Vector(2, 0)
    assert region == Region(Span(2, 12), Span(0, 5))


def test_Region_deslocamento_para_direita_no_eixo_y():
    region = Region(Span(0, 10), Span(0, 5)) + Vector(0, 3)
    assert region == Region(Span(0, 10), Span(3, 8))


def test_Region_deslocamento_para_esquerda_no_eixo_xy():
    region = Region(Span(5, 10), Span(10, 15)) - Vector(2, 3)
    assert region == Region(Span(3, 8), Span(7, 12))


def test_Region_deslocamento_para_esquerda_no_eixo_x():
    region = Region(Span(5, 10), Span(10, 15)) - Vector(2, 0)
    assert region == Region(Span(3, 8), Span(10, 15))


def test_Region_deslocamento_para_esquerda_no_eixo_y():
    region = Region(Span(5, 10), Span(10, 15)) - Vector(0, 3)
    assert region == Region(Span(5, 10), Span(7, 12))


def test_Region_uniao_de_duas_regioes():
    region = Region(Span(5, 10), Span(10, 15)) | Region(Span(7, 20), Span(12, 25))
    assert region == Region(Span(5, 20), Span(10, 25))


def test_Region_sobreposicao_de_duas_regioes():
    region = Region(Span(5, 10), Span(10, 15)) & Region(Span(7, 20), Span(12, 25))
    assert region == Region(Span(7, 10), Span(12, 15))


def test_Region_overlaps_no_eixo_xy():
    region = Region(Span(5, 10), Span(10, 15))
    assert region.overlaps(Region(Span(7, 20), Span(12, 25)))


def test_Region_overlaps_no_eixo_x():
    region = Region(Span(5, 10), Span(10, 15))
    assert not region.overlaps(Region(Span(15, 20), Span(12, 25)))


def test_Region_overlaps_no_eixo_y():
    region = Region(Span(5, 10), Span(10, 15))
    assert not region.overlaps(Region(Span(15, 20), Span(22, 25)))


def test_Region_area():
    region = Region(Span(5, 10), Span(10, 15))
    assert region.area == 25


def test_Region_width():
    region = Region(Span(5, 10), Span(10, 15))
    assert region.width == 5


def test_Region_height():
    region = Region(Span(5, 10), Span(10, 15))
    assert region.height == 5


def test_Region_expand_em_no_eixo_xy():
    region = Region(Span(5, 10), Span(10, 15)).expand(Vector(5, 5))
    assert region == Region(Span(0, 15), Span(5, 20))


def test_Region_expand_em_no_eixo_x():
    region = Region(Span(5, 10), Span(10, 15)).expand(Vector(5, 0))
    assert region == Region(Span(0, 15), Span(10, 15))


def test_Region_expand_em_no_eixo_y():
    region = Region(Span(5, 10), Span(10, 15)).expand(Vector(0, 5))
    assert region == Region(Span(5, 10), Span(5, 20))


def test_Region_shrink_em_no_eixo_xy():
    region = Region(Span(0, 15), Span(10, 25)).shrink(Vector(5, 5))
    assert region == Region(Span(5, 10), Span(15, 20))


def test_Region_shrink_em_no_eixo_x():
    region = Region(Span(0, 15), Span(10, 25)).shrink(Vector(5, 0))
    assert region == Region(Span(5, 10), Span(10, 25))


def test_Region_shrink_em_no_eixo_y():
    region = Region(Span(0, 15), Span(10, 25)).shrink(Vector(0, 5))
    assert region == Region(Span(0, 15), Span(15, 20))


def test_Region_offset_positivo():
    region = Region(Span(50, 60), Span(55, 80))
    result = Region(Span(0, 15), Span(10, 25)).offset_to(region)
    assert result == Vector(50, 45)


def test_Region_offset_negativo():
    region = Region(Span(0, 15), Span(10, 25))
    result = Region(Span(50, 60), Span(55, 80)).offset_to(region)
    assert result == Vector(-50, -45)
