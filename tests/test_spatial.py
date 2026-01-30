from anicrop.spatial import Span
from pytest import raises, mark


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
        Span(0, 10).expand(-1, Span(0, 20))
    result = str(excinfo.value)
    assert result == expect


def test_Span_expand_com_zero():
    span = Span(0, 10).expand(0, Span(0, 20))
    result = span.length
    assert result == 10


def test_Span_expand_com_start_zero():
    span = Span(0, 10).expand(1, Span(0, 20))
    result = span.length
    assert result == 11


def test_Span_expand_com_start_maior_que_zero():
    span = Span(1, 10).expand(1, Span(0, 20))
    result = span.length
    assert result == 11


def test_Span_expand_com_end_expandido_igual_ao_bound_end():
    span = Span(1, 10).expand(1, Span(0, 11))
    result = span.length
    assert result == 11


def test_Span_expand_com_end_expandido_maior_que_bound_end():
    span = Span(1, 10).expand(2, Span(0, 11))
    result = span.length
    assert result == 11


def test_Span_expand_com_start_expandido_menor_que_bound_start():
    span = Span(6, 10).expand(2, Span(5, 11))
    result = span.start
    assert result == 5


def test_Span_shrink_com_valor_negativo():
    expect = ("Margin for shrink() must be non-negative, but got -1. "
              "To expand the span, use the expand() method with a positive margin.")
    with raises(ValueError) as excinfo:
        Span(0, 2).shrink(-1)
    result = str(excinfo.value)
    assert result == expect


def test_Span_shrink_com_valor_zero():
    span = Span(0, 10).shrink(0)
    result = span.length
    assert result == 10


def test_Span_shrink_com_valor_positivo():
    span = Span(0, 10).shrink(1)
    result = span.length
    assert result == 8


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
