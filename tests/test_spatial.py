from anicrop.spatial import RelativeSpan
from pytest import raises


def test_RelativeSpan_com_start_igual_a_end():
    expect = "Start '(0)' cannot be greater than end '(0)'."
    with raises(ValueError) as excinfo:
        RelativeSpan(0, 0)
    result = str(excinfo.value)
    assert result == expect


def test_RelativeSpan_com_start_maior_que_end():
    expect = "Start '(1)' cannot be greater than end '(0)'."
    with raises(ValueError) as excinfo:
        RelativeSpan(1, 0)
    result = str(excinfo.value)
    assert result == expect


def test_RelativeSpan_com_start_maior_que_end_mas_negativo():
    expect = "Start '(-1)' cannot be greater than end '(-10)'."
    with raises(ValueError) as excinfo:
        RelativeSpan(-1, -10)
    result = str(excinfo.value)
    assert result == expect


def test_RelativeSpan_length_com_valores_positivos():
    span = RelativeSpan(0, 1)
    result = span.length
    assert result == 1


def test_RelativeSpan_length_com_valores_negativos():
    span = RelativeSpan(-11, -1)
    result = span.length
    assert result == 10


def test_RelativeSpan_expand_com_valor_negativo():
    expect = ("Margin for expand() must be non-negative. To contract "
              "the span, use the shrink() method with a positive margin.")
    with raises(ValueError) as excinfo:
        span = RelativeSpan(0, 2)
        span.expand(-1)
    result = str(excinfo.value)
    assert result == expect


def test_RelativeSpan_expand_com_valor_zero():
    span = RelativeSpan(0, 10).expand(0)
    result = span.length
    assert result == 10


def test_RelativeSpan_expand_com_valor_positivo():
    span = RelativeSpan(0, 10).expand(1)
    result = span.length
    assert result == 12


def test_RelativeSpan_shrink_com_valor_negativo():
    expect = ("Margin for shrink() must be non-negative, but got -1. "
              "To expand the span, use the expand() method with a positive margin.")
    with raises(ValueError) as excinfo:
        RelativeSpan(0, 2).shrink(-1)
    result = str(excinfo.value)
    assert result == expect


def test_RelativeSpan_shrink_com_valor_zero():
    span = RelativeSpan(0, 10).shrink(0)
    result = span.length
    assert result == 10


def test_RelativeSpan_shrink_com_valor_positivo():
    span = RelativeSpan(0, 10).shrink(1)
    result = span.length
    assert result == 8


def test_RelativeSpan_shrink_gerando_start_igual_a_end():
    expect = "Start '(1)' cannot be greater than end '(1)'."
    with raises(ValueError) as excinfo:
        RelativeSpan(0, 2).shrink(1)
    result = str(excinfo.value)
    assert result == expect


def test_RelativeSpan_shrink_gerando_start_maior_que_end():
    expect = "Start '(1)' cannot be greater than end '(0)'."
    with raises(ValueError) as excinfo:
        RelativeSpan(0, 1).shrink(1)
    result = str(excinfo.value)
    assert result == expect
