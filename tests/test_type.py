from anicrop.type import OperationFloat, Rotation, Translation, Vector
from operator import add, sub, mul, truediv
from pytest import raises
import pytest


OPS = [
    (add, 'add'),
    (sub, 'sub'),
    (mul, 'mul'),
    (truediv, 'div')
]


def test_OperationFloat_add():
    value = OperationFloat(5)
    result = value + 3
    assert result == 8
    assert result.operation == add
    assert result.origin_value == 3


def test_OperationFloat_radd():
    value = OperationFloat(5)
    result = 3 + value
    assert result == 8
    assert result.operation == add
    assert result.origin_value == 3


def test_OperationFloat_sub():
    value = OperationFloat(10)
    result = value - 3
    assert result == 7
    assert result.operation == sub
    assert result.origin_value == 3


def test_OperationFloat_rsub():
    value = OperationFloat(3)
    result = 10 - value
    assert result == 7
    assert result.operation == sub
    assert result.origin_value == 10


def test_OperationFloat_mul():
    value = OperationFloat(4)
    result = value * 2.5
    assert result == 10.0
    assert result.operation == mul
    assert result.origin_value == 2.5


def test_OperationFloat_rmul():
    value = OperationFloat(4)
    result = 2.5 * value
    assert result == 10.0
    assert result.operation == mul
    assert result.origin_value == 2.5


def test_OperationFloat_truediv():
    value = OperationFloat(10)
    result = value / 2
    assert result == 5.0
    assert result.operation == truediv
    assert result.origin_value == 2


@pytest.mark.skip(reason='deprecado')
def test_OperationFloat_rtruediv():
    value = OperationFloat(2)
    result = 10 / value
    assert result == 5.0
    assert result.operation == truediv
    assert result.origin_value == 10


def test_OperationFloat_repr():
    value = OperationFloat(5.5)
    assert repr(value) == "5.5"


# ########################### Teste para o tipo Rotation ##########################################

def test_Rotation_valor_padrao():
    value = Rotation()
    assert value == 0
    assert value.pivot == Vector(0.5, 0.5)


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', [-45, 0, 45], ids=['-45', '0', '45'])
def test_Rotation_operacoes_com_valor_padrao(op, op_name, value):
    rot = Rotation()
    if op_name == 'div' and value == 0:
        with raises(ZeroDivisionError, match='float division by zero'):
            op(rot, value)
    else:
        expect = op(float(rot), value)
        result = op(rot, value)
        assert result == expect
        assert result.pivot == Vector(0.5, 0.5)


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', [-45, 0, 45], ids=['-45', '0', '45'])
def test_Rotation_operacoes_com_valor_nao_padrao(op, op_name, value):
    rot = Rotation()
    if op_name == 'div' and value == 0:
        with raises(ZeroDivisionError, match='float division by zero'):
            op(rot, value)
    else:
        rot += 45
        expect = op(float(rot), value)
        result = op(rot, value)
        assert result == expect


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize('value', [(-45, 0.5, 0.5), (0, 0, 0), (45, 1, 1)], ids=['-45', '0', '45'])
def test_Rotation_operacoes_com_valor_nao_padrao_e_pivot(op, op_name, value):
    rot = Rotation()
    if op_name == 'div' and value[0] == 0:
        with raises(ZeroDivisionError, match='float division by zero'):
            op(rot, value)
    else:
        pivot = Vector(value[1], value[2])
        rot += 45
        expect = op(float(rot), value[0])
        result = op(rot, value)
        assert result == expect
        assert result.pivot == pivot


def test_Translation_valor_padrao():
    value = Translation()
    assert value.x == 0
    assert value.y == 0


@pytest.mark.parametrize("op,op_name", OPS[:2], ids=[name for _, name in OPS[:2]])
@pytest.mark.parametrize('value', [-45, 0, 45], ids=['-45', '0', '45'])
def test_Translation_operacoes_com_valores_padroes_nos_eixos(op, op_name, value):
    trans = Translation()
    expect = Translation(op(0, value), op(0, value))

    result_x = op(trans.x, value)
    result_y = op(trans.y, value)

    assert result_x == expect.x
    assert result_y == expect.y


@pytest.mark.parametrize("op,op_name", OPS[:2], ids=[name for _, name in OPS[:2]])
@pytest.mark.parametrize('value', [-45, 0, 45], ids=['-45', '0', '45'])
def test_Translation_operacoes_com_valores_nao_padroes_nos_eixos(op, op_name, value):
    trans = Translation(10, -10)
    expect = Translation(op(10, value), op(-10, value))

    result_x = op(trans.x, value)
    result_y = op(trans.y, value)

    assert result_x == expect.x
    assert result_y == expect.y


@pytest.mark.parametrize("op,op_name", OPS[:2], ids=[name for _, name in OPS[:2]])
@pytest.mark.parametrize('value', [-45, 0, 45], ids=['-45', '0', '45'])
def test_Translation_simetrica_nos_dois_eixos(op, op_name, value):
    trans = Translation()
    expect = Translation(op(0, value), op(0, value))
    result = op(trans, value)
    assert result == expect


@pytest.mark.parametrize("op,op_name", OPS[:2], ids=[name for _, name in OPS[:2]])
@pytest.mark.parametrize('dx,dy', [(-45, 15), (0, 0), (45, -15)], ids=['-45,15', '0,0', '45,-15'])
def test_Translation_com_tupla_valores_nao_padroes(op, op_name, dx, dy):
    trans = Translation(10, -15)
    expect = Translation(op(10, dx), op(-15, dy))
    result = op(trans, (dx, dy))
    assert result == expect


@pytest.mark.parametrize("op,op_name", OPS[:2], ids=[name for _, name in OPS[:2]])
@pytest.mark.parametrize('dx,dy', [(-45, 15), (0, 0), (45, -15)], ids=['-45,15', '0,0', '45,-15'])
def test_Translation_com_Vector_valores_nao_padroes(op, op_name, dx, dy):
    trans = Translation(10, -15)
    expect = Translation(op(10, dx), op(-15, dy))
    result = op(trans, Vector(dx, dy))
    assert result == expect


@pytest.mark.parametrize("op,op_name", OPS[:2], ids=[name for _, name in OPS[:2]])
@pytest.mark.parametrize('value', [{-45, 15}, {0, 0}, {45, -15}], ids=['-45,15', '0,0', '45,-15'])
def test_Translation_com_tipo_invalido(op, op_name, value):
    with raises(NotImplementedError, match='Unsupported type: set'):
        trans = Translation(10, -15)
        op(trans, value)
