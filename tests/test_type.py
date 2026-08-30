from anicrop.type import Rotation, Vector, Scale
from operator import add, sub
import numpy as np
import pytest

# Apenas add e sub são suportados pela nova implementação de Rotation e Scale
OPS = [
    (add, "add"),
    (sub, "sub"),
]


def test_Rotation_valor_padrao():
    value = Rotation()
    assert value.angle == 0.0
    assert value.pivot_x == 0.5
    assert value.pivot_y == 0.5


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize("value", [-45, 0, 45], ids=["-45", "0", "45"])
def test_Rotation_operacoes_com_valor_padrao(op, op_name, value):
    rot = Rotation()
    expect_angle = op(0.0, value)
    result = op(rot, value)

    assert isinstance(result, Rotation)
    assert result.angle == expect_angle
    assert result.pivot_x == 0.5
    assert result.pivot_y == 0.5


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize("value", [-45, 0, 45], ids=["-45", "0", "45"])
def test_Rotation_operacoes_com_valor_nao_padrao(op, op_name, value):
    rot = Rotation(angle=45.0)
    expect_angle = op(45.0, value)
    result = op(rot, value)

    assert result.angle == expect_angle
    assert result.pivot_x == 0.5
    assert result.pivot_y == 0.5


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize(
    "value", [(-45, 0.1, 0.2), (0, 0, 0), (45, 1, 1)], ids=["-45", "0", "45"]
)
def test_Rotation_operacoes_com_tupla_alterando_pivot(op, op_name, value):
    rot = Rotation(angle=10.0)
    expect_angle = op(10.0, value[0])

    result = op(rot, value)

    assert result.angle == expect_angle
    assert result.pivot_x == float(value[1])
    assert result.pivot_y == float(value[2])


def test_Rotation_tipo_invalido():
    rot = Rotation()
    with pytest.raises(TypeError):
        _ = rot + "string"


def test_Vector_valor_padrao():
    v = Vector()
    assert v.x == 0
    assert v.y == 0


def test_Vector_abs():
    v = Vector(-10, 20)
    v_abs = abs(v)
    assert v_abs.x == 10
    assert v_abs.y == 20


def test_Vector_iter():
    v = Vector(1, 2)
    x, y = v
    assert x == 1
    assert y == 2


def test_Scale_valor_padrao():
    # sx e sy são obrigatórios em Scale
    s = Scale(1.0, 1.0)
    assert s.sx == 1.0
    assert s.sy == 1.0
    assert s.pivot_x == 0.5
    assert s.pivot_y == 0.5
    assert s.pivot == (0.5, 0.5)


@pytest.mark.parametrize("op,op_name", OPS, ids=[name for _, name in OPS])
@pytest.mark.parametrize("value", [0.5, 1, 2], ids=["0.5", "1", "2"])
def test_Scale_operacoes_escalar(op, op_name, value):
    s = Scale(1.0, 1.0)
    result = op(s, value)

    assert result.sx == op(1.0, value)
    assert result.sy == op(1.0, value)
    assert result.pivot_x == 0.5
    assert result.pivot_y == 0.5


def test_Scale_tipo_invalido():
    s = Scale(1.0, 1.0)
    with pytest.raises(TypeError):
        _ = s - {"set", "invalido"}


def test_Rotation_matrix_identidade():
    rot = Rotation(0.0)
    m = rot.matrix
    np.testing.assert_allclose(m, np.eye(3), atol=1e-5)


def test_Rotation_matrix_90_graus():
    # Sentido horário para Y-down: cos(90)=0, sin(90)=1
    rot = Rotation(90.0)
    m = rot.matrix
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    np.testing.assert_allclose(m, expected, atol=1e-5)


def test_Scale_matrix_identidade():
    s = Scale(1.0, 1.0)
    m = s.matrix
    np.testing.assert_allclose(m, np.eye(3), atol=1e-5)


def test_Scale_matrix_com_valores():
    s = Scale(2.0, 3.0)
    m = s.matrix
    expected = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=np.float32)
    np.testing.assert_allclose(m, expected, atol=1e-5)
