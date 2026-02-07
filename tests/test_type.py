from anicrop.type import OperationFloat
from operator import add, sub, mul, truediv


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


def test_OperationFloat_rtruediv():
    value = OperationFloat(2)
    result = 10 / value
    assert result == 5.0
    assert result.operation == truediv
    assert result.origin_value == 10


def test_OperationFloat_repr():
    value = OperationFloat(5.5)
    assert repr(value) == "5.5"
