from operator import add, sub, mul, truediv


class OperationFloat(float):
    """
    Um float que lembra qual operação matemática o criou.
    """
    def __new__(cls, value, operation=None, origin_value=None):
        return super().__new__(cls, value)

    def __init__(self, value, operation=None, origin_value=None):
        self.operation = operation
        self.origin_value = origin_value

    def __add__(self, other):
        result = super().__add__(other)
        return OperationFloat(result, operation=add, origin_value=float(other))

    def __radd__(self, other):
        result = super().__radd__(other)
        return OperationFloat(result, operation=add, origin_value=float(other))

    def __sub__(self, other):
        result = super().__sub__(other)
        return OperationFloat(result, operation=sub, origin_value=float(other))

    def __rsub__(self, other):
        result = super().__rsub__(other)
        return OperationFloat(result, operation=sub, origin_value=float(other))

    def __mul__(self, other):
        result = super().__mul__(other)
        return OperationFloat(result, operation=mul, origin_value=float(other))

    def __rmul__(self, other):
        result = super().__rmul__(other)
        return OperationFloat(result, operation=mul, origin_value=float(other))

    def __truediv__(self, other):
        res = super().__truediv__(other)
        return OperationFloat(res, operation=truediv, origin_value=float(other))

    def __rtruediv__(self, other):
        raise NotImplementedError
        # res = super().__rtruediv__(other)
        # return OperationFloat(res, operation=truediv, origin_value=float(other))

    def __repr__(self):
        return f"{float(self)}"
