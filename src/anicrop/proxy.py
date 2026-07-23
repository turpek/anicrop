from typing import Any
from anicrop.layer import Layer
from anicrop.history import GlobalHistory
from anicrop.command import SnapshotLayerCommand
from anicrop.transform import Composer

# Whitelist explícita de tipos de sub-objetos mutáveis retornados pelo Layer que suportam encadeamento
CHAINABLE_TYPES = (Composer,)


def is_property_with_setter(cls: type, name: str) -> bool:
    """Retorna True se o atributo na classe cls for uma property com setter."""
    class_attr = getattr(cls, name, None)
    return isinstance(class_attr, property) and class_attr.fset is not None


class ProxyLayer:
    """Proxy principal dedicado à classe Layer."""

    def __init__(self, layer: Layer, history: GlobalHistory):
        super().__setattr__('_layer', layer)
        super().__setattr__('_history', history)

    def __getattr__(self, name: str) -> Any:
        layer = object.__getattribute__(self, '_layer')
        history = object.__getattribute__(self, '_history')

        history.commit()
        attr = getattr(layer, name)

        if callable(attr):
            def method_wrapper(*args, **kwargs) -> Any:
                history.start_action(SnapshotLayerCommand, name, layer)
                result = attr(*args, **kwargs)
                if isinstance(result, Layer):
                    return self
                return result
            return method_wrapper

        if isinstance(attr, CHAINABLE_TYPES):
            history.start_action(SnapshotLayerCommand, name, layer)
            return attr

        # Para propriedades e objetos retornados pelo Layer, apenas retornamos
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_layer', '_history'):
            super().__setattr__(name, value)
            return

        layer = object.__getattribute__(self, '_layer')
        if not hasattr(layer, name):
            raise AttributeError(f"A propriedade '{name}' não existe no objeto original.")

        history = object.__getattribute__(self, '_history')
        history.start_action(SnapshotLayerCommand, name, layer)
        setattr(layer, name, value)

    def __dir__(self) -> list[str]:
        layer = object.__getattribute__(self, '_layer')
        own_attrs = set(super().__dir__())
        layer_attrs = set(dir(layer))
        return sorted(own_attrs | layer_attrs)

    def __repr__(self) -> str:
        layer = object.__getattribute__(self, '_layer')
        return f'<ProxyLayer for {layer}>'


# Registra o ProxyLayer como uma subclasse virtual de Layer
Layer.register(ProxyLayer)
