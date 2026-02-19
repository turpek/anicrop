from anicrop.layer import Layer
from anicrop.command import SetAttributeCommand
from anicrop.history import GlobalHistory


class ProxyLayer:
    def __init__(self, layer: Layer, history: GlobalHistory):
        super().__setattr__('_layer', layer)
        super().__setattr__('_history', history)

    def __getattr__(self, name):
        original = object.__getattribute__(self, '_layer')
        return getattr(original, name)

    def __setattr__(self, name, value):
        if name in (
            '_layer',
            '_history',
        ):
            super().__setattr__(name, value)
            return

        if not hasattr(self._layer, name):
            raise AttributeError(f"A propriedade '{name}' não existe no objeto original.")

        self._history.push(SetAttributeCommand, name, self._layer, value)

    def __dir__(self) -> dict:
        own_attrs = set(super().__dir__())
        layer_attrs = set(dir(self._layer))
        return sorted(own_attrs | layer_attrs)

    def __repr__(self) -> str:
        return str(self._layer)
