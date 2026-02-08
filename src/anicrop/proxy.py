from anicrop.canvas import Canvas
from anicrop.layer import Layer
from anicrop.receiver import RotationHandler
from anicrop.type import Translation, Vector

HANDLERS = {'rotate': RotationHandler}


class ProxyLayer:
    def __init__(self, layer: Layer, canvas: Canvas):
        super().__setattr__('_canvas', canvas)
        super().__setattr__('_edits', [])
        super().__setattr__('_layer', layer)
        super().__setattr__('_history', [])
        super().__setattr__('_translation', Translation())

    def __getattr__(self, name):
        original = object.__getattribute__(self, '_layer')
        return getattr(original, name)

    def __setattr__(self, name, value):
        if name in (
            '_canvas',
            '_edits',
            '_layer',
            '_history',
            '_translation',
            'translation',
        ):
            super().__setattr__(name, value)
            return

        if not hasattr(self._layer, name):
            raise AttributeError(f"A propriedade '{name}' não existe no objeto original.")

        # handler = HANDLERS[name](self._layer, self._edits, value)
        # handler.rotate()
        # self._history.append(handler)

    def __dir__(self) -> dict:
        return dir(self._layer)

    @property
    def translation(self) -> Translation:
        return self._translation

    @translation.setter
    def translation(self, translation: Translation) -> None:
        if not isinstance(translation, Translation):
            raise TypeError("tipo errado")
        self._translation = translation

    @property
    def position(self) -> Vector:
        canvas = self._canvas.region
        return canvas.offset_to(self._layer.region)

    # @position.setter
    # def position(self, other) -> Vector:
    #     raise NotImplementedError
