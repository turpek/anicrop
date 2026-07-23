from typing import Any
from anicrop.layer import Layer
from anicrop.history import GlobalHistory
from anicrop.command import SnapshotCommand
from anicrop.transform import Composer

# Whitelist explícita de tipos de sub-objetos mutáveis retornados pelo Layer que suportam encadeamento
CHAINABLE_TYPES = (Composer,)


def is_property_with_setter(cls: type, name: str) -> bool:
    """Retorna True se o atributo na classe cls for uma property com setter."""
    class_attr = getattr(cls, name, None)
    return isinstance(class_attr, property) and class_attr.fset is not None


class GenericProxy:
    """Proxy genérico auxiliar para sub-objetos mutáveis do Layer (como Composer).

    Intercepta chamadas encadeadas em sub-objetos e registra snapshots no root_layer.
    """

    def __init__(
        self,
        target: Any,
        root_layer: Layer,
        history: GlobalHistory,
        action_name: str,
        initial_old_state: dict[str, Any] | None = None
    ):
        super().__setattr__('_target', target)
        super().__setattr__('_root_layer', root_layer)
        super().__setattr__('_history', history)
        super().__setattr__('_action_name', action_name)
        super().__setattr__('_initial_old_state', initial_old_state)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, '_target')
        root_layer = object.__getattribute__(self, '_root_layer')
        history = object.__getattribute__(self, '_history')
        action_name = object.__getattribute__(self, '_action_name')

        attr = getattr(target, name)

        if callable(attr):
            def method_wrapper(*args, **kwargs) -> Any:
                old_state = object.__getattribute__(self, '_initial_old_state')

                result = attr(*args, **kwargs)

                # 1. Se for transição para o Layer (ex: .finish())
                if isinstance(result, Layer):
                    raise NotImplementedError

                # 2. Se a cadeia CONTINUA (ex: .rotate() retornou o Composer)
                if isinstance(result, CHAINABLE_TYPES):
                    new_state = SnapshotCommand.capture_state(root_layer)
                    history.push(SnapshotCommand, action_name, root_layer, new_state)
                    return GenericProxy(result, root_layer, history, action_name, initial_old_state=old_state)

                # 3. FIM DO ENCADEAMENTO: Executa a gravação da foto no histórico!
                new_state = SnapshotCommand.capture_state(root_layer)
                history.push(SnapshotCommand, action_name, root_layer, new_state)
                return result

            return method_wrapper

        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_target', '_root_layer', '_history', '_action_name', '_initial_old_state'):
            super().__setattr__(name, value)
            return

        target = object.__getattribute__(self, '_target')
        root_layer = object.__getattribute__(self, '_root_layer')
        history = object.__getattribute__(self, '_history')
        action_name = object.__getattribute__(self, '_action_name')

        if not hasattr(target, name):
            raise AttributeError(f"A propriedade '{name}' não existe no objeto original.")

        setattr(target, name, value)
        new_state = SnapshotCommand.capture_state(root_layer)

        history.push(SnapshotCommand, action_name, root_layer, new_state)


class ProxyLayer:
    """Proxy principal dedicado à classe Layer."""

    def __init__(self, layer: Layer, history: GlobalHistory):
        super().__setattr__('_layer', layer)
        super().__setattr__('_history', history)

    def __getattr__(self, name: str) -> Any:
        layer = object.__getattribute__(self, '_layer')
        history = object.__getattribute__(self, '_history')

        # 1. Capturamos o estado do Layer ANTES de qualquer leitura/mutação no layer!
        state_current = SnapshotCommand.capture_state(layer)
        history.push(SnapshotCommand, name, layer, state_current)

        # 2. Avaliamos o atributo no layer real
        attr = getattr(layer, name)

        # 3. Se for um método do próprio Layer (ex: set_transform, add_edit)
        if callable(attr):
            def method_wrapper(*args, **kwargs) -> Any:
                result = attr(*args, **kwargs)

                if isinstance(result, Layer):
                    new_state = SnapshotCommand.capture_state(layer)
                    history.push(SnapshotCommand, name, layer, new_state)
                    return self
                if isinstance(result, CHAINABLE_TYPES):
                    return GenericProxy(result, layer, history, name)

                return result

            return method_wrapper

        # 4. Se retornar um sub-objeto mutável da whitelist (ex: layer.transform -> Composer)
        if isinstance(attr, CHAINABLE_TYPES):
            return GenericProxy(
                target=attr,
                root_layer=layer,
                history=history,
                action_name=name,
                initial_old_state=state_current
            )

        # 5. Leitura simples de propriedade ou valor primitivo
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_layer', '_history'):
            super().__setattr__(name, value)
            return

        layer = object.__getattribute__(self, '_layer')
        if not hasattr(layer, name):
            raise AttributeError(f"A propriedade '{name}' não existe no objeto original.")

        history = object.__getattribute__(self, '_history')

        old_state = SnapshotCommand.capture_state(layer)
        history.push(SnapshotCommand, name, layer, old_state)

        setattr(layer, name, value)
        new_state = SnapshotCommand.capture_state(layer)
        history.push(SnapshotCommand, name, layer, new_state)

    def __dir__(self) -> list[str]:
        layer = object.__getattribute__(self, '_layer')
        own_attrs = set(super().__dir__())
        layer_attrs = set(dir(layer))
        return sorted(own_attrs | layer_attrs)

    def __repr__(self) -> str:
        layer = object.__getattribute__(self, '_layer')
        return str(layer)
