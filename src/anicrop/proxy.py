import weakref
from typing import Any
from anicrop.layer import Layer
from anicrop.history import GlobalHistory
from anicrop.command import BaseLayerCommand, LayerImageCommand, ReparentCommand
from anicrop.container import Container, LayerStack, GroupLayer, BaseLayer


def is_property_with_setter(cls: type, name: str) -> bool:
    """Retorna True se o atributo na classe cls for uma property com setter."""
    class_attr = getattr(cls, name, None)
    return isinstance(class_attr, property) and class_attr.fset is not None


class BaseHistoryProxy:
    """Classe base que cuida da interceptação segura de estado e histórico."""
    _ACTION_ROUTER = {}
    _CHAINABLE_PROPERTIES = ()

    def __init__(self, target: Any, history: GlobalHistory):
        super().__setattr__('_target', target)
        super().__setattr__('_history', history)
        try:
            target._proxy = weakref.ref(self)
        except (AttributeError, TypeError):
            pass

    def __eq__(self, other: Any) -> bool:
        target = object.__getattribute__(self, '_target')
        other_target = getattr(other, '_target', other)
        return target is other_target or target == other_target

    def __hash__(self) -> int:
        return hash(object.__getattribute__(self, '_target'))

    def _resolve_command(self, name: str):
        if name not in self._ACTION_ROUTER:
            raise KeyError(f"Ação '{name}' não roteada no proxy {type(self).__name__}.")
        return self._ACTION_ROUTER[name]

    def _extract_command_value(self, name: str, cmd_cls: type, target: Any, args: tuple) -> Any:
        """Hook para subclasses extraírem o parâmetro `value` do comando a partir dos argumentos do método."""
        return None

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, '_target')
        history = object.__getattribute__(self, '_history')

        if name == "parent":
            p = getattr(target, "parent")
            proxy_ref = getattr(p, "_proxy", None)
            if proxy_ref is not None:
                p_proxy = proxy_ref()
                if p_proxy is not None:
                    return p_proxy
            return p

        attr = getattr(target, name)

        if name in self._CHAINABLE_PROPERTIES:
            cmd_cls = self._resolve_command(name)
            history.start_action(cmd_cls, name, self)
            return attr

        if callable(attr) and name in self._ACTION_ROUTER:
            def method_wrapper(*args, **kwargs) -> Any:
                history = object.__getattribute__(self, '_history')

                # Via expressa: se o histórico estiver desativado, repassa direto
                if not history.is_active:
                    result = attr(*args, **kwargs)
                    if result is target:
                        return self
                    return result

                cmd_cls = self._resolve_command(name)

                value = self._extract_command_value(name, cmd_cls, target, args)
                history.start_action(cmd_cls, name, self, value)

                # Executa a ação do Layer COM O HISTÓRICO DESATIVADO
                with history.disabled():
                    result = attr(*args, **kwargs)

                # Se o método retornar a si mesmo para encadeamento, devolve o proxy
                if result is target:
                    return self
                return result
            return method_wrapper

        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_target', '_history'):
            super().__setattr__(name, value)
            return

        target = object.__getattribute__(self, '_target')
        history = object.__getattribute__(self, '_history')

        # Via expressa: se o histórico estiver desativado, repassa direto
        if not history.is_active:
            setattr(target, name, value)
            return

        # Se não for uma ação rastreável, apenas passa para o alvo
        try:
            cmd_cls = self._resolve_command(name)
        except KeyError:
            setattr(target, name, value)
            return

        history.start_action(cmd_cls, name, self, value)

        with history.disabled():
            setattr(target, name, value)

    def __dir__(self) -> list[str]:
        return dir(object.__getattribute__(self, '_target'))


class ProxyLayer(BaseHistoryProxy):
    """Proxy dedicado à classe Layer."""

    _ACTION_ROUTER = {
        "name": BaseLayerCommand,
        "opacity": BaseLayerCommand,
        "blend_mode": BaseLayerCommand,
        "visible": BaseLayerCommand,
        "transform": BaseLayerCommand,
        "x": BaseLayerCommand,
        "y": BaseLayerCommand,
        "region": BaseLayerCommand,
        "add_edit": LayerImageCommand,
        "opacity_mask": LayerImageCommand,
    }

    _CHAINABLE_PROPERTIES = ("transform",)


class BaseContainerProxy(BaseHistoryProxy):
    """Proxy base para contêineres (Container)."""

    _ACTION_ROUTER = {
        "append": ReparentCommand,
        "insert": ReparentCommand,
        "remove": ReparentCommand,
        "move": ReparentCommand,
        "pop": ReparentCommand,
    }

    def _extract_command_value(self, name: str, cmd_cls: type, target: Any, args: tuple) -> Any:
        if cmd_cls is ReparentCommand:
            if name in ("append", "remove", "move"):
                return args[0]
            elif name == "insert":
                return args[1]
            elif name == "pop":
                idx = args[0] if args else -1
                return target[idx]
        return None

    def __iter__(self):
        return iter(object.__getattribute__(self, '_target'))

    def __len__(self):
        return len(object.__getattribute__(self, '_target'))

    def __contains__(self, item):
        target = object.__getattribute__(self, '_target')
        return item in target or getattr(item, '_target', item) in target

    def __getitem__(self, item):
        return object.__getattribute__(self, '_target')[item]


class LayerStackProxy(BaseContainerProxy):
    """Proxy dedicado a LayerStack (Root Container)."""
    pass


class GroupProxy(BaseContainerProxy):
    """Proxy dedicado a GroupLayer (Container + BaseLayer)."""

    _ACTION_ROUTER = {
        **BaseContainerProxy._ACTION_ROUTER,
        "name": BaseLayerCommand,
        "opacity": BaseLayerCommand,
        "blend_mode": BaseLayerCommand,
        "visible": BaseLayerCommand,
        "transform": BaseLayerCommand,
        "region": BaseLayerCommand,
    }

    _CHAINABLE_PROPERTIES = ("transform",)

    def __repr__(self):
        return f'GroupProxy(name="{self.name}")'


Container.register(BaseContainerProxy)
LayerStack.register(LayerStackProxy)
GroupLayer.register(GroupProxy)
BaseLayer.register(GroupProxy)
BaseLayer.register(ProxyLayer)
Layer.register(ProxyLayer)
