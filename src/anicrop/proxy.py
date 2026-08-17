import weakref
from typing import Any
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.history import GlobalHistory
from anicrop.command import BaseLayerCommand, LayerImageCommand, ReparentCommand, MaskCommand
from anicrop.container import Container, LayerStack, GroupLayer, BaseLayer, NullContainer


def is_property_with_setter(cls: type, name: str) -> bool:
    """Retorna True se o atributo na classe cls for uma property com setter."""
    class_attr = getattr(cls, name, None)
    return isinstance(class_attr, property) and class_attr.fset is not None


class ProxyRegistry:
    """Identity Map usando WeakValueDictionary para garantir instância única de Proxy por target."""

    def __init__(self, history: Any):
        self._history = history
        self._cache: weakref.WeakValueDictionary[int, BaseHistoryProxy] = (
            weakref.WeakValueDictionary()
        )

    def get_or_create(self, target: Any) -> Any:
        if target is None or type(target) is NullContainer:
            return target
        if isinstance(target, BaseHistoryProxy):
            return target

        target_id = id(target)
        if target_id in self._cache:
            return self._cache[target_id]

        proxy_cls: type[BaseHistoryProxy]
        if isinstance(target, GroupLayer):
            proxy_cls = GroupProxy
        elif isinstance(target, LayerStack):
            proxy_cls = LayerStackProxy
        elif isinstance(target, Layer):
            proxy_cls = ProxyLayer
        elif isinstance(target, Mask):
            proxy_cls = ProxyMask
        else:
            proxy_cls = BaseHistoryProxy

        return proxy_cls(target, self._history, registry=self)


def get_registry_for_history(history: Any) -> ProxyRegistry:
    reg = getattr(history, '_proxy_registry', None)
    if isinstance(reg, ProxyRegistry):
        return reg
    reg = ProxyRegistry(history)
    try:
        history._proxy_registry = reg
    except Exception:
        pass
    return reg


class BaseHistoryProxy:
    """Classe base que cuida da interceptação segura de estado e histórico."""
    _ACTION_ROUTER: dict[str, type] = {}
    _CHAINABLE_PROPERTIES: tuple[str, ...] = ()

    def __new__(cls, target: Any, history: GlobalHistory, registry: ProxyRegistry | None = None):
        if isinstance(target, BaseHistoryProxy):
            return target

        reg = registry or get_registry_for_history(history)
        target_id = id(target)
        if target_id in reg._cache:
            return reg._cache[target_id]

        return super().__new__(cls)

    def __init__(self, target: Any, history: GlobalHistory, registry: ProxyRegistry | None = None):
        if hasattr(self, '_target'):
            return

        if registry is None:
            registry = get_registry_for_history(history)

        super().__setattr__('_target', target)
        super().__setattr__('_history', history)
        super().__setattr__('_registry', registry)
        registry._cache[id(target)] = self

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
        if name == "__setitem__" and args:
            return args[0]
        return None

    def __dir__(self):
        target = object.__getattribute__(self, '_target')
        return dir(target)

    def __getattribute__(self, name: str) -> Any:
        if name in (
            "_target",
            "_history",
            "_registry",
            "_ACTION_ROUTER",
            "_CHAINABLE_PROPERTIES",
            "_resolve_command",
            "_extract_command_value",
            "__dict__",
            "__class__"
        ):
            return object.__getattribute__(self, name)

        if name == "parent":
            target = object.__getattribute__(self, "_target")
            p = getattr(target, "parent")
            if p is None or type(p) is NullContainer:
                return p
            registry = object.__getattribute__(self, "_registry")
            return registry.get_or_create(p)

        target = object.__getattribute__(self, "_target")
        attr = getattr(target, name)

        history = object.__getattribute__(self, "_history")
        chainable = object.__getattribute__(self, "_CHAINABLE_PROPERTIES")
        action_router = object.__getattribute__(self, "_ACTION_ROUTER")

        if name in chainable:
            if not history.is_active:
                return attr
            cmd_cls = self._resolve_command(name)
            history.start_action(cmd_cls, name, self)
            return attr

        if callable(attr) and name in action_router:
            def method_wrapper(*args, **kwargs) -> Any:
                history = object.__getattribute__(self, '_history')

                unwrapped_args = tuple(
                    object.__getattribute__(arg, '_target') if isinstance(
                        arg, BaseHistoryProxy) else arg
                    for arg in args
                )
                unwrapped_kwargs = {
                    k: (object.__getattribute__(v, '_target') if isinstance(
                        v, BaseHistoryProxy) else v)
                    for k, v in kwargs.items()
                }

                if not history.is_active:
                    result = attr(*unwrapped_args, **unwrapped_kwargs)
                    if result is target:
                        return self
                    return result

                cmd_cls = self._resolve_command(name)

                value = self._extract_command_value(
                    name, cmd_cls, target, args)
                history.start_action(cmd_cls, name, self, value)

                with history.disabled():
                    result = attr(*unwrapped_args, **unwrapped_kwargs)

                if result is target:
                    return self
                if hasattr(self, '_registry') and not isinstance(result, BaseHistoryProxy):
                    registry = object.__getattribute__(self, '_registry')
                    if isinstance(result, (BaseLayer, Container, Mask)):
                        return registry.get_or_create(result)
                return result
            return method_wrapper

        if hasattr(self, '_registry') and not isinstance(attr, BaseHistoryProxy):
            registry = object.__getattribute__(self, '_registry')
            if isinstance(attr, (BaseLayer, Container, Mask)):
                return registry.get_or_create(attr)

        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "parent":
            raise AttributeError(
                "Direct assignment to 'parent' is not supported. "
                "Use container methods like 'parent.append(child)' or 'parent.remove(child)' instead."
            )

        target = object.__getattribute__(self, '_target')
        action_router = object.__getattribute__(self, '_ACTION_ROUTER')

        if name in action_router:
            history = object.__getattribute__(self, '_history')

            if not history.is_active:
                setattr(target, name, value)
                return

            cmd_cls = self._resolve_command(name)
            history.start_action(cmd_cls, name, self, value)

            with history.disabled():
                setattr(target, name, value)
        else:
            setattr(target, name, value)

    def __getitem__(self, item: Any) -> Any:
        target = object.__getattribute__(self, '_target')
        return target[item]

    def __setitem__(self, item: Any, value: Any) -> None:
        target = object.__getattribute__(self, '_target')
        action_router = object.__getattribute__(self, '_ACTION_ROUTER')

        if "__setitem__" in action_router:
            history = object.__getattribute__(self, '_history')

            if not history.is_active:
                target[item] = value
                return

            cmd_cls = self._resolve_command("__setitem__")
            value_arg = self._extract_command_value("__setitem__", cmd_cls, target, (item, value))
            history.start_action(cmd_cls, "__setitem__", self, value_arg)

            with history.disabled():
                target[item] = value
        else:
            target[item] = value


class ProxyMask(BaseHistoryProxy):
    """Proxy reativo para a classe Mask."""

    _ACTION_ROUTER = {
        "__setitem__": MaskCommand,
        "visible": MaskCommand,
        "invert": MaskCommand,
        "offset": MaskCommand,
    }


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
        "layout": BaseLayerCommand,
        "set_mask": BaseLayerCommand,
        "remove_mask": BaseLayerCommand,
        "clear_mask": BaseLayerCommand,
        "add_effect": BaseLayerCommand,
        "clear_effects": BaseLayerCommand,
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
        registry = object.__getattribute__(self, '_registry')
        if cmd_cls is ReparentCommand:
            if name in ("append", "remove", "move"):
                return registry.get_or_create(args[0])
            elif name == "insert":
                return registry.get_or_create(args[1])
            elif name == "pop":
                idx = args[0] if args else -1
                return registry.get_or_create(target[idx])
        elif name == "__setitem__" and args:
            return args[0]
        return None

    def __iter__(self):
        registry = object.__getattribute__(self, '_registry')
        for item in object.__getattribute__(self, '_target'):
            yield registry.get_or_create(item)

    def __reversed__(self):
        registry = object.__getattribute__(self, '_registry')
        for item in reversed(object.__getattribute__(self, '_target')):
            yield registry.get_or_create(item)

    def __len__(self):
        return len(object.__getattribute__(self, '_target'))

    def __contains__(self, item):
        target = object.__getattribute__(self, '_target')
        return item in target or getattr(item, '_target', item) in target

    def __getitem__(self, item):
        registry = object.__getattribute__(self, '_registry')
        raw_item = object.__getattribute__(self, '_target')[item]
        return registry.get_or_create(raw_item)


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
        "layout": BaseLayerCommand,
        "set_mask": BaseLayerCommand,
        "remove_mask": BaseLayerCommand,
        "clear_mask": BaseLayerCommand,
        "add_effect": BaseLayerCommand,
        "clear_effects": BaseLayerCommand,
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
Mask.register(ProxyMask)
