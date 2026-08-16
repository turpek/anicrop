from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator
import cv2
import numpy as np

from anicrop.canvas import Canvas
from anicrop.container import Container, GroupLayer, LayerStack
from anicrop.enums import ImageFormat
from anicrop.history import GlobalHistory
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.proxy import BaseHistoryProxy, GroupProxy, LayerStackProxy, ProxyLayer
from anicrop.render import CanvasRender, ViewportRender
from anicrop.viewport import Viewport


class DocumentPolicy(ABC):
    @abstractmethod
    def setup(self) -> tuple[GlobalHistory | None, Container]:
        ...

    @abstractmethod
    def process_layer(self, layer: Layer | GroupLayer, history: GlobalHistory | None) -> Any:
        ...


class ReactiveDocumentPolicy(DocumentPolicy):
    """Política com Histórico e Proxies ativados."""

    def setup(self) -> tuple[GlobalHistory, LayerStackProxy]:
        history = GlobalHistory()
        stack = LayerStackProxy(LayerStack(), history)
        return history, stack

    def process_layer(self, layer: Layer | GroupLayer, history: GlobalHistory | None) -> Any:
        if isinstance(layer, BaseHistoryProxy):
            return layer
        if isinstance(layer, GroupLayer):
            return GroupProxy(layer, history)
        return ProxyLayer(layer, history)


class DirectDocumentPolicy(DocumentPolicy):
    """Política de alta performance sem Histórico e sem Proxies (modo direto)."""

    def setup(self) -> tuple[None, LayerStack]:
        return None, LayerStack()

    def process_layer(self, layer: Layer | GroupLayer, history: GlobalHistory | None) -> Any:
        return getattr(layer, "_target", layer)


class Document:
    """
    Facade principal da biblioteca anicrop.
    Gerencia o Canvas, a Pilha de Camadas (LayerStack), o Histórico Global e Pipelines de Renderização.
    """

    _POLICIES: dict[bool, DocumentPolicy] = {
        True: ReactiveDocumentPolicy(),
        False: DirectDocumentPolicy(),
    }

    def __init__(self, name: str, width: int, height: int, wrap_proxy: bool = True):
        self.name = name
        self.canvas = Canvas.from_size(width, height)
        self.wrap_proxy = wrap_proxy

        self._policy = self._POLICIES[wrap_proxy]
        self.history, self.stack = self._policy.setup()

        self._viewport_render = ViewportRender()
        self._canvas_render = CanvasRender()

    @staticmethod
    def create_layer_instance(name: str, path: str | Path, opacity: float = 1.0, canvas: Canvas | None = None) -> Layer:
        """Helper estático para carregar uma imagem do disco e instanciar um Layer bruto."""
        img = Image.open(str(path), ImageFormat.RGBA)
        return Layer(image=img, opacity=opacity, name=name, canvas=canvas)

    @classmethod
    def open(cls, path: str | Path, name: str | None = None, wrap_proxy: bool = True) -> Document:
        """
        Abre uma imagem do disco e cria um Documento baseado no seu tamanho,
        inserindo a imagem como primeira camada.
        """
        layer_name = name or Path(path).stem
        layer = cls.create_layer_instance(layer_name, path)
        w, h = layer.canvas_size

        doc = cls(name=layer_name, width=w, height=h, wrap_proxy=wrap_proxy)
        layer._canvas = doc.canvas
        doc.add(layer)
        return doc

    @classmethod
    def from_image(cls, name: str, path: str | Path, wrap_proxy: bool = True) -> Document:
        """Alias compatível para Document.open()."""
        return cls.open(path=path, name=name, wrap_proxy=wrap_proxy)

    def _validate_unique_name(self, name: str) -> None:
        """Verifica se já existe alguma camada ou grupo com este nome no documento."""
        if self._find_in_container(self.stack, name, recursive=True) is not None:
            raise ValueError(f"A layer named '{name}' already exists in the document.")

    def _find_in_container(self, container: Container, name: str, recursive: bool = True) -> Any | None:
        """Busca interna auxiliar por nome na hierarquia de um container."""
        for child in container:
            if getattr(child, "name", None) == name:
                return child
            raw_child = getattr(child, "_target", child)
            if recursive and isinstance(raw_child, Container):
                found = self._find_in_container(child, name, recursive=True)
                if found is not None:
                    return found
        return None

    def add(self, layer: Layer | GroupLayer) -> Any:
        """
        Adiciona um Layer ou GroupLayer na pilha do documento.
        Garante a unicidade do nome da camada no documento.
        """
        layer_name = getattr(layer, "name", None)
        if layer_name is not None:
            self._validate_unique_name(layer_name)

        processed_layer = self._policy.process_layer(layer, self.history)
        self.stack.append(processed_layer)
        return processed_layer

    def add_layer(self, layer: Layer | GroupLayer) -> Any:
        """Alias compatível para add()."""
        return self.add(layer)

    def add_group(self, name: str = "Group") -> Any:
        """
        Cria e adiciona um novo Grupo (GroupLayer) na pilha do Documento.
        """
        self._validate_unique_name(name)
        group = GroupLayer(name=name)
        return self.add(group)

    def create_group(self, name: str = "Group") -> Any:
        """Alias compatível para add_group()."""
        return self.add_group(name=name)

    def load_layer(self, path: str | Path, name: str | None = None, opacity: float = 1.0) -> Any:
        """
        Carrega uma imagem do disco e adiciona como camada na pilha do documento.
        """
        layer_name = name or Path(path).stem
        layer = self.create_layer_instance(layer_name, path, opacity=opacity, canvas=self.canvas)
        return self.add(layer)

    def create_layer(self, name: str, path: str | Path, opacity: float = 1.0) -> Any:
        """Alias compatível para load_layer()."""
        return self.load_layer(path=path, name=name, opacity=opacity)

    def find(self, name: str, recursive: bool = True) -> Any | None:
        """
        Busca uma camada pelo nome no documento. Retorna None se não encontrar.
        """
        return self._find_in_container(self.stack, name, recursive=recursive)

    def __len__(self) -> int:
        """Retorna a quantidade de camadas na raiz da pilha."""
        return len(self.stack)

    def __iter__(self) -> Iterator[Any]:
        """Itera pelas camadas raiz da pilha."""
        return iter(self.stack)

    def __getitem__(self, key: int | slice | str) -> Any:
        """
        Acesso polimórfico a camadas por índice inteiro, slice ou nome (string).
        """
        if isinstance(key, (int, slice)):
            return self.stack[key]
        if isinstance(key, str):
            layer = self.find(key, recursive=True)
            if layer is None:
                raise KeyError(f"Layer named '{key}' not found in document.")
            return layer
        raise TypeError(f"Invalid key type {type(key).__name__}. Expected int, slice, or str.")

    def __contains__(self, item: Any | str) -> bool:
        """
        Verifica se uma camada (objeto ou nome) está presente no documento.
        """
        if isinstance(item, str):
            return self.find(item, recursive=True) is not None
        return item in self.stack or getattr(item, "_target", item) in self.stack

    def __delitem__(self, key: int | str) -> None:
        """
        Remove uma camada por índice ou nome.
        """
        if isinstance(key, int):
            self.stack.pop(key)
        elif isinstance(key, str):
            layer = self[key]
            self.remove(layer)
        else:
            raise TypeError(f"Invalid key type {type(key).__name__}. Expected int or str.")

    def remove(self, layer_or_name: Any | str) -> None:
        """
        Remove uma camada da pilha (aceita a instância da camada ou seu nome).
        """
        if isinstance(layer_or_name, str):
            layer = self[layer_or_name]
        else:
            layer = layer_or_name

        if layer in self.stack:
            self.stack.remove(layer)
            return

        raw_target = getattr(layer, "_target", layer)
        if hasattr(raw_target, "parent") and raw_target.parent is not None:
            raw_target.parent.remove(layer)
        else:
            raise ValueError(f"Layer {layer} not found in document hierarchy.")

    def pop(self, index: int = -1) -> Any:
        """Remove e retorna a camada no índice especificado."""
        return self.stack.pop(index)

    def clear(self) -> None:
        """Remove todas as camadas da raiz do documento."""
        self.stack.clear()

    def get_bottom_layer(self) -> Any:
        """Retorna a camada base (fundo) da pilha."""
        return self.stack[-1]

    def render(self) -> Image:
        """
        Renderiza a composição final em alta resolução no Canvas e retorna o objeto Image (RGBA).
        """
        return self._canvas_render.render_scene(self.stack, self.canvas)

    def preview(self, viewport: Viewport) -> np.ndarray:
        """
        Gera o Preview para renderizar na interface de usuário via Viewport.
        """
        result_img = self._viewport_render.render_scene(self.stack, viewport)
        return result_img[...]

    def export(self, path: str | Path) -> None:
        """
        Renderiza a composição final em alta resolução e salva no disco.
        """
        final_img = self.render()
        frame = final_img[...]

        if frame.shape[2] == 4:
            frame_save = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        elif frame.shape[2] == 3:
            frame_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_save = frame

        cv2.imwrite(str(path), frame_save)
