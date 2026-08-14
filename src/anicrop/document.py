from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
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
        return getattr(layer, '_target', layer)


class Document:
    """
    Facade principal da biblioteca.
    Gerencia o Canvas, o Histórico Global e a Pilha de Camadas (LayerStack).
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

    @staticmethod
    def create_layer_instance(name: str, path: str, opacity: float = 1.0, canvas: Canvas | None = None) -> Layer:
        """Helper estático que faz o serviço pesado de carregar o arquivo e instanciar um Layer bruto."""
        img = Image.open(path, ImageFormat.RGBA)
        return Layer(image=img, opacity=opacity, name=name, canvas=canvas)

    @classmethod
    def from_image(cls, name: str, path: str, wrap_proxy: bool = True) -> Document:
        """
        Construtor alternativo que cria o Documento baseado no tamanho de uma imagem inicial.
        O Layer criado já é injetado como a base do documento.
        """
        layer = cls.create_layer_instance(name, path)
        w, h = layer.canvas_size

        doc = cls(name=name, width=w, height=h, wrap_proxy=wrap_proxy)
        layer._canvas = doc.canvas

        doc.add_layer(layer)
        return doc

    def add_layer(self, layer: Layer | GroupLayer) -> Any:
        """
        Adiciona um Layer ou GroupLayer na pilha do documento de acordo com a política ativa.
        """
        processed_layer = self._policy.process_layer(layer, self.history)
        self.stack.append(processed_layer)
        return processed_layer

    def create_layer(self, name: str, path: str, opacity: float = 1.0) -> Any:
        """
        Fábrica oficial para carregar imagens diretamente na pilha do Documento.
        """
        layer = self.create_layer_instance(
            name, path, opacity=opacity, canvas=self.canvas)
        return self.add_layer(layer)

    def create_group(self, name: str = "Group") -> Any:
        """
        Fábrica oficial para criar e adicionar um novo Grupo (GroupLayer) na pilha do Documento.
        """
        group = GroupLayer(name=name)
        return self.add_layer(group)

    def preview(self, viewport: Viewport) -> np.ndarray:
        """
        Gera o Preview para renderizar na interface de usuário.
        """
        renderer = ViewportRender()
        result_img = renderer.render_scene(self.stack, viewport)
        return result_img[...]

    def export(self, path: str) -> None:
        """
        Renderiza a composição final em alta resolução usando as dimensões do Canvas e salva no disco.
        """
        renderer = CanvasRender()
        final_img = renderer.render_scene(self.stack, self.canvas)

        frame = final_img[...]

        if frame.shape[2] == 4:
            frame_save = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        elif frame.shape[2] == 3:
            frame_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_save = frame

        cv2.imwrite(path, frame_save)

    def get_bottom_layer(self) -> Any:
        """
        Retorna o layer raiz (fundo) da pilha.
        """
        return self.stack[-1]
