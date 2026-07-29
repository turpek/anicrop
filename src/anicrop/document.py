from __future__ import annotations
import numpy as np

from anicrop.canvas import Canvas
from anicrop.history import GlobalHistory
from anicrop.container import LayerStack
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.proxy import ProxyLayer
from anicrop.render import CanvasRender, ViewportRender
from anicrop.viewport import Viewport


class Document:
    """
    Facade principal da biblioteca.
    Gerencia o Canvas, o Histórico Global e a Pilha de Camadas (LayerStack).
    """

    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.canvas = Canvas(width, height)
        self.history = GlobalHistory()
        self.stack = LayerStack()

    @staticmethod
    def create_layer_instance(name: str, path: str, opacity: float = 1.0, canvas: Canvas | None = None) -> Layer:
        """Helper estático que faz o serviço pesado de carregar o arquivo e instanciar um Layer bruto."""
        from anicrop.enums import ImageFormat
        img = Image.open(path, ImageFormat.RGBA)
        return Layer(image=img, opacity=opacity, name=name, canvas=canvas)

    @classmethod
    def from_image(cls, name: str, path: str) -> "Document":
        """
        Construtor alternativo que cria o Documento baseado no tamanho de uma imagem inicial.
        O Layer criado já é injetado como a base do documento.
        """
        # Cria o layer de forma independente (ainda sem canvas)
        layer = cls.create_layer_instance(name, path)
        w, h = layer.canvas_size

        # Instancia o documento usando a dimensão da imagem
        doc = cls(name=name, width=w, height=h)

        # Conecta o canvas no layer (para que matrizes globais funcionem)
        layer._canvas = doc.canvas

        # Adiciona o layer no documento pedindo envelopamento
        doc.add_layer(layer, wrap_proxy=True)
        return doc

    def add_layer(self, layer: Layer, wrap_proxy: bool = True) -> Layer:
        """
        Adiciona um Layer existente na pilha do documento.
        Se wrap_proxy for True, o Layer é devolvido envelopado para rastrear o histórico.
        """
        self.stack.append(layer)
        if wrap_proxy:
            return ProxyLayer(layer, self.history)
        return layer

    def create_layer(self, name: str, path: str, opacity: float = 1.0, wrap_proxy: bool = True) -> Layer:
        """
        Fábrica oficial para carregar imagens diretamente na pilha do Documento.
        """
        layer = self.create_layer_instance(
            name, path, opacity=opacity, canvas=self.canvas)
        return self.add_layer(layer, wrap_proxy=wrap_proxy)

    def preview(self, viewport: Viewport) -> np.ndarray:
        """
        Gera o Preview para renderizar na interface de usuário.
        """
        renderer = ViewportRender()
        # ViewportRender devolve um objeto Image. Extraímos o ndarray chamando o slicing [...]
        result_img = renderer.render_scene(self.stack, viewport)
        return result_img[...]

    def export(self, path: str) -> None:
        """
        Renderiza a composição final em alta resolução usando as dimensões do Canvas e salva no disco.
        """
        renderer = CanvasRender()
        final_img = renderer.render_scene(self.stack, self.canvas)

        frame = final_img[...]
        import cv2

        if frame.shape[2] == 4:
            frame_save = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        elif frame.shape[2] == 3:
            frame_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_save = frame

        cv2.imwrite(path, frame_save)

    def get_bottom_layer(self, wrap_proxy: bool = True) -> Layer:
        """
        Retorna o layer raiz (fundo) da pilha.
        Se wrap_proxy for True, devolve protegido pelo histórico.
        """
        layer = self.stack[-1]
        if wrap_proxy:
            return ProxyLayer(layer, self.history)
        return layer
