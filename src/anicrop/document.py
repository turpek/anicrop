from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, TypeVar, overload

from anicrop.canvas import Canvas
from anicrop.composition import Combine
from anicrop.container import BaseLayer, Container, GroupLayer, LayerStack, NullContainer
from anicrop.content import Content
from anicrop.enums import BlendMode, ImageFormat, InterpMode
from anicrop.history import GlobalHistory
from anicrop.image import Image
from anicrop.interfaces.io import AbstractImageIO, SaveOptions
from anicrop.layer import Layer
from anicrop.layout import Layout
from anicrop.proxy import BaseHistoryProxy, GroupProxy, LayerStackProxy, ProxyLayer
from anicrop.render import CanvasRender, ViewportRender
from anicrop.viewport import Viewport

LayerT = TypeVar("LayerT", bound=BaseLayer)


class DocumentPolicy(ABC):
    @abstractmethod
    def setup(self) -> tuple[GlobalHistory | None, LayerStack]:
        pass

    @abstractmethod
    def process_layer(self, layer: LayerT, history: GlobalHistory | None) -> LayerT:
        pass


class ReactiveDocumentPolicy(DocumentPolicy):
    """Política com Histórico e Proxies ativados."""

    def setup(self) -> tuple[GlobalHistory, LayerStack]:
        history = GlobalHistory()
        stack = LayerStackProxy(LayerStack(), history)
        return history, stack  # type: ignore[return-value]

    def process_layer(self, layer: LayerT, history: GlobalHistory | None) -> LayerT:
        if isinstance(layer, BaseHistoryProxy):
            return layer
        assert history is not None
        if isinstance(layer, GroupLayer):
            return GroupProxy(layer, history)  # type: ignore[return-value]
        return ProxyLayer(layer, history)  # type: ignore[return-value]


class DirectDocumentPolicy(DocumentPolicy):
    """Política de alta performance sem Histórico e sem Proxies (modo direto)."""

    def setup(self) -> tuple[None, LayerStack]:
        return None, LayerStack()

    def process_layer(self, layer: LayerT, history: GlobalHistory | None) -> LayerT:
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

    def __init__(
        self,
        name: str,
        width: float,
        height: float,
        history: bool = False,
        bg_color: tuple[int, ...] | None = None,
    ):
        """
        Inicializa um novo Documento 2D.

        Args:
            name: Nome identificador do documento.
            width: Largura do canvas em pixels.
            height: Altura do canvas em pixels.
            history: Habilita o rastreamento de histórico (Undo/Redo) via proxies reativos.
                Nota: Funcionalidade experimental sob refatoração; operações avançadas
                de árvore (como Combine) ainda não gravam passos no histórico. Padrão: False.
            bg_color: Cor de fundo opcional do canvas.
        """
        self.name = name
        self.canvas = Canvas.from_size(width, height, bg_color=bg_color)
        self.history_enabled = history

        self._policy = self._POLICIES[history]
        self.history, self.stack = self._policy.setup()

        self._viewport_render = ViewportRender()
        self._canvas_render = CanvasRender()
        self._layout = Layout()
        self._content = Content()
        self._combine = Combine(self)

    @classmethod
    def open(
        cls,
        path: str | Path,
        name: str,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        history: bool = False,
        format: ImageFormat = ImageFormat.RGBA,
        bg_color: tuple[int, ...] | None = None,
        backend: AbstractImageIO | str | None = None,
    ) -> Document:
        """
        Abre uma imagem do disco e cria um Documento baseado no seu tamanho,
        inserindo a imagem como primeira camada.

        Args:
            path: Caminho do arquivo de imagem.
            name: Nome da camada e do documento.
            opacity: Opacidade inicial da camada (0.0 a 1.0).
            blend_mode: Modo de mesclagem da camada.
            history: Habilita o histórico experimental (padrão: False).
            format: Formato de cor da imagem.
            bg_color: Cor de fundo opcional do canvas.
            backend: Backend de I/O a ser utilizado.
        """
        img = Image.open(str(path), format, backend=backend)
        layer = Layer(img, opacity=opacity, blend_mode=blend_mode, name=name)
        w, h = layer.region.size

        doc = cls(name=name, width=w, height=h, history=history, bg_color=bg_color)
        doc.add(layer)
        return doc

    @property
    def layout(self) -> Layout:
        """Instância do motor de Layout para operações espaciais no documento."""
        return self._layout

    @property
    def content(self) -> Content:
        """Instância do motor de manipulação de conteúdo/pixels do documento."""
        return self._content

    @property
    def combine(self) -> Combine:
        """Instância do serviço de combinação/fusão de camadas no documento."""
        return self._combine

    @property
    def canvas_render(self) -> CanvasRender:
        """Instância do renderizador de cena CanvasRender configurado no documento."""
        return self._canvas_render

    @property
    def viewport_render(self) -> ViewportRender:
        """Instância do renderizador de viewport ViewportRender configurado no documento."""
        return self._viewport_render

    def _validate_unique_name(self, name: str) -> None:
        """Verifica se já existe alguma camada ou grupo com este nome no documento."""
        if self._find_in_container(self.stack, name, recursive=True) is not None:
            raise ValueError(f"A layer named '{name}' already exists in the document.")

    def _find_in_container(
        self, container: Container, name: str, recursive: bool = True
    ) -> BaseLayer | None:
        """Busca interna auxiliar por nome na hierarquia de um container."""
        for child in container:
            if child.name == name:
                return child
            if recursive and isinstance(child, (GroupLayer, Container)):
                found = self._find_in_container(child, name, recursive=True)
                if found is not None:
                    return found
        return None

    def add(self, layer: LayerT) -> LayerT:
        """
        Adiciona um Layer ou GroupLayer na pilha do documento.
        Garante a unicidade do nome da camada no documento.
        """
        self._validate_unique_name(layer.name)
        processed_layer = self._policy.process_layer(layer, self.history)
        self.stack.append(processed_layer)
        return processed_layer

    def add_group(self, name: str) -> GroupLayer:
        """
        Cria e adiciona um novo Grupo (GroupLayer) na pilha do Documento com nome obrigatório.
        """
        self._validate_unique_name(name)
        group = GroupLayer(name=name)
        return self.add(group)  # type: ignore[return-value]

    def load_layer(
        self,
        path: str | Path,
        name: str,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        format: ImageFormat = ImageFormat.RGBA,
        backend: AbstractImageIO | str | None = None,
    ) -> Layer:
        """
        Carrega uma imagem do disco e adiciona como camada na pilha do documento com nome obrigatório.
        """
        img = Image.open(str(path), format, backend=backend)
        layer = Layer(img, opacity=opacity, blend_mode=blend_mode, name=name)
        return self.add(layer)  # type: ignore[return-value]

    def find(self, name: str, recursive: bool = True) -> BaseLayer | None:
        """
        Busca uma camada pelo nome no documento. Retorna None se não encontrar.
        """
        return self._find_in_container(self.stack, name, recursive=recursive)

    def __len__(self) -> int:
        """Retorna a quantidade de camadas na raiz da pilha."""
        return len(self.stack)

    def __iter__(self) -> Iterator[BaseLayer]:
        """Itera pelas camadas raiz da pilha."""
        return iter(self.stack)

    @overload
    def __getitem__(self, key: int | str) -> BaseLayer:
        pass

    @overload
    def __getitem__(self, key: slice) -> list[BaseLayer]:
        pass

    def __getitem__(self, key: int | slice | str) -> BaseLayer | list[BaseLayer]:
        """
        Acesso polimórfico a camadas por índice inteiro, slice ou nome (string).
        """
        if isinstance(key, int):
            return self.stack[key]
        if isinstance(key, slice):
            return list(self.stack)[key]
        if isinstance(key, str):
            layer = self.find(key, recursive=True)
            if layer is None:
                raise KeyError(f"Layer named '{key}' not found in document.")
            return layer
        raise TypeError(
            f"Invalid key type {type(key).__name__}. Expected int, slice, or str."
        )

    def __contains__(self, item: BaseLayer | str) -> bool:
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
            self.remove(key)
        else:
            raise TypeError(
                f"Invalid key type {type(key).__name__}. Expected int or str."
            )

    def remove(self, layer_or_name: BaseLayer | str) -> None:
        """
        Remove uma camada da pilha (aceita a instância da camada ou seu nome).
        """
        if isinstance(layer_or_name, str):
            found = self.find(layer_or_name, recursive=True)
            if found is None:
                raise KeyError(f"Layer named '{layer_or_name}' not found in document.")
            layer = found
        else:
            layer = layer_or_name

        if layer in self.stack:
            self.stack.remove(layer)
            return

        if not isinstance(layer.parent, NullContainer):
            layer.parent.remove(layer)
        else:
            raise ValueError(f"Layer {layer} not found in document hierarchy.")

    def render(
        self,
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
    ) -> Image:
        """
        Renderiza a composição final no formato especificado e retorna o objeto Image.
        """
        return self._canvas_render.render_scene(
            self.stack, self.canvas, format=format, interp=interp
        )

    def preview(
        self,
        viewport: Viewport,
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
    ) -> Image:
        """
        Gera o Preview para renderizar na interface de usuário via Viewport e retorna um objeto Image.
        """
        return self._viewport_render.render_scene(
            self.stack, viewport, format=format, interp=interp
        )

    def export(
        self,
        path: str | Path,
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
        options: SaveOptions | None = None,
        backend: AbstractImageIO | str | None = None,
    ) -> None:
        """
        Renderiza a composição final no formato especificado e salva no disco.
        """
        self.render(format=format, interp=interp).save(
            path, options=options, backend=backend
        )
