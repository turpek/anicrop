from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
import numpy as np
from anicrop.spatial import Region
from anicrop.interfaces.layout import LayoutStrategy

if TYPE_CHECKING:
    from anicrop.edit_layer import EditLayer
    from anicrop.container import NullContainer
    from anicrop.interfaces.container import AbstractContainer
    from anicrop.interfaces.content import ContentStrategy


class AbstractBaseLayer(ABC):
    """Classe base abstrata para todos os elementos gráficos da hierarquia."""

    parent: AbstractContainer | NullContainer
    _parent_inverse: np.ndarray
    opacity: float
    visible: bool
    blend_mode: Any
    name: str

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Matriz de transformação ativa do nó."""
        ...

    @property
    @abstractmethod
    def region(self) -> Region:
        """Região delimitadora local."""
        ...

    @property
    @abstractmethod
    def global_region(self) -> Region:
        """Região delimitadora projetada no espaço global."""
        ...

    @property
    @abstractmethod
    def transform(self) -> Any:
        """Compositor de transformações geométricas da camada."""
        ...

    @property
    @abstractmethod
    def layout(self) -> LayoutStrategy:
        """Estratégia de layout vinculada à moldura da camada."""
        ...

    @property
    @abstractmethod
    def base(self) -> Any:
        """Estratégia geométrica estrutural da base."""
        ...

    @property
    @abstractmethod
    def frame(self) -> Any:
        """Estratégia geométrica ativa da moldura."""
        ...

    @property
    @abstractmethod
    def canvas_size(self) -> tuple[int, int]:
        """Tamanho de canvas/moldura de referência."""
        ...

    @property
    @abstractmethod
    def mask(self) -> Any:
        """Máscara ativa da camada."""
        ...

    @property
    @abstractmethod
    def effects(self) -> tuple[Any, ...]:
        """Fila de efeitos de pós-processamento da camada."""
        ...

    @abstractmethod
    def add_effect(self, effect: Any) -> Any:
        """Adiciona um efeito à camada."""
        ...

    @abstractmethod
    def remove_effect(self, effect: Any) -> None:
        """Remove um efeito da camada."""
        ...

    @abstractmethod
    def clear_effects(self) -> None:
        """Remove todos os efeitos da camada."""
        ...

    @abstractmethod
    def set_mask(self, *args: Any, **kwargs: Any) -> Any:
        """Cria e vincula uma máscara à camada."""
        ...

    @abstractmethod
    def remove_mask(self) -> None:
        """Remove a máscara ativa da camada."""
        ...


class AbstractLayer(AbstractBaseLayer):
    """Classe base abstrata para camadas folha com pixels e edições locais."""

    @property
    @abstractmethod
    def edits(self) -> tuple[EditLayer, ...]:
        """Coleção de edições e patches locais da camada."""
        ...

    @property
    @abstractmethod
    def content(self) -> ContentStrategy:
        """Gerenciador de manipulação, transformação e ajuste de pixels/conteúdo."""
        ...

    @property
    @abstractmethod
    def image(self) -> Any:
        """Imagem base da camada."""
        ...

    @property
    @abstractmethod
    def format(self) -> Any:
        """Formato de cor da imagem base."""
        ...

    @abstractmethod
    def add_edit(self, *args: Any, **kwargs: Any) -> Any:
        """Adiciona um patch/edição à camada."""
        ...
