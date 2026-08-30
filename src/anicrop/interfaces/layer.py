from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from anicrop.interfaces.layout import LayoutStrategy
from anicrop.spatial import Region

if TYPE_CHECKING:
    from anicrop.container import NullContainer
    from anicrop.edit_layer import EditLayer
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
        pass

    @property
    @abstractmethod
    def region(self) -> Region:
        """Região delimitadora local."""
        pass

    @property
    @abstractmethod
    def global_region(self) -> Region:
        """Região delimitadora projetada no espaço global."""
        pass

    @property
    @abstractmethod
    def transform(self) -> Any:
        """Compositor de transformações geométricas da camada."""
        pass

    @property
    @abstractmethod
    def layout(self) -> LayoutStrategy:
        """Estratégia de layout vinculada à moldura da camada."""
        pass

    @property
    @abstractmethod
    def content(self) -> ContentStrategy:
        """Gerenciador de manipulação, transformação e ajuste de pixels/conteúdo."""
        pass

    @property
    @abstractmethod
    def base(self) -> Any:
        """Estratégia geométrica estrutural da base."""
        pass

    @property
    @abstractmethod
    def frame(self) -> Any:
        """Estratégia geométrica ativa da moldura."""
        pass

    @property
    @abstractmethod
    def mask(self) -> Any:
        """Máscara ativa da camada."""
        pass

    @property
    @abstractmethod
    def is_renderable(self) -> bool:
        """Indica se o nó (camada ou grupo) deve ser processado no pipeline de renderização."""
        pass

    @property
    @abstractmethod
    def effects(self) -> tuple[Any, ...]:
        """Fila de efeitos de pós-processamento da camada."""
        pass

    @abstractmethod
    def add_effect(self, effect: Any) -> Any:
        """Adiciona um efeito à camada."""
        pass

    @abstractmethod
    def remove_effect(self, effect: Any) -> None:
        """Remove um efeito da camada."""
        pass

    @abstractmethod
    def clear_effects(self) -> None:
        """Remove todos os efeitos da camada."""
        pass

    @abstractmethod
    def set_mask(self, *args: Any, **kwargs: Any) -> Any:
        """Cria e vincula uma máscara à camada."""
        pass

    @abstractmethod
    def remove_mask(self) -> None:
        """Remove a máscara ativa da camada."""
        pass


class AbstractLayer(AbstractBaseLayer):
    """Classe base abstrata para camadas folha com pixels e edições locais."""

    @property
    @abstractmethod
    def edits(self) -> tuple[EditLayer, ...]:
        """Coleção de edições e patches locais da camada."""
        pass

    @property
    @abstractmethod
    def format(self) -> Any:
        """Formato de cor da camada."""
        pass

    @format.setter
    @abstractmethod
    def format(self, value: Any) -> None:
        """Define o formato de cor da camada."""
        pass

    @abstractmethod
    def add_edit(self, *args: Any, **kwargs: Any) -> Any:
        """Adiciona um patch/edição à camada."""
        pass
