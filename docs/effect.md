# Guia do Sistema de Efeitos e Filtros (`anicrop.effect` / `anicrop.filter`)

O subsistema de efeitos do `anicrop` fornece uma arquitetura desacoplada e não-destrutiva para pós-processamento de imagens e camadas, com suporte a margens de expansão geométrica (*padding*), adaptação à rotação e escala da cena (*transform-aware*), modulação por máscaras de transparência e fusão analítica de filtros contínuos.

---

## 1. Arquitetura Geral

No `anicrop`, o pós-processamento gráfico é dividido em três pilares:

```
                  ┌──────────────────────────────────────────────┐
                  │             BaseLayer / Layer                │
                  │   - self.effects: list[Effect]               │
                  │   - self.mask: Mask | None                   │
                  └──────────────────────┬───────────────────────┘
                                         │
                        Executado no render_post_processing
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          BoundEffect (Envelope)              │
                  │   - self.matrix: Matriz inversa da camada    │
                  │   - self.mask: Máscara local opcional        │
                  │   - Calcula: ΔM = M_render @ M_base_inv      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         Effect Puro (ex: BlurFilter)         │
                  │   - get_padding() -> Margem de expansão      │
                  │   - apply(image, delta_matrix) -> Image      │
                  │   - merge(other, matrix) -> Effect unificado │
                  └──────────────────────────────────────────────┘
```

1. **Classe Abstrata Base (`Effect`)**: Classes herdando de `Effect(ABC)`, com controle de visibilidade (`visible`), nome (`name`) e métodos abstratos para processamento de pixels.
2. **Envelope Geométrico (`BoundEffect`)**: Ancara um `Effect` à matriz espacial da camada e gerencia visibilidade e modulação por máscara.
3. **Gerenciamento na Camada (`BaseLayer`)**: Fila sequencial de efeitos (`layer.effects`), acessível tanto em camadas folha (`Layer`) quanto em grupos aninhados (`GroupLayer`).

---

## 2. A Classe Base Abstrata `Effect` (`anicrop.effect.Effect`)

Qualquer efeito no `anicrop` herda da classe abstrata `Effect`, fornecendo controle padronizado de visibilidade (`visible`), nome identificador (`name`) e contratos para cálculo de padding, aplicação e fusão analítica:

```python
from abc import ABC, abstractmethod
import numpy as np
from anicrop.image import Image

class Effect(ABC):
    def __init__(self, visible: bool = True, name: str = "Effect"):
        self.visible = visible
        self.name = name

    @abstractmethod
    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna a margem extra (top, right, bottom, left) em pixels.
        
        Necessária para efeitos que expandem além da borda original da camada
        (como sombras projetadas, brilho externo ou desfoque gaussiano).
        """
        pass

    @abstractmethod
    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Processa e transforma o buffer de imagem recebendo a matriz afim ativa.
        
        Args:
            image: Instância contendo os pixels a serem processados.
            matrix: Matriz afim 3x3 no espaço de renderização para adaptar
                    direção, rotação ou escala do efeito.
        """
        pass

    @abstractmethod
    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Tenta combinar analiticamente este efeito com outro para otimização.
        
        Retorna uma nova instância combinada ou None caso não seja possível fundir.
        """
        pass
```

---

## 3. O Envelope Geométrico: `BoundEffect` (`anicrop.effect.BoundEffect`)

O `BoundEffect` decora um `Effect` puro, ligando-o à geometria da camada:

- **Matriz de Ancoragem**: Armazena a matriz inversa da camada no momento da vinculação (`mat_inverse(mat_global(layer))`).
- **Delta Espacial**: No método `apply`, calcula $\Delta M = M_{\text{render}} \cdot M_{\text{ancoragem}}$ e passa para o efeito interno, permitindo que efeitos direcionais (como desfoque em ângulo ou sombras) acompanhem a rotação da camada ou da câmera.
- **Modulação por Máscara**: Se uma máscara (`Mask`) for associada, o resultado do efeito é modulado pixel a pixel via `mask.modulate_blend(image, filtered)`.
- **Visibilidade**: Respeita a flag `visible: bool`. Se `False`, retorna a imagem original sem processamento e padding zero.

---

## 4. Filtro Concreto: `BlurFilter` (`anicrop.filter.BlurFilter`)

O `BlurFilter` é a implementação nativa de desfoque de alta performance do `anicrop`:

```python
from anicrop.filter import BlurFilter
from anicrop.enums import BlurMode

blur = BlurFilter(
    radius=5.0,                  # Raio em pixels (ou tupla (rx, ry) para desfoque anisotrópico)
    angle=45.0,                  # Ângulo de inclinação em graus (desfoque direcional)
    mode=BlurMode.GAUSSIAN,       # GAUSSIAN, BOX ou MEDIAN
    affect_alpha=True,           # Se True, expande e desfoca o canal de transparência
    strength=1.0,                # Intensidade da mesclagem (0.0 a 1.0)
    name="GaussianBlur",
)
```

### Principais Recursos do `BlurFilter`:
1. **Desfoque Anisotrópico e Direcional:**
   - Suporta raios distintos em X e Y (`radius=(12.0, 2.0)`).
   - Com `angle != 0.0`, calcula dinamicamente um kernel 2D rotacionado por matriz afim, gerando efeito de velocidade (*motion blur*) perfeito em qualquer inclinação contínua.
2. **Padding Automático Inteligente:**
   - Expande a área de renderização da camada em $3\sigma$ (gaussiano) ou $1\times$ raio (box), evitando que as bordas do desfoque sejam cortadas abruptamente na moldura da camada.
3. **Fusão Analítica de Covariâncias (`merge`):**
   - Ao renderizar dois filtros gaussianos acumulados (ex: desfoque da camada + desfoque do grupo pai), o `anicrop` **não** aplica dois passes lentos de convolução na imagem.
   - O método `merge()` soma analiticamente os tensores de covariância no plano $\Sigma_{\text{total}} = \Sigma_1 + \Sigma_2$ e decompõe os autovalores e autovetores para gerar um **único filtro gaussiano equivalente**, preservando precisão e velocidade máxima.

---

## 5. Como Gerenciar Efeitos na Camada (`BaseLayer` / `Layer` / `GroupLayer`)

Todos os métodos de controle de efeitos estão presentes em `BaseLayer`:

```python
import numpy as np
from anicrop import Layer, Image, ImageFormat, Region
from anicrop.filter import BlurFilter
from anicrop.enums import BlurMode

# 1. Cria a camada
img = Image.new((400, 300), ImageFormat.RGBA, color=(255, 0, 0, 255))
layer = Layer(img)

# 2. Adiciona um efeito direto à fila:
blur = BlurFilter(radius=8.0, mode=BlurMode.GAUSSIAN)
layer.add_effect(blur)

# 3. Ou vincula como BoundEffect com máscara local:
mask_img = Image.new((400, 300), ImageFormat.GRAY, color=128)
mask = layer.set_mask(mask_img, Region.from_size(400, 300))
bound_blur = layer.bind_effect(blur, mask=mask, visible=True)

# 4. Inspeciona os efeitos ativos:
print(layer.effects)  # Retorna tupla imutável com os efeitos

# 5. Consulta o padding total somado dos efeitos:
top, right, bottom, left = layer.get_effects_padding()

# 6. Remove um efeito específico ou limpa todos:
layer.remove_effect(blur)
layer.clear_effects()
```

---

## 6. Criação de Efeitos Customizados

Criar novos efeitos para o `anicrop` é direto: basta herdar da classe `Effect`, chamar `super().__init__(visible=visible, name=name)` e implementar os métodos obrigatórios:

### Exemplo: Filtro de Ajuste de Brilho e Contraste

```python
from __future__ import annotations
import numpy as np
from anicrop.effect import Effect
from anicrop.image import Image

class BrightnessContrastEffect(Effect):
    """Ajusta o brilho e contraste da camada de forma não-destrutiva."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        visible: bool = True,
        name: str = "BrightnessContrast",
    ):
        super().__init__(visible=visible, name=name)
        self.brightness = brightness  # Deslocamento [-255, 255]
        self.contrast = contrast      # Multiplicador [0.0, ...]

    def get_padding(self) -> tuple[int, int, int, int]:
        # Este efeito não expande a geometria dos pixels
        return (0, 0, 0, 0)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        # Acessa os pixels originais
        data = image[...].astype(np.float32)

        # Aplica transformação de cor nos canais RGB (preservando o canal Alfa intacto)
        rgb = data[..., :3] * self.contrast + self.brightness
        data[..., :3] = np.clip(rgb, 0, 255)

        return Image(data.astype(image.dtype), image.format)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        if isinstance(other, BrightnessContrastEffect):
            # Combinação matemática simples de dois ajustes lineares
            new_contrast = self.contrast * other.contrast
            new_brightness = self.brightness * other.contrast + other.brightness
            return BrightnessContrastEffect(
                new_brightness, new_contrast, visible=self.visible, name=self.name
            )
        return None

# Uso na camada:
effect = BrightnessContrastEffect(brightness=20.0, contrast=1.2)
layer.add_effect(effect)
```

---

## 7. Pipeline de Execução e Isolamento de Buffer durante a Renderização

Durante o `CanvasRender` ou `ViewportRender`:

1. **Detecção de Pós-Processamento Ativo (`has_active_post_processing`):**
   - O renderizador avalia se a camada possui ao menos um `effect` com `visible is True` ou uma `mask` com `visible is True`.
2. **Isolamento de Buffer Não-Destrutivo:**
   - No Fast-Path 1 de renderização (quando uma camada não possui distorção de escala/rotação e cobre 100% do frame), se houver pós-processamento ativo, o renderizador desvincula os dados físicos chamando `edit_image.crop()`.
   - Isso garante imunidade absoluta: mesmo se um efeito customizado ou filtro alterar canais in-place (como `image[..., -1] = 0`), a imagem original persistida no `EditLayer` permanece 100% intacta para renderizações subsequentes.
3. **Pós-Processamento (`apply_post_processing`):**
   - Itera pelos efeitos ativos (`effect.visible is True`) chamando `effect.apply(image, frame.matrix)`.
   - Se houver máscara ativa (`base.mask.visible is True`), rasteriza e modula o canal de opacidade da camada com `base.mask.apply_modulation`.
4. O resultado pós-processado é então mesclado no buffer de destino com a opacidade e modo de mesclagem (`blend_mode`) da camada.
