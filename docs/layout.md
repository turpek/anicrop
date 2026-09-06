# Guia de Enquadramento e Disposição Espacial — Módulo `Layout` (`layout.py`)

O módulo `anicrop.layout` fornece operações de alto nível para ajuste, enquadramento e alinhamento geométrico do `Canvas`, de camadas individuais (`Layer`) e grupos (`GroupLayer`).

---

## 1. Princípios Arquiteturais do `Layout`

- **Escopo Puramente Espacial (Moldura / Retrato)**: O `Layout` opera exclusivamente sobre a geometria lógica e enquadramento das camadas e do Canvas. **Não realiza manipulação ou destruição de pixels de imagem**.
- **Imunidade contra o "Efeito Pêndulo"**: Estratégias como `fit` e `resize_bounds` em camadas atuam definindo uma estratégia geométrica (`GeometryStrategy`, como `FitGeometry`) no controlador da camada (`GeometryController`). A `base.region` original permanece intacta, preservando a geometria estrutural e o pivô de rotação natural.
- **Projeção em Espaço Global (`global_region`)**: As operações de resolução e alinhamento utilizam a projeção em Espaço Global (`global_region` / `mat_global`), tornando os alinhamentos imunes a distorções causadas por rotações, escalas acumuladas e hierarquias de contêineres pais.
- **Polimorfismo Total**: Opera de forma transparente e uniforme sobre `Canvas`, `Layer`, `GroupLayer`, `Viewport`, `ProxyLayer` e `GroupProxy`.
- **Apelido de Tipo de Referência (`LayoutRef`)**:
  ```python
  LayoutRef = tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer
  ```
  Aceito uniformemente em todos os métodos `fit` e `align`.

---

## 2. Métodos da Classe `Layout` (`anicrop.layout.Layout`)

### `fit(target: Canvas | Layer | GroupLayer | Viewport, ref: LayoutRef) -> bool`
- **Descrição**: Enquadra a moldura ou câmera do alvo (`target`) dentro do retângulo delimitador (`ref`) preservando a proporção de aspecto original (*Aspect Ratio*).
- **Parâmetros**:
  - `target` (`Canvas | Layer | GroupLayer | Viewport`): O alvo a ser enquadrado.
  - `ref` (`LayoutRef`): Região de referência (`Region`, tupla `(x, y, w, h)`, `Canvas` ou camada).
- **Retorno**: `bool` — `True` se a geometria/câmera foi modificada.

```python
# Enquadra a foto dentro de uma caixa 800x600 centralizada no Canvas
doc.layout.fit(doc["foto"], (100, 100, 800, 600))

# Enquadra o Canvas para os limites exatos de outra camada
doc.layout.fit(doc.canvas, doc["fundo"])

# Enquadra a câmera da Viewport em uma camada específica
doc.layout.fit(viewport, doc["detalhe"])
```

---

### `align(target: Canvas | Layer | GroupLayer | Viewport, ref: LayoutRef, anchor_x: float = 0.5, anchor_y: float = 0.5) -> bool`
- **Descrição**: Alinha a posição global do alvo em relação ao retângulo de referência (`ref`), utilizando âncoras normalizadas (`0.0` a `1.0`). Na `Viewport`, desloca a visualização preservando o nível de zoom atual.
- **Parâmetros**:
  - `target` (`Canvas | Layer | GroupLayer | Viewport`): Alvo a ser alinhado.
  - `ref` (`LayoutRef`): Região de referência (ex: `doc.canvas.region` ou a `global_region` de outra camada).
  - `anchor_x` (`float`): `0.0` (esquerda), `0.5` (centro horizontal), `1.0` (direita).
  - `anchor_y` (`float`): `0.0` (topo), `0.5` (centro vertical), `1.0` (base).
- **Retorno**: `bool` — `True` se a posição/pan foi alterada.

```python
# Centraliza a camada no Canvas
doc.layout.align(doc["avatar"], doc.canvas.region, anchor_x=0.5, anchor_y=0.5)

# Alinha a câmera da Viewport no canto superior esquerdo do Canvas
doc.layout.align(viewport, doc.canvas.region, anchor_x=0.0, anchor_y=0.0)
```

---

### `pin(target: Canvas | Layer | GroupLayer | Viewport, point: tuple[float, float] | Point, anchor_x: float = 0.5, anchor_y: float = 0.5) -> bool`
- **Descrição**: Posiciona o alvo de modo que sua âncora interna (`anchor_x, anchor_y`) coincida exatamente com a coordenada global (`point`). Na `Viewport`, move a visualização (Pan) centralizando ou ancorando a janela de exibição sobre o ponto focal informado, sem necessidade de definir caixas delimitadoras artificiais.
- **Parâmetros**:
  - `target` (`Canvas | Layer | GroupLayer | Viewport`): Alvo a ser posicionado.
  - `point` (`tuple[float, float] | Point`): Coordenada global `(x, y)` no Canvas.
  - `anchor_x` (`float`): Ponto de ancoragem horizontal no alvo (`0.0` esquerda, `0.5` centro, `1.0` direita). Padrão `0.5`.
  - `anchor_y` (`float`): Ponto de ancoragem vertical no alvo (`0.0` topo, `0.5` centro, `1.0` base). Padrão `0.5`.
- **Retorno**: `bool` — `True` se a posição ou pan foi alterado.

```python
# Fixa a base inferior central do chapéu (0.5, 1.0) na coordenada da cabeça (520, 196)
chapeu.layout.pin((520, 196), anchor_x=0.5, anchor_y=1.0)

# Centraliza a câmera da Viewport diretamente no ponto focal global (anchor=(0.5, 0.5) por padrão)
viewport.layout.pin((920, 130))
```

---

### Função Auxiliar: `anchor_point(ref: LayoutRef, anchor_x: float = 0.5, anchor_y: float = 0.5) -> Point`
- **Descrição**: Função utilitária pura que calcula e retorna a coordenada global `Point(x, y)` correspondente a uma âncora normalizada em uma referência espacial (camada, grupo, Canvas, Region ou tupla).
- **Parâmetros**:
  - `ref` (`LayoutRef`): Entidade de referência espacial.
  - `anchor_x` (`float`): Fator horizontal normalizado (`0.0` a `1.0`).
  - `anchor_y` (`float`): Fator vertical normalizado (`0.0` a `1.0`).
- **Retorno**: `Point` — Coordenada `(x, y)` projetada no Espaço Global.

```python
from anicrop.layout import anchor_point

# Descobre o ponto global correspondente ao topo central de uma personagem
topo_cabeca = anchor_point(doc["personagem"], anchor_x=0.5, anchor_y=0.0)

# Posiciona um acessório diretamente nesse ponto
doc["coroa"].layout.pin(topo_cabeca, anchor_x=0.5, anchor_y=1.0)
```

---

### `resize_bounds(target: Canvas | Layer | GroupLayer | Viewport, new_width: float, new_height: float, anchor_x: float = 0.5, anchor_y: float = 0.5) -> bool`
- **Descrição**: Redimensiona a moldura lógica do alvo para as novas dimensões especificadas mantendo o alinhamento ancorado. Em camadas, **não distorce** a escala dos pixels internos. Na `Viewport`, redimensiona a janela de exibição preservando o ponto focal.
- **Parâmetros**:
  - `target` (`Canvas | Layer | GroupLayer | Viewport`): Camada, grupo, Canvas ou Viewport.
  - `new_width` (`float`): Nova largura.
  - `new_height` (`float`): Nova altura.
  - `anchor_x` (`float`): Ponto de ancoragem horizontal.
  - `anchor_y` (`float`): Ponto de ancoragem vertical.
- **Retorno**: `bool` — `True` se as dimensões foram modificadas.

```python
doc.layout.resize_bounds(doc.canvas, 1920, 1080, anchor_x=0.5, anchor_y=0.5)
viewport.layout.resize_bounds(1280, 720, anchor_x=0.5, anchor_y=0.5)
```

---

### `fit_content(target: Canvas | Layer | GroupLayer | Viewport, container: Container | Sequence[Layer] | None = None) -> bool`
- **Descrição**: 
  - **Para `Layer`**: Ajusta a moldura da camada para englobar perfeitamente o conteúdo visível/não-transparente analisado via canal alpha (`calculate_content_rect`).
  - **Para `Canvas`**: Redimensiona e reposiciona o Canvas para envolver a união do conteúdo visível das camadas contidas no `container` fornecido (ex: `doc.stack` ou lista de camadas).
  - **Para `Viewport`**: Enquadra a câmera no `Canvas` associado quando `container=None`, ou na ROI do conteúdo do `container` intersectada com os limites do Canvas.
- **Parâmetros**:
  - `target`: O `Layer`, `GroupLayer`, `Canvas` ou `Viewport` alvo.
  - `container`: Obrigatório para `Canvas`; opcional para `Viewport`.
- **Retorno**: `bool` — `True` se os limites ou enquadramento foram ajustados.

```python
# Ajusta o Canvas para englobar todas as camadas visíveis do documento:
doc.layout.fit_content(doc.canvas, doc.stack)

# Ajusta a moldura da camada ao seu conteúdo não-transparente:
doc.layout.fit_content(doc["logo"])

# Enquadra a câmera da Viewport no Canvas inteiro:
viewport.layout.fit_content()

# Enquadra a câmera no conteúdo visível da pilha:
viewport.layout.fit_content(doc.stack)
```

---

## 3. Acesso Direto via `Document`, Camadas e `Viewport`

```python
doc = Document("Cena", 1920, 1080)
doc.load_layer("avatar.png", name="avatar")
viewport = Viewport((800, 600), canvas=doc.canvas)

# Via Fachada Document:
doc.layout.align(doc["avatar"], doc.canvas.region, anchor_x=0.5, anchor_y=0.5)

# Diretamente pela Camada:
doc["avatar"].layout.align(doc.canvas.region, anchor_x=0.5, anchor_y=0.5)

# Diretamente pela Viewport:
viewport.layout.fit(doc.canvas)
viewport.layout.fit_content()
viewport.layout.align(doc["avatar"], anchor_x=0.5, anchor_y=0.5)
```
