# Guia de Enquadramento e Disposição Espacial — Módulo `Layout` (`layout.py`)

O módulo `anicrop.layout` fornece operações de alto nível para ajuste, enquadramento e alinhamento geométrico do `Canvas`, de camadas individuais (`Layer`) e grupos (`GroupLayer`).

---

## 1. Princípios Arquiteturais do `Layout`

- **Escopo Puramente Espacial (Moldura / Retrato)**: O `Layout` opera exclusivamente sobre a geometria lógica e enquadramento das camadas e do Canvas. **Não realiza manipulação ou destruição de pixels de imagem**.
- **Imunidade contra o "Efeito Pêndulo"**: Estratégias como `fit` e `resize_bounds` em camadas atuam definindo uma estratégia geométrica (`GeometryStrategy`, como `FitGeometry`) no controlador da camada (`GeometryController`). A `base.region` original permanece intacta, preservando a geometria estrutural e o pivô de rotação natural.
- **Projeção em Espaço Global (`global_region`)**: As operações de resolução e alinhamento utilizam a projeção em Espaço Global (`global_region` / `mat_global`), tornando os alinhamentos imunes a distorções causadas por rotações, escalas acumuladas e hierarquias de contêineres pais.
- **Polimorfismo Total**: Opera de forma transparente e uniforme sobre `Canvas`, `Layer`, `GroupLayer`, `ProxyLayer` e `GroupProxy`.

---

## 2. Métodos da Classe `Layout` (`anicrop.layout.Layout`)

### `fit(target: Canvas | Layer | GroupLayer, ref: Region | tuple[int, int, int, int] | Canvas | BaseLayer) -> bool`
- **Descrição**: Enquadra a moldura do alvo (`target`) dentro do retângulo delimitador (`ref`) preservando a proporção de aspecto original (*Aspect Ratio*).
- **Parâmetros**:
  - `target` (`Canvas | Layer | GroupLayer`): O alvo a ser enquadrado.
  - `ref`: Região de referência (`Region`, tupla `(x, y, w, h)`, `Canvas` ou camada).
- **Retorno**: `bool` — `True` se a geometria foi modificada.

```python
# Enquadra a foto dentro de uma caixa 800x600 centralizada no Canvas
doc.layout.fit(doc["foto"], (100, 100, 800, 600))

# Enquadra o Canvas para os limites exatos de outra camada
doc.layout.fit(doc.canvas, doc["fundo"])
```

---

### `align(target: Canvas | Layer | GroupLayer, ref: Region | tuple[int, int, int, int] | Canvas | BaseLayer, anchor_x: float = 0.5, anchor_y: float = 0.5) -> bool`
- **Descrição**: Alinha a posição global do alvo em relação ao retângulo de referência (`ref`), utilizando âncoras normalizadas (`0.0` a `1.0`).
- **Parâmetros**:
  - `target` (`Canvas | Layer | GroupLayer`): Alvo a ser alinhado.
  - `ref`: Região de referência (ex: `doc.canvas.region` ou a `global_region` de outra camada).
  - `anchor_x` (`float`): `0.0` (esquerda), `0.5` (centro horizontal), `1.0` (direita).
  - `anchor_y` (`float`): `0.0` (topo), `0.5` (centro vertical), `1.0` (base).
- **Retorno**: `bool` — `True` se a posição foi alterada.

```python
# Centraliza a camada no Canvas
doc.layout.align(doc["avatar"], doc.canvas.region, anchor_x=0.5, anchor_y=0.5)
```

---

### `resize_bounds(target: Canvas | Layer | GroupLayer, new_width: int, new_height: int, anchor_x: float = 0.5, anchor_y: float = 0.5) -> bool`
- **Descrição**: Redimensiona a moldura lógica do alvo para as novas dimensões especificadas mantendo o alinhamento ancorado. Em camadas, **não distorce** a escala dos pixels internos.
- **Parâmetros**:
  - `target` (`Canvas | Layer | GroupLayer`): Camada, grupo ou Canvas.
  - `new_width` (`int`): Nova largura em pixels.
  - `new_height` (`int`): Nova altura em pixels.
  - `anchor_x` (`float`): Ponto de ancoragem horizontal.
  - `anchor_y` (`float`): Ponto de ancoragem vertical.
- **Retorno**: `bool` — `True` se as dimensões foram modificadas.

```python
doc.layout.resize_bounds(doc.canvas, 1920, 1080, anchor_x=0.5, anchor_y=0.5)
```

---

### `fit_content(target: Canvas | Layer | GroupLayer, container: Container | Sequence[Layer] | None = None) -> bool`
- **Descrição**: 
  - **Para `Layer`**: Ajusta a moldura da camada para englobar perfeitamente o conteúdo visível/não-transparente analisado via canal alpha (`calculate_content_rect`).
  - **Para `Canvas`**: Redimensiona e reposiciona o Canvas para envolver a união do conteúdo visível das camadas contidas no `container` fornecido (ex: `doc.stack` ou lista de camadas).
- **Parâmetros**:
  - `target`: O `Layer`, `GroupLayer` ou `Canvas` alvo.
  - `container`: Obrigatório quando o alvo for um `Canvas`.
- **Retorno**: `bool` — `True` se os limites foram ajustados.

```python
# Ajusta o Canvas para englobar todas as camadas visíveis do documento:
doc.layout.fit_content(doc.canvas, doc.stack)

# Ajusta a moldura da camada ao seu conteúdo não-transparente:
doc.layout.fit_content(doc["logo"])
```

---

## 3. Acesso Direto via `Document` e Camadas

```python
doc = Document("Cena", 1920, 1080)
doc.load_layer("avatar.png", name="avatar")

# Via Fachada Document:
doc.layout.align(doc["avatar"], doc.canvas.region, anchor_x=0.5, anchor_y=0.5)

# Diretamente pela Camada:
doc["avatar"].layout.align(doc.canvas.region, anchor_x=0.5, anchor_y=0.5)
```
