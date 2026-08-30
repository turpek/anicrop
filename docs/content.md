# Guia de Manipulação de Conteúdo e Pixels — Módulo `Content` (`content.py`)

O módulo `anicrop.content` fornece operações de alto nível para transformação e manipulação direta de conteúdo em camadas (`Layer`) e grupos de camadas (`GroupLayer`), operando sobre matrizes afins, patches não-destrutivos (`BlendMode.CLIP`) e máscaras de corte.

---

## 1. Princípios Arquiteturais do `Content`

- **Escopo sobre o Conteúdo**: Diferente do módulo `Layout` (que gerencia molduras e enquadramentos de nós), o `Content` manipula a escala visual, espelhamentos, recortes e dimensões efetivas do elemento.
- **Simetria com `Layout`**:
  - `ContentStrategy` (Protocol / Interface base em `anicrop.interfaces.content`)
  - `BaseContentStrategy` (Lógica compartilhada de escala, ajuste e espelhamento)
  - `LayerContentStrategy` (Estratégia para `Layer`, com `crop` via patch `BlendMode.CLIP`)
  - `GroupContentStrategy` (Estratégia para `GroupLayer`, com `crop` via máscara de grupo `set_mask`)
- **Corte Não-Destrutivo**:
  - Em `Layer`: A operação `crop` ajusta a moldura e injeta um patch de máscara `EditLayer` com `BlendMode.CLIP`, preservando $100\%$ dos pixels originais.
  - Em `GroupLayer`: A operação `crop` ajusta a moldura do grupo e aplica uma `Mask` sólida retangular, modulando a visibilidade de todos os filhos sem rasterização prematura.
- **Transformações Afins de Alta Precisão**: Operações como `resize`, `flip_x`, `flip_y` e `fit` alteram diretamente o compositor afim (`target.transform`), propagando-se perfeitamente para nós filhos em grupos.

---

## 2. Métodos da Classe `Content` (`anicrop.content.Content`)

### `crop(target: BaseLayer, ref: Region | tuple[int, int, int, int] | Canvas | BaseLayer) -> bool`
- **Descrição**: Recorta o conteúdo visual da camada ou grupo para os limites especificados em `ref`.
- **Parâmetros**:
  - `target` (`BaseLayer`): A camada folha (`Layer`) ou grupo (`GroupLayer`) a ser recortado.
  - `ref`: Limites do recorte (tupla `(x, y, w, h)`, `Region`, `Canvas` ou outra camada).
- **Retorno**: `bool` — `True` se o recorte foi aplicado com sucesso.

```python
# Recorta uma camada para uma caixa 400x300 na posição (50, 50)
doc.content.crop(doc["foto"], (50, 50, 400, 300))

# Recorta um grupo inteiro para a área do Canvas
doc.content.crop(doc["grupo_personagens"], doc.canvas)
```

---

### `resize(target: BaseLayer, width: int, height: int) -> bool`
- **Descrição**: Redimensiona a escala visual do conteúdo da camada ou grupo para atingir a largura e altura especificadas no Espaço Global.
- **Parâmetros**:
  - `target` (`BaseLayer`): A camada ou grupo alvo.
  - `width` (`int`): Nova largura em pixels (deve ser > 0).
  - `height` (`int`): Nova altura em pixels (deve ser > 0).
- **Retorno**: `bool` — `True` se as dimensões foram modificadas.

```python
doc.content.resize(doc["foto"], 1280, 720)
doc.content.resize(doc["grupo_cenario"], 1920, 1080)
```

---

### `fit(target: BaseLayer, ref: Region | tuple | Canvas | BaseLayer) -> bool`
- **Descrição**: Ajusta a escala e translada o conteúdo da camada ou grupo para preencher exatamente a região de referência `ref`.
- **Sobrecargas (`@ovld`)**:
  - `fit(target, ref)`: Ajuste direto para uma região ou elemento.
  - `fit(payload)`: Recebe uma tupla gerada por um contexto de ajuste proporcional (como `FitContext`).

```python
# Ajusta o conteúdo do grupo para cobrir exatamente a região do Canvas
doc.content.fit(doc["grupo_elementos"], doc.canvas)
```

---

### `flip_x(target: BaseLayer) -> bool` / `flip_y(target: BaseLayer) -> bool`
- **Descrição**: Espelha o conteúdo da camada ou grupo horizontalmente (`flip_x`) ou verticalmente (`flip_y`) invertendo o sinal da escala afim (`scale(-1, 1)` / `scale(1, -1)`).
- **Parâmetros**:
  - `target` (`BaseLayer`): A camada ou grupo alvo.
- **Retorno**: `bool` — `True`.

```python
# Espelha o grupo inteiro horizontalmente
doc.content.flip_x(doc["grupo_personagens"])
```

---

## 3. Ajustes Proporcionais Avançados (`FitContext`)

A classe `FitContext` permite calcular enquadramentos proporcionais complexos antes de aplicá-los em camadas ou grupos:

- **`fit_contain`**: Redimensiona proporcionalmente para caber totalmente dentro de `ref` sem cortar nada.
- **`fit_cover`**: Redimensiona proporcionalmente para cobrir totalmente `ref` (com corte das sobras).
- **`scale_width`**: Ajusta proporcionalmente travando a largura em `ref.width`.
- **`scale_height`**: Ajusta proporcionalmente travando a altura em `ref.height`.

### Exemplo de Uso com `FitContext`:
```python
from anicrop.content import FitContext

# Calcula o enquadramento proporcional "contain" centralizado para um grupo:
ctx = FitContext(doc["grupo_logos"], doc.canvas, x_factor=0.5, y_factor=0.5)

# Aplica o resultado calculado no motor de Content:
doc.content.fit(ctx.fit_contain)
```

---

## 4. Acesso Direto via `Document`, `Layer` e `GroupLayer`

Você pode acessar as operações de conteúdo tanto através da fachada do documento quanto diretamente na propriedade `.content` de qualquer camada ou grupo:

```python
# Via Fachada Document:
doc.content.resize(doc["foto"], 800, 600)
doc.content.flip_x(doc["grupo_cenario"])

# Via Propriedade de Camada / Grupo:
layer = doc["foto"]
layer.content.resize(800, 600)
layer.content.flip_x()

group = doc["grupo_cenario"]
group.content.fit(doc.canvas)
group.content.crop((100, 100, 500, 500))
```
