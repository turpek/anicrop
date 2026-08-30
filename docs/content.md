# Guia de Manipulação de Conteúdo e Pixels — Módulo `Content` (`content.py`)

O módulo `anicrop.content` fornece operações de alto nível para transformação e manipulação direta de conteúdo e pixels de camadas (`Layer`), operando sobre matrizes afins e patches não-destrutivos (`BlendMode.CLIP`).

---

## 1. Princípios Arquiteturais do `Content`

- **Escopo sobre o Conteúdo**: Diferente do módulo `Layout` (que gerencia molduras e enquadramentos de nós), o `Content` manipula a escala visual, espelhamentos, recortes e dimensões efetivas da imagem.
- **Corte Não-Destrutivo (`BlendMode.CLIP`)**: A operação `crop` ajusta a moldura e injeta um patch de máscara `EditLayer` com modo de mesclagem `BlendMode.CLIP`, preservando $100\%$ dos pixels originais da imagem para renderização e histórico.
- **Transformações Afins de Alta Precisão**: Operações como `resize`, `flip_x`, `flip_y` e `fit` alteram diretamente o compositor afim (`target.transform`), evitando re-amostragens destrutivas acumuladas.
- **Transações Atômicas**: Quando executadas sob um proxy reativo no `Document`, operações compostas são encapsuladas em transações atômicas (`history.atomic`), garantindo um único `Undo` para desfazer a alteração completa.

---

## 2. Métodos da Classe `Content` (`anicrop.content.Content`)

### `crop(target: Layer, ref: Region | tuple[int, int, int, int] | Canvas | BaseLayer) -> bool`
- **Descrição**: Recorta o conteúdo visual da camada para os limites especificados em `ref`. Aplica um patch de máscara não-destrutivo com `BlendMode.CLIP`.
- **Parâmetros**:
  - `target` (`Layer`): A camada a ser recortada.
  - `ref`: Limites do recorte (tupla `(x, y, w, h)`, `Region`, `Canvas` ou outra camada).
- **Retorno**: `bool` — `True` se o recorte foi aplicado com sucesso.

```python
# Recorta a camada para uma caixa 400x300 na posição (50, 50)
doc.content.crop(doc["foto"], (50, 50, 400, 300))
```

---

### `resize(target: Layer, width: int, height: int) -> bool`
- **Descrição**: Redimensiona a escala visual do conteúdo da camada para atingir a largura e altura especificadas no Espaço Global.
- **Parâmetros**:
  - `target` (`Layer`): A camada alvo.
  - `width` (`int`): Nova largura em pixels (deve ser > 0).
  - `height` (`int`): Nova altura em pixels (deve ser > 0).
- **Retorno**: `bool` — `True` se as dimensões foram modificadas.

```python
doc.content.resize(doc["foto"], 1280, 720)
```

---

### `fit(target: Layer, ref: Region | tuple | Canvas | BaseLayer) -> bool`
- **Descrição**: Ajusta a escala e translada o conteúdo da camada para preencher exatamente a região de referência `ref`.
- **Sobrecargas (`@ovld`)**:
  - `fit(target, ref)`: Ajuste direto para uma região ou elemento.
  - `fit(payload)`: Recebe uma tupla gerada por um contexto de ajuste proporcional (como `FitContext`).

```python
# Ajusta o conteúdo da foto para cobrir exatamente a região do Canvas
doc.content.fit(doc["foto"], doc.canvas)
```

---

### `flip_x(target: Layer) -> bool` / `flip_y(target: Layer) -> bool`
- **Descrição**: Espelha o conteúdo da camada horizontalmente (`flip_x`) ou verticalmente (`flip_y`) invertendo o sinal da escala afim (`scale(-1, 1)` / `scale(1, -1)`).
- **Parâmetros**:
  - `target` (`Layer`): A camada alvo.
- **Retorno**: `bool` — `True`.

```python
# Espelha a camada horizontalmente
doc.content.flip_x(doc["avatar"])
```

---

## 3. Ajustes Proporcionais Avançados (`FitContext`)

A classe `FitContext` permite calcular enquadramentos proporcionais complexos antes de aplicá-los:

- **`fit_contain`**: Redimensiona proporcionalmente para caber totalmente dentro de `ref` sem cortar nada.
- **`fit_cover`**: Redimensiona proporcionalmente para cobrir totalmente `ref` (com corte das sobras).
- **`scale_width`**: Ajusta proporcionalmente travando a largura em `ref.width`.
- **`scale_height`**: Ajusta proporcionalmente travando a altura em `ref.height`.

### Exemplo de Uso com `FitContext`:
```python
from anicrop.content import FitContext

# Calcula o enquadramento proporcional "contain" centralizado:
ctx = FitContext(doc["foto"], doc.canvas, x_factor=0.5, y_factor=0.5)

# Aplica o resultado calculado no motor de Content:
doc.content.fit(ctx.fit_contain)
```

---

## 4. Acesso Direto via `Document` e `Layer`

Você pode acessar as operações de conteúdo tanto através da fachada do documento quanto diretamente na camada:

```python
# Via Fachada Document:
doc.content.resize(doc["foto"], 800, 600)
doc.content.flip_x(doc["foto"])

# Via Propriedade da Camada:
layer = doc["foto"]
layer.content.resize(800, 600)
layer.content.flip_x()
```
