# Guia do Sistema de Cache Incremental de Camadas (`anicrop.cache`)

O módulo `anicrop.cache` implementa o motor de cache e aceleração gráfica do `anicrop`, projetado para otimizar renderizações sequenciais contínuas (ex: reprodução de animações, streaming de frames, renderização interativa na Viewport ou nós compostos no motor de animação Anifuse).

O sistema atinge até **$10.5\times$ de aceleração** (elevando taxas de ~12 FPS para 94 a 133 FPS) através de reuso cirúrgico de buffers pré-assados (*baked warps*), invariância afim em translações puras, particionamento de efeitos dinâmicos e controle contextual estrito em patches.

---

## 1. Arquitetura Geral do Sistema

```
                  ┌──────────────────────────────────────────────┐
                  │                 LayerCache                   │
                  │   - states: dict[Layer, LayerFrameState]     │
                  │   - register(layer) / unregister(layer)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                         Cria escopo temporário no render
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               LayerCacheScope                │
                  │   - with cache(container, effective_region): │
                  │   - Filtra camadas 100% contidas na região   │
                  │   - Troca contexto: _activate_layer          │
                  └──────────────────────┬───────────────────────┘
                                         │
                         Durante o pipeline de render
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               LayerFrameState                │
                  │   - baked_warp: Image (warp afim 2x2)        │
                  │   - baked_effects: Image (filtros estáticos) │
                  │   - edits: list[EditLayer] (deltas)          │
                  │   - effects: list[Effect] (deltas)           │
                  └──────────────────────────────────────────────┘
```

1. **`LayerCache`**: Fachada central e repositório de estados registrados. Permite consulta de sujeira via `is_dirty(layer)` e injeção manual de bakes com `set_baked(layer, image)`.
2. **`LayerFrameState`**: Contêiner puro de dados que retém os buffers rasterizados (`baked_warp`, `baked_effects`), matrizes de referência e contadores de visibilidade.
3. **`LayerCacheScope`**: Gerenciador de contexto ativado automaticamente pelo renderizador (`render_scene`, `render_container`, `render_patch`). Aplica *patching* seguro em `layer.background`, `layer._edits` e `layer._effects`, restaurando-os integralmente ao término do frame.

---

## 2. Invalidação Inteligente na Submatriz $2 \times 2$

Em gráficos 2D por matrizes homogêneas $3 \times 3$, a transformação espacial é decomposta em duas partes:

$$\mathbf{M} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

* A **submatriz afim $2 \times 2$** ($\begin{bmatrix} a & b \\ c & d \end{bmatrix}$) define **escala, rotação e cisalhamento (*skew*)**. Qualquer modificação nesses coeficientes exige reamostragem física dos pixels via OpenCV (`warpAffine`).
* O **vetor de translação** ($\begin{bmatrix} t_x \\ t_y \end{bmatrix}$) apenas desloca a origem geométrica da camada no Canvas.

O `LayerCache` monitora estritamente a submatriz $2 \times 2$:
```python
def is_dirty(self, layer: Layer) -> bool:
    if layer not in self._states:
        return True
    status = self._states[layer]
    if status.baked_warp is None or status.matrix is None:
        return True
    return not np.allclose(layer.matrix[:2, :2], status.matrix[:2, :2], atol=1e-5)
```

### Comportamento Resultante:
* **Translação Pura (Pan / Deslocamento de Personagem):** O `baked_warp` permanece intacto. A imagem já renderizada é reutilizada em $O(1)$ e carimbada na nova posição global sem acionar interpolação bilinear ou Lanczos.
* **Rotação ou Escala (Zoom / Giro de Câmera):** A submatriz $2 \times 2$ é alterada, invalidando o `baked_warp` e disparando um novo *bake* afim no frame.

---

## 3. Particionamento de Efeitos Estáticos vs. Dinâmicos (`DynamicEffect`)

Em composições gráficas, certos efeitos são puramente estáticos (ex: um `BlurFilter` constante de profundidade de campo), enquanto outros variam a cada frame (ex: corte dinâmico de costuras no Anifuse, ondas ou brilhos oscilantes).

Para conciliar máxima aceleração com reatividade total, o `anicrop` introduz a classe abstrata `DynamicEffect`:

```python
from anicrop.effect import DynamicEffect, Effect

class MeuEfeitoAnimado(DynamicEffect):
    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        # Executado a cada frame sobre o buffer em cache
        return processar_pixels_dinamicos(image)
```

### Fluxo de Bake de Efeitos:
1. **Filtros Estáticos (antes do primeiro `DynamicEffect`):** São executados exatamente 1 vez e seus pixels congelados em `status.baked_effects`.
2. **Efeitos Dinâmicos (`DynamicEffect`):** São isolados e aplicados sequencialmente a cada novo frame sobre a cópia do `baked_effects`.
3. **Reconhecimento Transparente:** Efeitos envelopados por `BoundEffect` são inspecionados automaticamente pelo cache via desembrulho polimórfico (`_unwrap_effect`).

---

## 4. Invalidação Reativa por Visibilidade

A ativação do cache rastreia alterações na propriedade `.visible` de cada componente individual:
* **Mudança de Visibilidade em Edits (`EditLayer.visible`):** Se qualquer um dos edits pré-assados tiver sua visibilidade alternada, `status.baked_warp` e `status.baked_effects` são invalidados.
* **Mudança de Visibilidade em Efeitos Estáticos (`Effect.visible`):** Se a visibilidade de um efeito estático mudar, o `baked_warp` afim é preservado intacto e apenas o `baked_effects` é recalculado.

---

## 5. Isolamento Contextual em Patches (`render_patch` & `effective_region`)

Ao renderizar recortes parciais através de `render_patch(..., view_region=...)`, o renderizador calcula a `effective_region = surface.region & view_region` e a injeta no `LayerCacheScope`.

### Regra de Preservação de Integridade:
Uma camada só tem seu contexto ativado e seus métodos interceptados se a interseção com a região efetiva **não alterar o seu tamanho geométrico** (`layer.global_region`):

```python
if self._effective_region is not None:
    if not self._effective_region.overlaps(layer.global_region):
        continue
    if (layer.global_region & self._effective_region).size != layer.global_region.size:
        # A camada está parcialmente cortada pela borda do patch.
        # O contexto de cache NÃO é ativado para este layer!
        continue
```

### Garantias do Mecanismo:
1. **Camadas 100% Contidas no Patch:** Têm o cache ativado normalmente e reaproveitam o `baked_warp`.
2. **Camadas Parcialmente Cortadas:** Renderizam sob demanda através do pipeline nativo sem cache (`wrap_background` não é instalado).
3. **Imunidade contra Corrupção:** Uma renderização parcial em patch nunca sobrescreve nem contamina o `baked_warp` de alta resolução gerado para a cena completa.

---

## 6. Injeção Externa de Bakes (`set_baked`)

Para ferramentas de pipeline externo (como o compositor de 2 passos do Anifuse), é possível injetar buffers pré-renderizados diretamente na camada:

```python
cache = LayerCache()
cache.set_baked(layer, imagem_pre_renderizada, matrix=layer.matrix)

# A partir deste ponto, o cache considera a camada pronta e limpa
assert not cache.is_dirty(layer)
```

---

## 7. Exemplo Completo de Uso

```python
from anicrop import Canvas, CanvasRender, Image, Layer
from anicrop.cache import LayerCache
from anicrop.filter import BlurFilter

# 1. Cria a camada e o cache
layer = Layer(Image.open("personagem.png"))
layer.add_effect(BlurFilter(radius=3.0))

cache = LayerCache()
cache.register(layer)

renderer = CanvasRender()
canvas = Canvas(layer.global_region)

# Frame 1: Bake inicial completo (calcula warp e efeitos estáticos)
renderer.render_scene([layer], canvas, cache=cache)
assert not cache.is_dirty(layer)

# Frame 2: Apenas translação (Move 50px para a direita)
layer.transform.translate(50, 0)
assert not cache.is_dirty(layer)  # Submatriz 2x2 inalterada!

# Frame 2 renderiza instantaneamente reusando baked_warp em O(1)
renderer.render_scene([layer], canvas, cache=cache)

# Frame 3: Rotação (Invalida o warp afim)
layer.transform.rotate(15)
assert cache.is_dirty(layer)  # Requer novo warp

renderer.render_scene([layer], canvas, cache=cache)
```
