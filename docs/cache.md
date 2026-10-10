# Guia do Sistema de Cache Incremental de Camadas (`anicrop.cache`)

O módulo `anicrop.cache` implementa o motor de cache e aceleração gráfica do `anicrop`, projetado para otimizar renderizações sequenciais contínuas (ex: reprodução de animações, streaming de frames, renderização interativa na Viewport ou nós compostos no motor de animação Anifuse).

O sistema atinge até **$10.5\times$ de aceleração** (elevando taxas de ~12 FPS para 94 a 133 FPS) através de:
- Reuso cirúrgico de buffers pré-assados (*baked warps*) em transformações afins;
- Invariância em translações puras ($O(1)$) com invalidação por comparação direta de bytes (`memcmp`);
- Particionamento de efeitos estáticos vs. dinâmicos (`DynamicEffect`);
- Inspeção por bytecode (`dis`) do método `apply` para detecção cirúrgica de mutações em filtros;
- Snapshots leves de primitivos em edições locais (`snapshot_edit`), imunes a vazamento de memória;
- Gerenciamento fraco de ciclo de vida (`weakref.WeakKeyDictionary`), liberando buffers automaticamente no Garbage Collector;
- Acesso de zero-alocação a coleções com a view somente-leitura `ListView`.

---

## 1. Arquitetura Geral do Sistema

```
                  ┌────────────────────────────────────────────────────────┐
                  │                       LayerCache                       │
                  │   - _states: WeakKeyDictionary[Layer, State]           │
                  │   - register(layer) / unregister(layer)                │
                  │   - is_dirty(layer) / set_baked(layer, image)          │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                              Cria escopo temporário no render
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                    LayerCacheScope                     │
                  │   - with cache(container, effective_region):           │
                  │   - Filtra camadas 100% contidas na região efetiva     │
                  │   - Troca contexto: _activate_layer                    │
                  │   - Expõe apenas edits pendentes sobre o baked_warp    │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                              Durante o pipeline de render
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                    LayerFrameState                     │
                  │   - baked_warp: Image | None (warp afim 2x2)           │
                  │   - baked_effects: Image | None (filtros estáticos)    │
                  │   - matrix_2x2_bytes: bytes | None (memcmp O(1))       │
                  │   - baked_effects_snapshot: tuple (bytecode apply)     │
                  │   - baked_edits_snapshot: tuple[tuple[int, bool, BM]]  │
                  └────────────────────────────────────────────────────────┘
```

1. **`LayerCache`**: Fachada central e repositório de estados registrados. Utiliza `WeakKeyDictionary` para indexar camadas, garantindo que o descarte de uma camada pelo usuário libere imediatamente todos os buffers pesados de imagem sem necessidade de desregistro manual.
2. **`LayerFrameState`**: Contêiner de estado de alta densidade que armazena referências a buffers rasterizados e snapshots de primitivos (tuplas de inteiros e bytes), sem reter instâncias completas de objetos pesados.
3. **`LayerCacheScope`**: Gerenciador de contexto ativado automaticamente pelo renderizador (`CanvasRender`, `ViewportRender`). Modula o fatiamento de edits pendentes e restaura a integridade da camada ao término do frame sem criar referências circulares fortes.

---

## 2. Invalidação de Baixa Latência na Submatriz $2 \times 2$ (Zero-Alloc & Memcmp)

Em gráficos 2D por matrizes homogêneas $3 \times 3$, a transformação espacial é decomposta em duas partes:

$$\mathbf{M} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

* A **submatriz afim $2 \times 2$** ($\begin{bmatrix} a & b \\ c & d \end{bmatrix}$) define **escala, rotação e cisalhamento (*skew*)**. Qualquer modificação nesses coeficientes altera a geometria do raster e exige reamostragem física dos pixels via OpenCV (`warpAffine`).
* O **vetor de translação** ($\begin{bmatrix} t_x \\ t_y \end{bmatrix}$) apenas desloca a origem geométrica da camada no Canvas.

### Otimização por Comparação Direta de Bytes:

Em loops interativos a 60–120 FPS, utilizar verificações tradicionais do NumPy (`np.allclose` ou `np.array_equal`) introduzia sobrecarga de ~6.500 ns por camada, gerando alocações temporárias no heap.

O `LayerCache` serializa a submatriz $2 \times 2$ para bytes contíguos (`matrix[:2, :2].tobytes()`) e compara diretamente em nível de memória C (`memcmp`):

```python
def is_dirty(self, layer: Layer) -> bool:
    if layer not in self._states:
        return True
    status = self._states[layer]
    if status.baked_warp is None or status.matrix_2x2_bytes is None:
        return True
    curr_2x2_bytes = layer.matrix[:2, :2].tobytes()
    return curr_2x2_bytes != status.matrix_2x2_bytes
```

### Resultados de Desempenho:
* **Latência Reduzida:** De ~6.500 ns para **~15 a 105 ns** ($60\times$ a $400\times$ mais rápido).
* **Translação Pura (Pan / Câmera):** A submatriz $2 \times 2$ é idêntica; o buffer `baked_warp` é reaproveitado em $O(1)$ sem re-amostragem de pixels.
* **Rotação ou Escala (Zoom / Giro):** Os bytes divergem, disparando novo warp afim no frame.

---

## 3. Particionamento de Efeitos e Inspeção por Bytecode (`DynamicEffect` & `dis`)

O motor particiona efeitos entre componentes estáticos (que podem ser pré-assados) e componentes dinâmicos a cada frame através da classe abstrata `DynamicEffect`:

```python
from anicrop.effect import DynamicEffect, Effect

class CorteDeCostura(DynamicEffect):
    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        # Executado a cada frame sobre o buffer em cache
        return processar_pixels_dinamicos(image)
```

### Invalidação Cirúrgica via Inspeção de Bytecode de `cls.apply`:

Filtros frequentemente possuem parâmetros mutáveis (ex: `blur.radius = 10.0` ou atributos privados `_radius`). Para detectar qualquer alteração sem exigir que os filtros sejam envolvidos por proxies pesados ou que o usuário notifique o cache manualmente:

1. **Inspeção Estrita do Escopo de `apply`:**
   A função `inspect_apply_attributes(cls)` inspeciona as instruções compiladas do método `cls.apply` usando `dis.get_instructions`.
   - Rastreia pares de opcodes `LOAD_FAST 'self'` seguidos de `LOAD_ATTR <nome>`.
   - **Filtros rigorosos:** Ignora dunders (`__...__`) e métodos chamáveis (`callable`).
   - **Preservação de atributos privados:** Atributos como `_radius` são monitorados, assegurando que métodos modificadores internos sejam capturados.
   - Inclui explicitamente o atributo `"visible"`.
2. **Cache de Metadados por Tipo (`WeakKeyDictionary`):**
   Os atributos relevantes são compilados exatamente 1 vez por classe de efeito e reutilizados em todas as instâncias daquele tipo.
3. **Snapshot de Estado (`snapshot_effect`):**
   A cada ativação, extrai os valores dos atributos monitorados em uma tupla imutável. Para arrays NumPy, converte para `.tobytes()`. Para instâncias de `BoundEffect`, inspeciona recursivamente o efeito encapsulado.

Se qualquer parâmetro de filtro variar ou se sua visibilidade mudar:
* O `baked_warp` afim **permanece intacto** (evitando novo warp pesado de geometria).
* Apenas o `baked_effects` é recalculado e reaplicado sobre o warp já em memória.

---

## 4. Snapshots Leves de Edições Locais (`snapshot_edit`)

Na arquitetura do `anicrop`, um `EditLayer` representa um patch de edição com imagem, região e matriz fixas. Os **únicos** atributos mutáveis de um `EditLayer` após sua criação são:
1. `visible: bool`
2. `blend_mode: BlendMode`

### Eliminação de Acumuladores Paralelos e Vazamentos de Buffer:

Em versões anteriores, o cache acumulava instâncias de `EditLayer` em listas paralelas via monkey-patching em `add_edit`. Isso falhava ao realizar Undo/Redo no histórico e retinha referências fortes para imagens pesadas.

A arquitetura atual utiliza `snapshot_edit`:

```python
def snapshot_edit(edit: EditLayer) -> tuple[int, bool, BlendMode]:
    return (id(edit), edit.visible, edit.blend_mode)
```

### Mecânica de Operação:
1. **Zero Retenção de Imagens:** O snapshot armazena apenas uma tupla de três primitivos `(id, visible, blend_mode)`. Se a camada de edição for descartada, sua memória RAM ou `MMapBuffer` é liberada imediatamente pelo GC.
2. **Renderização Incremental de Edits:**
   Quando `baked_warp` está assado com $N$ edits, os edits excedentes ($M > N$) são expostos como fatia limpa (`layer._edits[N:]`) durante o frame do cache e compostos sobre o `baked_warp`.
3. **Detecção Instantânea de Undo/Redo:**
   Se a contagem de edits diminuir (`len(layer._edits) < N`), ou se qualquer um dos edits pré-assados tiver seu modo de mesclagem ou visibilidade alterados, o cache detecta a divergência via `zip(status.baked_edits_snapshot, layer._edits)`, invalida o `baked_warp` e recompõe a base do zero.

---

## 5. Gerenciamento Seguro de Ciclo de Vida (`weakref.WeakKeyDictionary`)

Para garantir que o motor possa ser utilizado em processos de longa duração sem vazamentos graduais de memória:

- **Tabela de Estados com Referência Fraca:**
  `LayerCache._states` é implementado como `weakref.WeakKeyDictionary[Layer, LayerFrameState]`.
  Quando uma camada é removida de um contêiner e de todas as referências do aplicativo, sua entrada na tabela e todos os seus buffers assados (`baked_warp`, `baked_effects`) são expurgados automaticamente pelo Garbage Collector.
- **Eliminação de Referências Circulares (`orig_*`):**
  Todos os atributos legados de *bound methods* que retinham referências cíclicas fortes para `Layer` foram eliminados. A restauração de métodos ocorre diretamente sobre o `__dict__` da instância, permitindo que a hierarquia retome o despacho original sem criar âncoras na memória.

---

## 6. Acesso de Baixa Latência com `ListView[T]`

Para eliminar a alocação contínua de tuplas descartáveis a cada frame (`return tuple(self._edits)`):

- A coleção interna de edições (`Layer._edits`) foi migrada para `list` pura em Python, otimizando fatiamentos rápidos e indexação direta.
- A propriedade pública `layer.edits` retorna uma instância de `ListView[T]` (`anicrop.type.ListView`).
- Trata-se de uma view imutável baseada em `__slots__ = ("_data",)` que implementa `Sequence[T]` (suportando `len()`, iteração, indexação por inteiro e por slice) com tempo de criação de **~30 ns** e **zero cópia de memória**.

---

## 7. Isolamento Contextual em Patches (`render_patch` & `effective_region`)

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
2. **Camadas Parcialmente Cortadas:** Renderizam sob demanda através do pipeline nativo sem cache.
3. **Imunidade contra Corrupção:** Uma renderização parcial em patch nunca sobrescreve nem contamina o `baked_warp` de alta resolução gerado para a cena completa.

---

## 8. Injeção Externa de Bakes (`set_baked`)

Para pipelines especialistas (como pré-renderizadores de fundo ou geradores de miniaturas no Anifuse), é possível injetar buffers pré-renderizados diretamente na camada:

```python
cache = LayerCache()
cache.set_baked(layer, imagem_pre_renderizada, matrix=layer.matrix)

# A partir deste ponto, o cache considera a camada pronta e limpa
assert not cache.is_dirty(layer)
```

---

## 9. Pontos de Integração Normalizados

O parâmetro `cache: AbstractLayerCache | None = None` é aceito de maneira uniforme em todos os pontos de renderização e composição do motor:

- **Renderizadores (`BaseRenderer`, `CanvasRender`, `ViewportRender`):**
  - `render_scene(container, surface, ..., cache=cache)`
  - `render_patch(container, surface, view_region, ..., cache=cache)`
  - `render_container(container, ..., cache=cache)`
  - `render_layer(layer, ..., cache=cache)`
- **Fachada `Document`:**
  - `doc.render(..., cache=cache)`
  - `doc.preview(viewport, ..., cache=cache)`
  - `doc.export(path, ..., cache=cache)`
- **Composição (`anicrop.composition`):**
  - `flatten(layers, ..., cache=cache)`
  - `LayerComposition.flatten(layers, ..., cache=cache)`
  - `doc.combine.flatten(target, name, ..., cache=cache)`
  - `doc.combine.bake(group, ..., cache=cache)`
  - `doc.combine.bake_stack(..., cache=cache)`

---

## 10. Exemplo Completo de Uso

```python
from anicrop import Canvas, CanvasRender, Image, Layer
from anicrop.cache import LayerCache
from anicrop.filter import BlurFilter

# 1. Cria a camada e o cache
layer = Layer(Image.open("personagem.png"))
blur = BlurFilter(radius=3.0)
layer.add_effect(blur)

cache = LayerCache()
cache.register(layer)

renderer = CanvasRender()
canvas = Canvas(layer.global_region)

# Frame 1: Bake inicial completo (calcula warp afim e efeitos estáticos)
renderer.render_scene([layer], canvas, cache=cache)
assert not cache.is_dirty(layer)

# Frame 2: Apenas translação (Move 50px para a direita)
layer.transform.translate(50, 0)
assert not cache.is_dirty(layer)  # Submatriz 2x2 idêntica (memcmp O(1))

# Frame 2 renderiza instantaneamente reutilizando baked_warp sem reamostragem
renderer.render_scene([layer], canvas, cache=cache)

# Frame 3: Mutação de parâmetro de efeito (detectada via bytecode de apply)
blur.radius = 8.0
# baked_warp é preservado; apenas baked_effects é re-assado no render
renderer.render_scene([layer], canvas, cache=cache)

# Frame 4: Rotação geométrica (Invalida a submatriz 2x2)
layer.transform.rotate(15)
assert cache.is_dirty(layer)  # Bytes divergentes: novo warp afim disparado

renderer.render_scene([layer], canvas, cache=cache)
```

---

## 8. Compatibilidade com Proxies e Sistema de Histórico

Quando a cena é orquestrada através do `Document(history=True)` ou instâncias de proxies reativos (`ProxyLayer`, `LayerStackProxy`):
- O `LayerCache` desempacota automaticamente as camadas e contêineres recebidos via `getattr(item, "_target", item)`.
- As injeções efêmeras de desempenho (como o monkey-patch de `layer.background` no `LayerCacheScope`) operam estritamente sobre as instâncias de domínio puras, sem acionar interceptores de `__setattr__`.
- Isso previne a injeção de comandos fantasmas no `GlobalHistory` e assegura que chamadas de `undo()` e `redo()` preservem exatidão de pixels e velocidade máxima sob cache.
