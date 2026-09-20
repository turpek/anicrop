# Sistema de Cache de Camadas com Decorators e Renderização Incremental (`LayerCache`)

> **Módulo / Domínio:** `anicrop.cache` / `anicrop.render` / `anicrop.layer`  
> **Status:** Mapeado para Implementação  
> **Referência Principal:** Tarefa 28 em `planos/plano.md`

---

## 1. Visão Geral e Motivação

O objetivo deste sistema é acelerar dramaticamente a composição e renderização de camadas que possuem múltiplos recortes (`EditLayer`) e efeitos (`Effect`), mantendo a pureza matemática do `BaseRenderer` e a simplicidade de uso da engine.

### Problemas Resolvidos:
1. **Recálculo Quadrático de Edits ($O(N)$):** Em camadas com muitas edições cumulativas (como stitching, pintura ou múltiplos recortes), cada novo edit adicionado forçava o renderizador a recompor todos os edits anteriores a partir do zero.
2. **Duplo Warp em Pipelines de Alinhamento (`anifuse`):** No `anifuse`, a imagem é rotacionada e escalada na primeira passada para estimar a translação. Sem um mecanismo de injeção de cache, o renderizador era forçado a executar o `warpAffine` de novo na cena final, ou o usuário precisava recorrer a *workarounds* complexos (como injetar `EditLayer` com matriz inversa).
3. **Zero Poluição de Estado no Renderizador:** O `BaseRenderer` não possui `self._cache`. O cache é um parâmetro opcional injetado pelo cliente (`render_scene(..., cache=cache)`), preservando o renderizador como um serviço puro e sem estado interno.
4. **Contrato de API Intacto para o Consumidor:** Fora do escopo de renderização, o `Layer` comporta-se de forma 100% padrão: `layer.edits` e `layer.effects` retornam suas listas completas, e `layer.background(...)` retorna um buffer limpo (`Image.new`). Apenas durante a janela de renderização ativada pelo context manager `with cache(container):` os métodos e properties retornam deltas e buffers pré-compostos.

---

## 2. Arquitetura do Sistema

A arquitetura baseia-se em **4 pilares fundamentais**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. API Pública e Registro: LayerCache                                  │
│    - cache = LayerCache()                                              │
│    - cache.register(layer) -> Registra a camada para rastreamento      │
│    - cache.set_baked(layer, image) -> Injeção externa (anifuse)        │
│    - with cache(container): -> Context manager com escopo cirúrgico    │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Context Manager de Renderização: LayerCacheScope                    │
│    - Itera sobre o container e identifica camadas registradas.         │
│    - Ativa temporariamente o decorator durante o traverse:            │
│        * layer.background() -> Retorna buffer cacheado                 │
│        * layer.edits        -> Retorna apenas deltas (unbaked_edits)   │
│        * layer.effects      -> Retorna apenas deltas de pós-processo   │
│    - No __exit__: Restaura os métodos/properties originais da camada   │
│      e consolida os novos resultados renderizados no cache.            │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Pós-Processamento Otimizado: CacheEffect                            │
│    - Efeito especializado injetado na fila de efeitos da camada.       │
│    - Reutiliza a imagem pós-efeitos sem recalcular filtros estáticos.  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Ponto de Injeção no Renderizador (BaseRenderer)                     │
│    - cache passado como argumento: render_scene(..., cache=None)       │
│    - with (cache(container) if cache else nullcontext()):              │
│    - layer_image = layer.background(...) no _flatten_edits.            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Especificação Detalhada dos Componentes

### 3.1. Método `layer.background(...)` em `Layer`

Criação do método oficial na classe `Layer` para fornecer a superfície inicial de desenho:

```python
class Layer(BaseLayer, AbstractLayer):
    ...
    def background(
        self,
        size: tuple[int, int],
        format: ImageFormat,
        dtype: np.dtype = np.uint8,
    ) -> Image:
        """Retorna o buffer inicial para a composição da camada.

        No modo normal (fora do context manager de cache), retorna um novo
        buffer transparente Image.new.
        Quando dentro do context manager with cache(container):, este método
        é interceptado para entregar a imagem consolidada dos edits anteriores.
        """
        return Image.new(size, format, dtype=dtype)
```

---

### 3.2. Gerenciador `LayerCache` e Context Manager com Container

O `LayerCache` gerencia as camadas registradas. Ao ser chamado como context manager com o contêiner alvo (`cache(container)`), ele ativa o escopo cirúrgico apenas nas camadas pertencentes à cena:

```python
class LayerCache:
    def __init__(self) -> None:
        self._decorators: dict[Id, CachedLayerDecorator] = {}

    def register(self, layer: Layer) -> None:
        """Registra a camada para gerenciamento de cache incremental."""
        if layer._id not in self._decorators:
            self._decorators[layer._id] = CachedLayerDecorator(layer, self)

    def set_baked(self, layer: Layer, image: Image) -> None:
        """Permite injeção externa de imagens pré-assadas (ex: anifuse)."""
        if layer._id in self._decorators:
            self._decorators[layer._id].set_baked_image(image)

    def __call__(self, container: Sequence[BaseLayer] | Container) -> LayerCacheScope:
        """Cria o context manager otimizado para o container fornecido."""
        return LayerCacheScope(self, container)


class LayerCacheScope:
    """Escopo de contexto temporário que ativa os decorators apenas durante o render."""

    def __init__(
        self,
        cache: LayerCache,
        container: Sequence[BaseLayer] | Container,
    ) -> None:
        self._cache = cache
        self._container = container
        self._active_decorators: list[CachedLayerDecorator] = []

    def __enter__(self) -> LayerCacheScope:
        # Itera pelo contêiner e ativa cirurgicamente apenas as camadas registradas
        for item in self._container:
            if isinstance(item, Layer) and item._id in self._cache._decorators:
                decorator = self._cache._decorators[item._id]
                decorator.activate()
                self._active_decorators.append(decorator)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restaura imediatamente as camadas para seu comportamento normal e comita o cache
        for decorator in self._active_decorators:
            decorator.deactivate(commit=(exc_type is None))
        self._active_decorators.clear()
```

---

### 3.3. Decorator da Camada (`CachedLayerDecorator`)

O `CachedLayerDecorator` gerencia o estado da camada:

1. **Rastreamento de Deltas:**
   - Mantém referências a:
     - `_baked_image: Image | None`: Imagem resultante das passadas anteriores.
     - `_unbaked_edits: list[EditLayer]`: Edits adicionados após o último bake.
     - `_is_dirty: bool`: Indica se houve invalidação destrutiva (remoção/reordenação de edits).
2. **Ativação (`activate()`):**
   - Sobrescreve temporariamente na instância da camada:
     - `layer.background = self.cached_background`
     - `layer.__class__.edits` (ou via wrapper): retorna `self.delta_edits`
     - `layer.__class__.effects` (ou via wrapper): retorna `self.delta_effects`
3. **Desativação (`deactivate(commit=True)`):**
   - Restaura as referências originais na camada (`del layer.background`, etc.).
   - Se `commit=True`:
     - O buffer gerado no render torna-se o novo `_baked_image`.
     - `_unbaked_edits.clear()`.
     - `_is_dirty = False`.
4. **Comportamento Fora do Contexto:**
   - Para qualquer código cliente:
     - `layer.edits`: retorna todos os `EditLayer`s da camada.
     - `layer.effects`: retorna todos os efeitos da camada.
     - `layer.background(...)`: retorna `Image.new(...)`.
   - Nenhuma funcionalidade de inspeção ou manipulação externa é quebrada.

---

### 3.4. Efeito de Cache para Pós-Processamento (`CacheEffect`)

- Efeito especializado injetado na lista `effects` da camada durante o escopo ativo.
- Ao ser executado por `apply_post_processing`:
  - Se nenhum efeito novo foi adicionado e os parâmetros não mudaram, retorna a imagem final cacheada com efeitos.
  - Se novos efeitos forem adicionados após ele, atua como o ponto de partida acelerado para aplicar apenas os deltas de efeitos.

---

### 3.5. Modificações Mínimas no `BaseRenderer` (`render.py`)

O renderizador permanece **100% puro e sem estado interno** (`self._cache` NÃO existe). As únicas alterações necessárias são:

1. **Parâmetro `cache` nas Funções de Renderização:**
   ```python
   def render_scene(
       self,
       container: Sequence[BaseLayer] | Container,
       surface: SurfaceProtocol,
       format: ImageFormat = ImageFormat.RGBA,
       interp: InterpMode = InterpMode.LANCZOS,
       cache: LayerCacheProtocol | None = None,
   ) -> Image:
       with freeze_geometry(container):
           with (cache(container) if cache is not None else nullcontext()):
               traverser = SceneTraverser(...)
               images = traverser.traverse(container)
               ...
   ```

2. **No `render_patch` e `CanvasRender.render_container`:**
   - Propagam o argumento opcional `cache: LayerCacheProtocol | None = None`.

3. **No `_flatten_edits`:**
   ```python
   def _flatten_edits(
       self,
       visible_edits: list[EditLayer],
       layer: Layer,
       plan: BaseFrame,
       interp: InterpMode,
   ) -> Image:
       target_dtype = visible_edits[0].image.dtype if visible_edits else np.uint8
       
       # layer.background entrega Image.new (normal) ou o buffer cacheado (no escopo with cache):
       layer_image = layer.background(
           plan.dst_region.size,
           layer.format,
           dtype=target_dtype,
       )
       for edit_layer in visible_edits:
           scratch = self._scratch_buffer.configure(
               layer_image.size, edit_layer.image.format, dtype=edit_layer.image.dtype
           )
           result = render_edit(edit_layer, plan, interp=interp, dst=scratch)
           if result is None:
               continue
           edit_image, dst_region = result
           edit_layer.blend_into(layer_image, edit_image, dst_region)

       return layer_image
   ```

4. **No `render_area` (Roteamento com Cache):**
   - Se `visible_edits` estiver vazio (nenhum delta pendente) e houver cache ativo: `_flatten_edits` retorna diretamente `layer.background(...)` em tempo $O(1)$.
   - Se `len(visible_edits) == 1` e houver cache ativo: compõe o delta sobre `layer.background(...)`.
   - Se não houver cache ativo e `len(visible_edits) == 1`: mantém o fast-path `_render_single_edit` com zero-copy.

---

## 4. Integração com o `anifuse` (Eliminação do Workaround de Matriz Inversa)

No `anifuse`, o pipeline passa a ser limpo e sem truques matemáticos:

```python
# 1. Passo 1: Transforma a imagem e acha ângulo/escala
warped_image = transform_image(img, angle=angle, scale=scale)

# 2. Injeta direto no cache do anicrop:
cache.set_baked(layer, warped_image)

# 3. Passo 2: Define a transformação real na camada (sem matriz inversa!):
layer.transform.rotate(angle).scale(scale).translate(dx, dy)

# 4. Renderização:
# O anicrop reaproveita a imagem do cache e apenas translada para o Canvas.
# Zero warps redundantes calculados!
```

---

## 5. Plano de Ação e Fases de Implementação

1. **Fase 1: Método `Layer.background` e Suporte no `BaseRenderer`**
   - Implementar `Layer.background(size, format, dtype)` em `src/anicrop/layer.py`.
   - Atualizar `_flatten_edits` e chamadas de render em `src/anicrop/render.py` para invocar `layer.background(...)`.
   - Adicionar o parâmetro opcional `cache: LayerCacheProtocol | None = None` em `render_scene`, `render_patch` e `render_container`.

2. **Fase 2: Infraestrutura `LayerCache`, `LayerCacheScope` e `CachedLayerDecorator`**
   - Criar módulo `src/anicrop/cache.py` com `LayerCache`, `LayerCacheScope` e `CachedLayerDecorator`.
   - Implementar rastreamento de deltas de edits (`unbaked_edits`) e substituição contextual de `layer.background`, `edits` e `effects`.

3. **Fase 3: Suporte a `CacheEffect` para Efeitos / Pós-Processamento**
   - Implementar `CacheEffect` e integração com `base.effects` dentro do escopo de renderização.

4. **Fase 4: Injeção Externa de Imagens Assadas (`set_baked`)**
   - Implementar `cache.set_baked(layer, image)` para atender pipelines como o `anifuse`.

5. **Fase 5: Testes Unitários e Validação de Equivalência**
   - Testar renderização de edits única vez vs incremental com múltiplos `add_edit`.
   - Testar restauração de comportamento normal fora do context manager (`layer.edits`, `layer.effects`, `layer.background`).
   - Testar invalidação por remoção e por alteração de parâmetros.
   - Validar equivalência visual exata de pixels (com cache vs sem cache).
