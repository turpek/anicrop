# Release Notes — anicrop v0.8.0

> **Versão:** `0.8.0`  
> **Data de Lançamento:** 11 de Outubro de 2026  
> **Tag Git:** `v0.8.0`

---

## Destaques da Versão (Highlights)

A versão **v0.8.0** é um marco de evolução arquitetural e de performance no motor `anicrop`, introduzindo aceleração nativa de baixo nível em Cython para o modo de mesclagem **MULTIPLY**, a reestruturação canônica de coleções com **`NamedStack`**, o protocolo declarativo **`Cacheable`** para cache incremental e a blindagem completa do subsistema reativo de histórico.

---

## 1. Novo Modo de Mesclagem Nativo: `BlendMode.MULTIPLY` (Cython / C)

Implementação do modo de mesclagem W3C Porter-Duff Multiply acelerado com paralelismo OpenMP e aritmética inteira de baixo nível:

* **Conformidade Matemática W3C:** Interpolação completa com suporte a alfa cruzado em ambas as camadas (onde o overlay é transparente, a base é preservada; onde a base é transparente, o overlay prevalece).
* **Aritmética Inteira de Ponto Fixo Q8 (Zero `float` / Zero `roundf`):** Eliminação de mais de 6 milhões de conversões de ponto flutuante por frame Full HD. Divisões por 255 com arredondamento exato via `div255_round` compiladas pelo GCC em constantes mágicas recíprocas (`imulq` + `shrq`) de 1 ciclo.
* **Tabela Estática Recíproca Q48 (`RCP_DENOM_Q48`):** Eliminação de 100% das instruções de divisão inteira (`idiv`) por canal de cor através de uma tabela de 2 KB residente na Cache L1 da CPU.
* **Atalho para Fundo Opaco ($\alpha_{\text{base}} = 255$):** Simplificação algébrica direta para duas multiplicações inteiras sem divisão.

### Métricas Oficiais de Benchmark:
| Cenário | anicrop (NumPy) | anicrop (v0.8.0 C) | Baseline Pillow | Baseline OpenCV | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1080p RGBA / RGBA** *(Alfa duplo)* | 372.35 ms | **14.04 ms** (71.2 FPS) | 14.12 ms | 4.36 ms | **26.5x** |
| **1080p RGBA / RGB** *(Fundo opaco)* | 266.42 ms | **7.96 ms** (125.6 FPS) | — | — | **33.5x** |
| **1080p Opacity=0.7** | 361.18 ms | **17.38 ms** (57.6 FPS) | 14.17 ms | 4.09 ms | **20.8x** |
| **Patch 512x512 RGBA** | 45.48 ms | **1.80 ms** (556.0 FPS) | 1.96 ms | 0.52 ms | **25.3x** |
| **1080p RGB puro** *(Sem alfa)* | 260.10 ms | **1.07 ms** (934.0 FPS) | 14.10 ms | 3.62 ms | **3.4x sobre OpenCV** |

---

## 2. Pilhas Canônicas de Camadas (`NamedStack`, `EffectStack`, `EditStack`)

* **Coleção Abstrata `NamedStack[T]`:** Container ordenado e fortemente tipado que suporta acesso por índice numérico, fatiamento (`slice`) e busca por nome da camada/efeito (`stack["Nome"]`).
* **`EffectStack`:** Substitui coleções avulsas de efeitos em `BaseLayer.effects`, oferecendo cálculo agregado de margem para filtros (`get_padding()`).
* **`EditStack`:** Gerencia a fila sequencial de patches em `Layer.edits`, provendo liberação determinística de buffers de memória via `close()`.
* **Remoção de Código Legado:** Eliminação completa da antiga classe `ListView` e centralização nas novas pilhas canônicas.
* **Undo/Redo Reativo de 1 Clique:** Integração completa com `ProxyNamedStack` e `StackCommand` para reversão atômica de inserções, remoções e reordenações.

---

## 3. Sistema de Cache Incremental & Protocolo `Cacheable`

* **Protocolo Declarativo `Cacheable`:** Eliminação definitiva de inspeção de bytecode Python (`dis`) e descontinuação da classe `DynamicEffect`. Efeitos agora declaram estaticamente se são cacheados através do protocolo.
* **Isolamento de Snapshots com `weakref.ref`:** Prevenção de colisões de `id()` e retenção de memória em snapshots incrementais de histórico e cache.
* **Fail-Safe em `LayerCacheScope`:** Garantia de restauração de estado e cleanup robusto mesmo na ocorrência de exceções durante o ciclo de renderização.

---

## 4. Blindagem do Sistema Reativo e Exportações Públicas

* **Proteção de Atributos Privados:** O proxy reativo de domínio agora ignora interceptação em atributos internos (`_parent`, `_cache`, `_parent_inverse`), prevenindo criação espúria de comandos no histórico.
* **Novas Exportações Públicas em `anicrop`:**
  - `Canvas` e `CanvasRender`
  - `ViewportRender`
  - `Point` com suporte a hashing (`Point.__hash__`) para uso seguro em dicionários e conjuntos.
* **I/O e Máscaras:**
  - Implementação de `Mask.__eq__` para comparação semântica.
  - Repasse correto de parâmetros `shrink` e `roi` em chamadas `read_large` de backends de imagem.

---

## Verificação e Qualidade

* **Suíte de Testes:** 1.404 testes passando (`100% verde`).
* **Formatadores / Linter:** 100% de conformidade com `ruff check` e `autopep8`.
