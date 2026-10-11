# Release Notes — anicrop v0.8.1

> **Versão:** `0.8.1`  
> **Data de Lançamento:** 11 de Outubro de 2026  
> **Tag Git:** `v0.8.1`

---

## Destaques da Versão (Highlights)

A versão **v0.8.1** é um *patch release* focado em segurança de memória de baixo nível, integridade de cache de renderização e suporte robusto a pipelines de imagens em **16 bits e ponto flutuante** nos backends de I/O (`PyvipsBackend` e `OpenCVBackend`).

---

## 1. Segurança de Memória e Correções no Cython (`min_pool_alpha`)

* **Prevenção de Stack Overflow:** As tabelas pré-computadas de limites (`x_start_lut` e `y_start_lut`) eram alocadas na pilha C com tamanho estático de 1024 posições sem validação de limites, o que gerava risco de *stack buffer overflow* e *segfault* caso dimensões maiores fossem solicitadas. A rotina agora utiliza alocação dinâmica (`malloc`/`free` protegidos por bloco `try/finally`) sob demanda para resoluções $> 1024$, preservando o fast-path em pilha para miniaturas habituais.
* **Suporte a Striding Horizontal:** Substituição do cálculo de passo fixo `stride_x = ch` por `stride_x = src.strides[1]`, permitindo que matrizes com passos não contíguos no eixo horizontal sejam processadas com total segurança.

---

## 2. Suporte Multi-Dtype e Correções no Pyvips (`PyvipsBackend`)

* **Paridade de Tipos (uint16 e float32):** Implementação de tabelas de conversão bidirecional de formatos entre a libvips e o NumPy (`_VIPS_FORMAT_TO_DTYPE` e `_DTYPE_TO_VIPS_FORMAT`).
* **Preservação de Espaço de Cor 16-bit (`rgb16`):** Correção da conversão de espaço de cor que convertia inadvertidamente imagens de 16 bits para 8 bits (`uchar srgb`). Imagens `ushort` agora mantêm a interpretação `rgb16`.
* **Cálculo Analítico de Alfa Opaco:** Implementação de `_vips_opaque_alpha` fornecendo opacidade analítica exata conforme a profundidade de bits (`65535` em 16-bit, `1.0` em float e `255` em 8-bit).
* **Exportação PNG em 16 bits:** Configuração automática de `bitdepth=16` na chamada de salvamento (`pngsave`) quando a matriz de entrada for de 16 bits.
* **Fatiamento em `VipsStreamingBuffer`:** Correção do método `__getitem__` para utilizar `self.dtype` em vez de `np.uint8` fixo tanto no operador ellipsis (`stream[...]`) quanto nos recortes de fatias retangulares (`stream[y, x]`).

---

## 3. Despacho Estático e Tratamento de Erros no OpenCV (`OpenCVBackend`)

* **Caminhos Unicode no Windows sem Custo no Linux:** Implementação de despacho estático no momento da importação (`_cv2_imread` / `_cv2_imwrite`). Em sistemas POSIX (Linux/macOS), a chamada é direta e pura para o C runtime do OpenCV; no Windows, ativa-se o fallback via `np.fromfile` e `cv2.imencode` para blindagem contra caracteres não-ASCII.
* **Propagação Adequada de `OSError`:** Eliminação de blocos `except Exception: return None` silenciosos que mascaravam falhas de acesso a dados ou permissão como `FileNotFoundError`. Agora o backend levanta explicitamente `OSError(...) from exc`.

---

## 4. Preservação de Cache e Dtype no Renderizador (`render.py`)

* **Correção no Retorno de Edits Fora do Canvas:** Ao renderizar uma camada onde um novo edit não intersecta a área visível do canvas, o fallback anterior gerava uma `Image.new` crua sem vínculo com o cache e com `dtype` global fixo em 8 bits, descartando o buffer pré-assado (`status.baked_warp`).
* **Resolução com `layer.background`:** A chamada agora utiliza `layer.background(dst_region.size, layer.format, dtype=visible_edits[0].image.dtype)`, preservando o cache da camada e a fidelidade de tipo.

---

## 5. Qualidade e Cobertura de Testes

* **11 Novos Testes de Regressão:**
  * Teste do cenário de preservação de cache com edits fora do canvas (`tests/test_cache.py`).
  * Testes de alocação dinâmica ($1050\times 1050$) e *stride* não-contíguo em `min_pool_alpha` (`tests/test_render.py`).
  * Testes de leitura, escrita e fatiamento de streaming de 16 bits no Pyvips (`tests/test_io_pyvips.py`).
  * Testes de caminhos Unicode e levantamento de `OSError` no OpenCV (`tests/test_io_opencv.py`).
* **Suíte Completa:** **1.462 testes passando** em 13.70s.
* **Tipagem:** Mypy com 0 erros em 63 arquivos.
