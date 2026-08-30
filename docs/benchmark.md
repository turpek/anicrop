# Benchmarks de Desempenho do anicrop

Este documento consolida as métricas oficiais de desempenho do motor gráfico `anicrop`, cobrindo tanto o processamento geométrico/árvore de camadas quanto o pipeline de renderização e blend de pixels.

---

## 1. Benchmark de Matrizes e Geometria (`freeze_geometry`)

- **Script:** [`scripts/benchmark_matrix_strategies.py`](file:///home/gui/python/anicrop/scripts/benchmark_matrix_strategies.py) e [`scripts/benchmark_freeze_10k.py`](file:///home/gui/python/anicrop/scripts/benchmark_freeze_10k.py)
- **Metodologia:** Simulação de árvore hierárquica balanceada de grupos e camadas com fator de ramificação 3/4. Avalia o tempo total para executar consultas completas de geometria (`matrix`, `region`, `global_region`) por nó.

### Resultados Comparativos:

| Quantidade de Grupos | Total de Nós | Modo Dinâmico (s) | Modo Congelado (`freeze_geometry`) (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **1 grupo** | 2 nós | 0.0001s | 0.0000s | **2.51x** |
| **10 grupos** | 19 nós | 0.0041s | 0.0002s | **18.38x** |
| **50 grupos** | 99 nós | 0.1005s | 0.0014s | **71.66x** |
| **100 grupos** | 199 nós | 0.3989s | 0.0026s | **153.17x** |
| **250 grupos** | 499 nós | 2.5921s | 0.0072s | **359.47x** |
| **500 grupos** | 999 nós | 10.6828s | 0.0156s | **684.62x** |
| **1.000 grupos** | 1.999 nós | 46.1000s | 0.0409s | **1.126,65x** 🏆 |
| **10.000 grupos** | 19.999 nós | *(inviável dinamicamente)* | 0.4321s (100k queries) | **231.433 matrizes/s** |

> [!NOTE]
> O `freeze_geometry` congela temporariamente o cálculo de matrizes afins ($3 \times 3$), regiões locais e bounding boxes globais via *snapshot* preguiçoso sob demanda, reduzindo a complexidade de consulta em árvores profundas de $O(N \cdot D)$ para $O(1)$.

---

## 2. Benchmark de Estresse de Renderização (`CanvasRender`)

- **Script:** [`scripts/benchmark_canvas_render_stress.py`](file:///home/gui/python/anicrop/scripts/benchmark_canvas_render_stress.py)
- **Metodologia:** Cena em resolução Full HD ($1920 \times 1080$) com camadas sobrepostas de $400 \times 400$ pixels reais em formato RGBA, contendo rotações aleatórias, escalas, translações e interpolação Lanczos4.
- **Ambiente:** CPU x86_64, Python 3.12 com OpenCV/NumPy e Extensão C nativa (`_c_blend`).

### Resultados de Renderização (Lanczos4 - 1080p):

| Camadas Ativas | Cena Completa 1080p (s) | FPS (Cena) | Retalho Patch 800x600 (s) | FPS (Patch) |
| :--- | :--- | :--- | :--- | :--- |
| **5 camadas** | **0.0753s** | **13.28 FPS** | **0.0371s** | **26.94 FPS** |
| **10 camadas** | **0.1460s** | **6.85 FPS** | **0.0691s** | **14.46 FPS** |
| **20 camadas** | **0.3097s** | **3.23 FPS** | **0.1252s** | **7.98 FPS** |
| **40 camadas** | **0.5556s** | **1.80 FPS** | **0.3184s** | **3.14 FPS** |
| **80 camadas** | **1.1546s** | **0.87 FPS** | **0.6743s** | **1.48 FPS** |

---

## 3. Benchmark de I/O de Imagens (`Image.open` e `Image.save`)

- **Script:** [`scripts/benchmark_image_io.py`](file:///home/gui/python/anicrop/scripts/benchmark_image_io.py)
- **Metodologia:** Avaliação de throughput (Megapixels/s) e latência média por operação de leitura e gravação em disco para resoluções de $256 \times 256$ (LOD/Ícone) até $3840 \times 2160$ (4K UHD) nos formatos JPEG, PNG e WebP.
- **Configurações de Teste:** `SaveOptions(quality=90, compression_level=6)`.

### 3.1. Resultados Comparativos: `OpenCVBackend` vs `PyvipsBackend`

#### 📁 WebP (`.webp` - RGBA, `quality=90`):
| Resolução | Leitura OpenCV | Leitura Pyvips | Escrita OpenCV | Escrita Pyvips | Speedup na Leitura |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **256x256** (Ícone / LOD) | 3.74 ms | **0.75 ms** | 17.50 ms | 18.51 ms | **5.0x mais rápido** ⚡ |
| **1280x720** (720p HD) | 51.26 ms | **1.21 ms** | 229.45 ms | 243.19 ms | **42.3x mais rápido** 🚀 |
| **1920x1080** (1080p FHD) | 115.27 ms | **1.99 ms** | 559.95 ms | 563.39 ms | **57.9x mais rápido** 🏆 |
| **3840x2160** (4K UHD) | 464.80 ms | **5.92 ms** | 2420.00 ms | 2446.40 ms | **78.5x mais rápido** 🏆 |

#### 📁 PNG (`.png` - RGBA, `compression_level=6`):
| Resolução | Leitura OpenCV | Leitura Pyvips | Escrita OpenCV | Escrita Pyvips | Speedup Geral |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **256x256** (Ícone / LOD) | **0.53 ms** | 1.55 ms | 7.52 ms | **6.41 ms** | **1.17x escrita** |
| **1280x720** (720p HD) | 7.97 ms | **4.52 ms** | 107.52 ms | **78.05 ms** | **1.76x leitura / 1.37x escrita** ⚡ |
| **1920x1080** (1080p FHD) | 19.38 ms | **10.27 ms** | 250.32 ms | **179.05 ms** | **1.88x leitura / 1.40x escrita** ⚡ |
| **3840x2160** (4K UHD) | 74.67 ms | **28.57 ms** | 971.56 ms | **712.75 ms** | **2.61x leitura / 1.36x escrita** ⚡ |

#### 📁 JPEG (`.jpg` - RGB, `quality=90`):
| Resolução | Leitura OpenCV | Leitura Pyvips | Escrita OpenCV | Escrita Pyvips | Destaque |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **256x256** (Ícone / LOD) | **0.64 ms** | 2.68 ms | **0.46 ms** | 1.29 ms | OpenCV 2x mais rápido |
| **1280x720** (720p HD) | **9.59 ms** | 17.07 ms | **5.55 ms** | 11.70 ms | OpenCV 1.8x mais rápido |
| **1920x1080** (1080p FHD) | **20.83 ms** | 35.06 ms | **12.61 ms** | 24.47 ms | OpenCV 1.7x mais rápido |
| **3840x2160** (4K UHD) | **76.64 ms** | 130.17 ms | **55.55 ms** | 94.71 ms | OpenCV 1.7x mais rápido |

---

### 3.2. Conclusões e Guia de Uso:

1. **WebP e PNG:** O `PyvipsBackend` é imbatível, decodificando WebP 1080p em **menos de $2\text{ ms}$** (speedup de até **$78.5\times$** em 4K) e gravando PNGs até **$29\%$ mais rápido**.
2. **JPEG:** O `OpenCVBackend` é mais rápido devido à integração direta da `libjpeg-turbo` com memória contígua BGR/RGB.
3. **Recomendação de Arquitetura:**
   * Utilizar `set_default_backend("vips")` para pipelines com PNG/WebP e suporte a transparência alfa.
   * Utilizar `set_default_backend("opencv")` para pipelines dedicados exclusivamente a sequências de frames JPEG.

---

---

## 4. Benchmark de Formatos de Cor e Mesclagem (`PRGBA` e `RGBX`)

- **Script:** [`scripts/benchmark_color_blend_formats.py`](file:///home/gui/python/anicrop/scripts/benchmark_color_blend_formats.py)
- **Metodologia:** Avaliação comparativa de latência média por operação de mesclagem (`blend_normal` com opacidade 0.9) entre formatos com alfa (*Straight* `RGBA`, *Premultiplied* `PRGBA` e `RGBX` opaco de 32 bits compilados nativamente em C/OpenMP via Cython), e latência de conversões bidirecionais via `anicrop.color`.

### 4.1. Mesclagem Normal (`blend_normal` - Extensão C / Cython):

| Resolução | `RGBA -> RGBA` *(Cython C)* | `PRGBA -> PRGBA` *(Cython C)* | `PRGBA -> RGBX` *(Cython C)* | Speedup vs NumPy Puro |
| :--- | :---: | :---: | :---: | :---: |
| **256x256** (Retalho Pequeno) | **0.06 ms** | **0.06 ms** | 0.14 ms | **121.8x mais rápido** ⚡ |
| **512x512** (Camada Padrão) | **0.22 ms** | **0.23 ms** | 0.36 ms | **147.4x mais rápido** ⚡ |
| **1280x720** (720p HD) | **0.79 ms** | **0.83 ms** | **0.80 ms** | **130.6x mais rápido** 🚀 |
| **1920x1080** (1080p FHD) | 2.31 ms | 3.62 ms | **1.81 ms** | **58.9x mais rápido** 🏆 |
| **3840x2160** (4K UHD) | **6.72 ms** | 9.70 ms | 7.94 ms | **102.6x mais rápido** 🏆 |

> [!NOTE]
> Com os kernels dedicados em Cython (`_cy_blend_normal_prgba` e `_cy_blend_prgba_over_opaque`), o tempo de mesclagem para `PRGBA` em 1080p caiu de **213.39 ms (NumPy)** para **3.62 ms** (ou **1.81 ms** sobre fundo opaco `RGBX`), trazendo um ganho de performance de até **$147\times$** sem comprometer a qualidade visual e eliminando o artefato de *alpha bleeding*.

### 4.2. Velocidade de Conversão de Formatos ([`anicrop.color`](file:///home/gui/python/anicrop/src/anicrop/color.py)):

| Resolução | `RGBA -> PRGBA` | `PRGBA -> RGBA` | `RGB -> RGBX` | `RGBX -> RGB` |
| :--- | :---: | :---: | :---: | :---: |
| **256x256** (Retalho Pequeno) | 1.13 ms | 3.38 ms | 362.1 µs | 339.3 µs |
| **512x512** (Camada Padrão) | 5.79 ms | 16.72 ms | 1.62 ms | 1.44 ms |
| **1280x720** (720p HD) | 22.96 ms | 61.91 ms | 5.77 ms | 5.06 ms |
| **1920x1080** (1080p FHD) | 50.78 ms | 152.15 ms | 13.02 ms | 11.38 ms |
| **3840x2160** (4K UHD) | 241.27 ms | 678.79 ms | 53.26 ms | 46.94 ms |

---

## 5. Como Reproduzir os Benchmarks

Para executar localmente a suíte de benchmarks de estresse, matrizes, I/O e formatos:

```bash
# Benchmark de I/O de imagens (JPEG, PNG, WebP)
uv run python scripts/benchmark_image_io.py

# Benchmark de formatos de cor e mesclagem (RGBA, PRGBA, RGBX)
uv run python scripts/benchmark_color_blend_formats.py

# Benchmark de estresse de renderização (CanvasRender Full HD vs Patch)
uv run python scripts/benchmark_canvas_render_stress.py

# Benchmark de matrizes dinâmicas vs. freeze snapshot
uv run python scripts/benchmark_matrix_strategies.py

# Benchmark de estresse com 10.000 grupos
uv run python scripts/benchmark_freeze_10k.py
```
