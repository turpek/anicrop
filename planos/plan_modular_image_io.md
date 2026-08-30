# Plano de Arquitetura: Módulo Modular de I/O de Imagens (`anicrop.io`)

> **Objetivo:** Substituir o acoplamento direto com o OpenCV por uma arquitetura extensível de codificação/decodificação de imagens baseada em interfaces abstratas, suportando múltiplos backends de alta performance (**Pyvips / libvips** e **OpenCV**) com suporte completo a auto-detecção de `ImageFormat`, streaming e controle avançado de parâmetros de compressão/leitura.

---

## 1. Visão Geral e Motivação

### Problemas Atuais do Pipeline OpenCV:
1. **Decodificação/Codificação Sequencial:** O `cv2.imread` e `cv2.imwrite` do OpenCV são síncronos e lentos no Python para processamento em lote.
2. **Dupla Leitura de Arquivo:** O `Image.open` atual abre o arquivo primeiramente com o Pillow para extrair dimensões (`pil_img.size`) e depois reabre do disco com `cv2.imread`.
3. **Conversões Redundantes:** O OpenCV carrega em BGR/BGRA, forçando sucessivas chamadas de `cv2.cvtColor` para converter para RGB/RGBA.
4. **Falta de Parâmetros de Compressão:** Não há controle direto de qualidade (JPEG/WebP quality), compressão PNG (zlib levels), dpi ou tratamento de transparência ao exportar para formatos sem canal alfa.

### Benefícios do Novo Módulo com `pyvips`:
* **Multi-threading Nativo e Streaming:** `libvips` utiliza pipelines sob demanda e SIMD, sendo de **3x a 8x mais rápido** que o OpenCV para I/O.
* **Subamostragem no Decoder (`shrink`):** Decodificar imagens gigantes (4K/8K) diretamente em resolução reduzida (LOD/Miniaturas) reduz o tempo de I/O em até 80% e a memória em até 75%.
* **Tratamento Seguro de Canais:** Conversão automática de espaços de cor (`sRGB`, `flatten` para JPEG, `bandjoin` para alfa) sem cópias intermediárias em Python.

---

## 2. Estrutura de Diretórios e Arquivos Proposta

```text
src/anicrop/
├── interfaces/
│   └── io.py                  # Contrato base: AbstractImageIO, SaveOptions
├── io/
│   ├── __init__.py            # Facade de I/O, registro de backends e default backend
│   ├── base.py                # Implementações compartilhadas / helpers
│   ├── opencv.py              # Backend OpenCV (cv2.imread / cv2.imwrite)
│   ├── vips.py                # Backend Pyvips (pyvips.Image)
│   └── zarr_io.py             # Backend Zarr para imagens gigantes (>= 8192px)
```

---

## 3. Especificação das Interfaces e Tipos

### 3.1. Opções de Salvamento (`SaveOptions`)
```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SaveOptions:
    """Opções de codificação para salvar imagens no disco."""
    quality: int = 90                   # 1-100 para JPEG e WebP Lossy
    lossless: bool = False              # True para WebP / AVIF sem perdas
    compression_level: int = 6          # 0-9 para PNG (zlib)
    bg_color: tuple[int, ...] = (255, 255, 255)  # Cor de fundo se imagem RGBA for salva em JPG
    strip_metadata: bool = False        # Remove metadados EXIF/ICC para reduzir tamanho
    dpi: tuple[int, int] | None = None  # Resolução para impressão
```

### 3.2. Contrato Abstrato de I/O (`AbstractImageIO`)
```python
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import zarr
from anicrop.enums import ImageFormat
from anicrop.spatial import Region

class AbstractImageIO(ABC):
    """Contrato formal para decodificadores e codificadores de imagens."""

    @abstractmethod
    def read(
        self,
        file_path: str | Path,
        format: ImageFormat | None = None,
        shrink: int = 1,
        roi: Region | None = None,
    ) -> tuple[np.ndarray | zarr.Array, ImageFormat, tuple[int, int]]:
        """
        Lê e decodifica uma imagem a partir do disco.
        
        Args:
            file_path: Caminho do arquivo.
            format: Formato desejado. Se None, auto-detecta o formato nativo do arquivo.
            shrink: Fator de redução direta no decoder (ex: 2 para metade da largura/altura).
            roi: Recorte espacial opcional para ler apenas uma área sem decodificar o todo.
            
        Returns:
            Tupla contendo (array de dados, formato resolvido, tamanho original (W, H)).
        """
        ...

    @abstractmethod
    def write(
        self,
        file_path: str | Path,
        data: np.ndarray | zarr.Array,
        format: ImageFormat,
        options: SaveOptions | None = None,
    ) -> None:
        """
        Codifica e grava a imagem no disco.
        
        Args:
            file_path: Caminho de destino.
            data: Matriz de pixels.
            format: Formato dos canais em memória (RGB, RGBA, GRAY, etc.).
            options: Configurações de qualidade e compressão.
        """
        ...

    @abstractmethod
    def get_size(self, file_path: str | Path) -> tuple[int, int]:
        """
        Extrai as dimensões (largura, altura) da imagem lendo apenas o cabeçalho,
        sem decodificar a matriz de pixels.
        """
        ...
```

---

## 4. Regras para Tratamento do `ImageFormat` no I/O

### 4.1. Na Leitura (`read` / `open`):
1. **Auto-Detecção (`format=None`):**
   * **1 canal:** `ImageFormat.GRAY`
   * **2 canais:** `ImageFormat.GRAY_ALPHA`
   * **3 canais:** `ImageFormat.RGB`
   * **4 canais:** `ImageFormat.RGBA`
2. **Formato Explícito (`format=ImageFormat.X`):**
   * O backend realiza a conversão na própria decodificação antes de retornar a matriz final.
   * `pyvips`: aplica `vips_img.colourspace()`, `flatten()` ou `bandjoin()` em C sem custo de cópia Python.
   * `opencv`: converte via matrizes de conversão direta `cv2.cvtColor`.

### 4.2. Na Escrita (`write` / `save`):
1. **Compatibilidade Automática de Canais:**
   * Se o formato de destino for `.jpg` / `.jpeg` e a matriz em memória possuir canal alfa (`RGBA` ou `GRAY_ALPHA`), o canal alfa é composto contra a cor `options.bg_color` (padrão branco) antes da compressão JPEG.
2. **Compressão e Qualidade:**
   * Arquivos `.png` usam `compression_level` (0 = mais rápido para previews, 9 = máxima compressão).
   * Arquivos `.jpg` e `.webp` usam `quality` (padrão 90).

---

## 5. Integração com a Classe `Image` e Fachada

### Assinaturas na Classe `Image`:
```python
class Image:
    def __init__(
        self,
        image: np.ndarray | zarr.Array,
        image_format: ImageFormat,
        backend: AbstractImageIO | None = None,
    ):
        ...
        self.backend = backend or get_default_io_backend()

    @classmethod
    def open(
        cls,
        file_path: str | Path,
        format: ImageFormat | None = None,
        backend: AbstractImageIO | str | None = None,
        shrink: int = 1,
        roi: Region | None = None,
    ) -> Image:
        ...

    def save(
        self,
        file_path: str | Path,
        backend: AbstractImageIO | str | None = None,
        quality: int = 90,
        lossless: bool = False,
        compression_level: int = 6,
        bg_color: tuple[int, ...] = (255, 255, 255),
        strip_metadata: bool = False,
    ) -> None:
        ...
```

### Gerenciador de Backends Padrão:
```python
from anicrop.io import set_default_backend, get_default_backend, PyvipsBackend, OpenCVBackend

# Define o backend padrão para todas as operações do sistema
set_default_backend(PyvipsBackend()) # ou OpenCVBackend()
```

---

## 6. Roteiro de Implementação Passo a Passo

### Fase 1: Interfaces e Tipos Base
- [ ] Criar `src/anicrop/interfaces/io.py` com `SaveOptions` e `AbstractImageIO`.
- [ ] Criar pacote `src/anicrop/io/` com registro de backends (`registry.py` ou `manager.py`).

### Fase 2: Backend OpenCV Refatorado
- [ ] Implementar `OpenCVBackend` em `src/anicrop/io/opencv.py` com suporte a `SaveOptions`, `get_size` rápido via cabeçalho e `read` sem dupla abertura.
- [ ] Adicionar testes unitários dedicados em `tests/test_io_opencv.py`.

### Fase 3: Backend Pyvips de Alta Performance
- [ ] Implementar `PyvipsBackend` em `src/anicrop/io/vips.py`.
- [ ] Adicionar suporte a streaming, conversão sem cópia e `shrink` direto no decoder.
- [ ] Adicionar fallback gracioso (se `pyvips` não estiver instalado, o sistema usa `OpenCVBackend` automaticamente).
- [ ] Adicionar testes unitários em `tests/test_io_pyvips.py`.

### Fase 4: Integração com `Image` e `Document`
- [ ] Atualizar `Image.open`, `Image.save`, `Document.open` e `Document.export` para utilizar o novo subsistema de I/O.
- [ ] Atualizar documentação e benchmarks comparativos de I/O em `docs/image.md` e `docs/benchmark.md`.
