# Guia de Dados de Imagem — Classe `Image` e Subsistema de I/O (`anicrop.image` / `anicrop.io`)

O módulo `anicrop.image` fornece o wrapper `Image`, que encapsula matrizes de pixels em formato `numpy.ndarray` ou buffers mapeados em memória virtual no disco `np.memmap` (`MMapBuffer`) sob uma API padronizada e expressiva para manipulação e edição gráfica.

As operações de leitura e gravação em disco são gerenciadas de forma modular pelo subsistema `anicrop.io`.

---

## 1. `Image` (`anicrop.image.Image`)

A classe `Image` garante a integridade dos dados de imagem (validação de formato, número de canais e suporte a transparência) e permite indexação espacial nativa por meio de objetos `Region`.

---

### Principais Métodos e Propriedades de `Image`

#### `open(file_path: str | Path, image_format: ImageFormat | None = None, backend: AbstractImageIO | str | None = None, shrink: int = 1, roi: Region | None = None, dtype: Any = ...) -> Image` *(Class Method)*
- **Descrição**: Abre e decodifica uma imagem a partir do disco utilizando o backend de I/O ativo (padrão `PyvipsBackend` ou `OpenCVBackend`). Suporta auto-detecção de formato de canais, subamostragem direta no decoder (`shrink`) e recorte de ROI sem carregar a imagem inteira. Harmoniza automaticamente a profundidade de bits para `dtype` (padrão herdado de `config.dtype`), ou preserva o tipo nativo se `dtype=None`. Para imagens gigantes ($\ge 8192 \times 8192\text{px}$), chaveia automaticamente para backend em disco **`MMapBuffer`** (`np.memmap`) ou streaming via `Pyvips`.
- **Parâmetros**:
  - `file_path` (`str | Path`): Caminho do arquivo no disco.
  - `image_format` (`ImageFormat | None`): Formato alvo desejado (`RGBA`, `RGB`, `GRAY`, `GRAY_ALPHA`). Se `None`, auto-detecta o formato nativo da imagem no disco.
  - `backend` (`AbstractImageIO | str | None`): Backend específico para esta leitura (`"vips"`, `"opencv"` ou instância). Se `None`, utiliza o backend padrão ativo.
  - `shrink` (`int`): Fator de subamostragem direta no decoder (ex: `shrink=2` reduz a resolução pela metade durante a leitura, economizando CPU e RAM).
  - `roi` (`Region | None`): Recorte espacial opcional para carregar apenas uma região específica do arquivo.
  - `dtype` (`Any`): Tipo de dado dos pixels (`np.uint8`, `np.uint16`, `np.float32`). Se omitido (`...`), utiliza `config.dtype`. Se `None`, preserva o tipo original da leitura em disco.
- **Retorno**: `Image` — Instância contendo os pixels decodificados no formato e dtype solicitados.

#### `save(file_path: str | Path, options: SaveOptions | None = None, backend: AbstractImageIO | str | None = None) -> None`
- **Descrição**: Codifica e grava a imagem no disco no caminho especificado utilizando o backend de I/O selecionado.
- **Parâmetros**:
  - `file_path` (`str | Path`): Caminho do arquivo de saída no disco.
  - `options` (`SaveOptions | None`): Objeto de configurações de compressão/qualidade. Se `None`, utiliza as opções padrão (`quality=90`, `compression_level=6`, `bg_color=(255, 255, 255)`).
  - `backend` (`AbstractImageIO | str | None`): Backend específico para a gravação (`"vips"`, `"opencv"` ou instância).
- **Retorno**: `None`.

#### `new(size: tuple[int, int], fmt: ImageFormat, color: int | tuple[int, ...] = 0, threshold_pixels: int | None = ..., dtype: Any = ...) -> Image` *(Class Method)*
- **Descrição**: Cria uma nova imagem em memória preenchida com uma cor constante. Se o total de pixels (`width * height`) ultrapassar o limite configurado (padrão de **64 Megapixels** / $8192 \times 8192\text{px}$), aloca automaticamente um buffer mapeado em memória virtual no disco via **`np.memmap`** (`MMapBuffer`); caso contrário, aloca um `numpy.ndarray` em memória RAM de alta velocidade. O tipo de dado é configurado via `dtype` (padrão herdado de `config.dtype`).
- **Parâmetros**:
  - `size` (`tuple[int, int]`): Dimensões `(width, height)` da imagem.
  - `fmt` (`ImageFormat`): Formato de cor da imagem (`RGBA`, `RGB`, `GRAY`, etc.).
  - `color` (`int | tuple[int, ...]`): Valor ou tupla de cor para preenchimento inicial (padrão `0` transparente/preto).
  - `threshold_pixels` (`int | None`): Limite de pixels antes de paginar em disco. Se omitido, herda o valor global de `get_memory_threshold()`.
  - `dtype` (`Any`): Tipo de dado dos pixels (`np.uint8`, `np.uint16`, `np.float32`). Se omitido (`...`), utiliza `config.dtype`.
- **Retorno**: `Image` — Nova instância alocada.

#### `set_memory_threshold(threshold_pixels: int | None) -> None` / `get_memory_threshold() -> int | None`
- **Descrição**: Configura ou consulta globalmente o limite de pixels para alocação de buffers em RAM antes de usar paginação em disco.
- **Exemplo de Uso**:
  ```python
  from anicrop import get_memory_threshold, set_memory_threshold

  # Desativa a paginação em disco (tudo alocado 100% em RAM pura):
  set_memory_threshold(None)

  # Define um limite customizado (ex: 100 Megapixels):
  set_memory_threshold(100_000_000)

  # Consulta o limite ativo em pixels (padrão inicial: 67.108.864 pixels = 8K x 8K):
  threshold = get_memory_threshold()
  ```

#### `bgr(region: Ellipsis | Region = ...) -> np.ndarray`
- **Descrição**: Extrai a matriz NumPy da sub-região especificada convertendo automaticamente os canais de cor para a ordem **BGR / BGRA** esperada pelas funções do OpenCV (`cv2.imshow`, `cv2.imwrite`, processamento de visão).
- **Parâmetros**:
  - `region` (`Ellipsis | Region`): A sub-região espacial a ser extraída (padrão `...` para a imagem inteira).
- **Retorno**: `np.ndarray` — Matriz NumPy pronta para o OpenCV.


#### `to_format(target_format: ImageFormat) -> Image`
- **Descrição**: Converte a imagem para o formato de canais e espaço de cores especificado (`RGBA`, `PRGBA`, `RGBX`, `RGB`, `GRAY`, `GRAY_ALPHA`) utilizando a tabela de despacho de estratégias de conversão do módulo `anicrop.color`.
- **Parâmetros**:
  - `target_format` (`ImageFormat`): O formato de destino desejado.
- **Retorno**: `Image` — Nova instância contendo os pixels convertidos.

#### `to_dtype(target_dtype: Any) -> Image`
- **Descrição**: Converte a imagem para outro tipo de dado (`np.uint8`, `np.uint16`, `np.float32`) de forma segura, não-destrutiva e com escalonamento de bits preciso (ex: uint8 `255` -> uint16 `65535`, uint16 `65535` -> uint8 `255` via bitshift, float32 em `[0.0, 1.0]`). Se a imagem já estiver no `target_dtype`, retorna `self` sem alocações desnecessárias.
- **Parâmetros**:
  - `target_dtype` (`Any`): Tipo NumPy desejado (ex: `np.uint8`, `np.uint16`, `np.float32`, ou strings como `"uint8"`).
- **Retorno**: `Image` — Nova instância contendo os pixels convertidos.

#### `to_uint8() -> Image`
- **Descrição**: Atalho conveniente e expressivo para `self.to_dtype(np.uint8)`.
- **Retorno**: `Image` — Instância convertida para uint8.

#### Propriedades de Dimensão e Metadados:
- `@property size -> tuple[int, int]`: Retorna `(width, height)` da imagem em pixels.
- `@property width -> int` / `@property height -> int`: Retornam a largura e a altura da imagem.
- `@property shape -> tuple[int, ...]`: Retorna a tupla de dimensões da matriz interna `(height, width, channels)`.
- `@property channels -> int`: Retorna o número de canais da imagem (ex: `4` para RGBA/PRGBA/RGBX, `3` para RGB, `1` para GRAY).
- `@property format -> ImageFormat`: Retorna o enum `ImageFormat` associado (`RGBA`, `PRGBA`, `RGBX`, `RGB`, `GRAY`, `GRAY_ALPHA`, `CMYK`, `CMYK_ALPHA`).
- `@property has_alpha -> bool`: Retorna `True` se o formato da imagem incluir canal de transparência (Alpha ativo em `RGBA`, `PRGBA`, `GRAY_ALPHA`, `CMYK_ALPHA`).
- `@property dtype -> np.dtype`: Retorna o tipo de dados NumPy subjacente da imagem (`np.uint8`, `np.uint16`, `np.float32`).

---

## 2. Subsistema de I/O Modular (`anicrop.io`)

O pacote `anicrop.io` fornece a camada extensível e modular para codificação e decodificação de arquivos no disco.

### 2.1. Backends Disponíveis:
- **`PyvipsBackend` (Padrão de Alta Performance):**
  - Utiliza `libvips` multithread em C com suporte a SIMD e streaming.
  - Até **$58\times$ mais rápido na leitura de WebP** e **$2.6\times$ mais rápido em PNG**.
- **`OpenCVBackend`:**
  - Utiliza OpenCV (`cv2.imread` / `cv2.imwrite`) com fallback transparente caso a `libvips` não esteja instalada no sistema.
  - Oferece velocidade ligeiramente superior para arquivos JPEG contíguos em memória.

### 2.2. Configuração de Opções de Salvamento (`SaveOptions`)
```python
from anicrop.interfaces.io import SaveOptions

options = SaveOptions(
    quality=95,  # Qualidade para JPEG e WebP (1-100)
    lossless=False,  # Modo sem perdas para WebP
    compression_level=6,  # Nível de compressão zlib para PNG (0-9)
    bg_color=(255, 255, 255),  # Cor de fundo sólida ao exportar RGBA para JPEG
    strip_metadata=True,  # Remove metadados EXIF/ICC para reduzir tamanho
)

img.save("export.jpg", options=options)
```

### 2.3. Gerenciamento Global de Configurações (`anicrop.config`)

O `anicrop.config` é o objeto centralizado para gerenciar opções globais do motor (backend de decodificação/gravação, limites de alocação de memória RAM e profundidade de bits/dtype padrão):

```python
import anicrop
import numpy as np

# Consulta as configurações atuais:
print(anicrop.config.backend)  # Padrão: "opencv" (ou "vips")
print(anicrop.config.memory_threshold)  # Padrão: 67108864 pixels (64 MP)
print(anicrop.config.dtype)  # Padrão: np.uint8

# 1. Configuração permanente no script:
anicrop.config.backend = "vips"  # Chaveia para o Pyvips
anicrop.config.memory_threshold = None  # Desativa paginação em disco (100% RAM pura)
anicrop.config.dtype = np.uint16  # Opera pipelines com precisão de 16-bit por padrão

# 2. Configuração temporária e segura via Context Manager (com restauração automática ao sair):
with anicrop.config(backend="vips", memory_threshold=None, dtype=np.uint16):
    img = anicrop.Image.open("frame_16bit.png")
    # ... processamento de alta precisão ...

# Fora do bloco, backend, threshold e dtype voltam automaticamente aos valores anteriores!
```


---

## 3. Funções Utilitárias do Módulo

#### `calculate_content_rect(image: Image) -> Region`
- **Descrição**: Analisa o canal alpha da imagem e calcula o menor retângulo delimitador (*bounding box*) que engloba todos os pixels visíveis/não-transparentes (`alpha > 0`).
- **Parâmetros**:
  - `image` (`Image`): A imagem a ser analisada.
- **Lança**: `ValueError` se a imagem possuir canal alpha mas estiver totalmente transparente.
- **Retorno**: `Region` — Região delimitadora do conteúdo visível.

#### `transform_image(image: Image, angle: float = 0.0, scale: float | tuple[float, float] = 1.0, pivot_angle: tuple[float, float] | Point = (0.5, 0.5), pivot_scale: tuple[float, float] | Point = (0.5, 0.5), interp: InterpMode = InterpMode.LINEAR, dst: AbstractScratchBuffer | None = None, auto_pad: bool = True) -> Image`
- **Descrição**: Aplica transformações afins (rotação e escala) diretamente sobre uma instância de `Image` com composição analítica de matrizes, cálculo automático e exato do *bounding box* resultante (sem cortes de cantos) e proteção automática contra contaminação de cor/franja escura (*dark halo*) em formatos com Straight Alpha (`RGBA`, `GRAY_ALPHA`, `CMYK_ALPHA`).
- **Parâmetros**:
  - `image` (`Image`): Imagem de entrada a ser transformada.
  - `angle` (`float`): Ângulo de rotação em graus (sentido horário).
  - `scale` (`float | tuple[float, float]`): Fator de escala uniforme (`float`) ou anisotrópico `(sx, sy)`.
  - `pivot_angle` (`tuple[float, float] | Point`): Ponto pivô para rotação (coordenadas normalizadas `0.0` a `1.0`, padrão `(0.5, 0.5)` no centro).
  - `pivot_scale` (`tuple[float, float] | Point`): Ponto pivô para escala (padrão `(0.5, 0.5)` no centro).
  - `interp` (`InterpMode`): Algoritmo de interpolação (`LINEAR`, `LANCZOS`, `CUBIC`, `NEAREST`, etc.). Padrão `InterpMode.LINEAR`.
  - `dst` (`AbstractScratchBuffer | None`): Instância de buffer reutilizável (`ScratchBuffer`). O tamanho e formato de destino são reconfigurados dinamicamente para o novo *bounding box*, garantindo zero alocações contínuas em loops de streaming/processamento em lote. Se `None`, aloca uma nova `Image`.
  - `auto_pad` (`bool`): Quando `True` (padrão), estende cirurgicamente as cores da borda da imagem para o canal alpha vazio, eliminando o escurecimento periférico causado por interpolação bilinear/lanczos com preto transparente.
- **Retorno**: `Image` — Nova instância contendo os pixels transformados.
- **Exemplo de Uso**:
  ```python
  import anicrop
  from anicrop import InterpMode, ScratchBuffer

  img = anicrop.Image.open("asset.png")

  # 1. Transformação direta simples (rotação no centro e escala 1.5x)
  rotated = anicrop.transform_image(img, angle=45.0, scale=1.5, interp=InterpMode.LANCZOS)

  # 2. Processamento em lote de alto desempenho com reutilização de memória via ScratchBuffer
  scratch = ScratchBuffer()
  for frame_angle in range(0, 360, 15):
      # O ScratchBuffer adapta-se automaticamente ao bounding box variável de cada ângulo
      transformed_frame = anicrop.transform_image(img, angle=frame_angle, dst=scratch)
      # Consome o frame transformado sem gerar pressão sobre o Garbage Collector...
  ```

---

## 4. Gerenciamento de Memória Temporária: `ScratchBuffer` (`anicrop.ScratchBuffer`)

O `ScratchBuffer` (que implementa `AbstractScratchBuffer`) é um buffer volátil de alto desempenho projetado para eliminar alocações repetitivas de arrays NumPy em operações intensivas (como renderização de patches no `CanvasRender`, loops de rotação/transformação ou pipelines de stitching/vídeo).

### 4.1. Princípios de Design
1. **Alocação Sob Demanda (*Lazy Allocation*):**
   Configurar o buffer via `configure(size, fmt, dtype)` apenas registra as dimensões pretendidas. O array NumPy só é efetivamente alocado quando o método `__getitem__` for chamado com uma `Region`.
2. **Reutilização Zero-Copy:**
   Se uma operação subsequente solicitar dimensões menores ou iguais às já alocadas (e mantiver o mesmo `ImageFormat` e `dtype`), o buffer reaproveita exatamente o mesmo bloco de memória contíguo na RAM, retornando um *slice* sem custo de realocação.
3. **Crescimento Amortizado ($1.5\times$):**
   Quando dimensões maiores são necessárias, a capacidade interna é expandida multiplicando a dimensão anterior por um fator de $1.5\times$, minimizando a ocorrência de realocações sucessivas.
4. **Ciclo de Vida da Flag `was_used`:**
   A propriedade `buf.was_used` é inicializada como `False`, vira `True` ao acessar fatias de memória e reseta automaticamente a cada nova chamada de `buf.configure()`.

### 4.2. API do `ScratchBuffer`
- `configure(size: tuple[float, float], fmt: ImageFormat = ImageFormat.RGBA, dtype: Any = np.uint8) -> ScratchBuffer`: Define os requisitos de geometria e formato para o próximo acesso. Retorna o próprio buffer.
- `__getitem__(region: Region) -> np.ndarray`: Garante a alocação e retorna o slice contíguo como `numpy.ndarray`.
- `@property was_used -> bool`: Indica se houve acesso à memória desde o último `configure`.

### 4.3. Exemplo de Uso
```python
from anicrop import ImageFormat, Region, ScratchBuffer
import numpy as np

# Cria uma instância compartilhada para o pipeline de processamento
scratch = ScratchBuffer()

# Configura para o primeiro frame (1920x1080 RGBA uint8)
scratch.configure(size=(1920, 1080), fmt=ImageFormat.RGBA, dtype=np.uint8)
array_view = scratch[Region.from_size(1920, 1080)]
# array_view é um ndarray de shape (1080, 1920, 4) pronto para OpenCV / NumPy

# Próxima iteração com dimensões menores: ZERO realocações de memória!
scratch.configure(size=(1280, 720), fmt=ImageFormat.RGBA)
sub_view = scratch[Region.from_size(1280, 720)]
```
