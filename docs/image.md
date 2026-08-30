# Guia de Dados de Imagem — Classe `Image` e Subsistema de I/O (`anicrop.image` / `anicrop.io`)

O módulo `anicrop.image` fornece o wrapper `Image`, que encapsula matrizes de pixels em formato `numpy.ndarray` ou arranjos em disco `zarr.core.Array` sob uma API padronizada e expressiva para manipulação e edição gráfica.

As operações de leitura e gravação em disco são gerenciadas de forma modular pelo subsistema `anicrop.io`.

---

## 1. `Image` (`anicrop.image.Image`)

A classe `Image` garante a integridade dos dados de imagem (validação de formato, número de canais e suporte a transparência) e permite indexação espacial nativa por meio de objetos `Region`.

---

### Principais Métodos e Propriedades de `Image`

#### `open(file_path: str | Path, image_format: ImageFormat | None = None, backend: AbstractImageIO | str | None = None, shrink: int = 1, roi: Region | None = None) -> Image` *(Class Method)*
- **Descrição**: Abre e decodifica uma imagem a partir do disco utilizando o backend de I/O ativo (padrão `PyvipsBackend` ou `OpenCVBackend`). Suporta auto-detecção de formato de canais, subamostragem direta no decoder (`shrink`) e recorte de ROI sem carregar a imagem inteira. Para imagens gigantes ($\ge 8192 \times 8192\text{px}$), chaveia automaticamente para backend em disco **Zarr**.
- **Parâmetros**:
  - `file_path` (`str | Path`): Caminho do arquivo no disco.
  - `image_format` (`ImageFormat | None`): Formato alvo desejado (`RGBA`, `RGB`, `GRAY`, `GRAY_ALPHA`). Se `None`, auto-detecta o formato nativo da imagem no disco.
  - `backend` (`AbstractImageIO | str | None`): Backend específico para esta leitura (`"vips"`, `"opencv"` ou instância). Se `None`, utiliza o backend padrão ativo.
  - `shrink` (`int`): Fator de subamostragem direta no decoder (ex: `shrink=2` reduz a resolução pela metade durante a leitura, economizando CPU e RAM).
  - `roi` (`Region | None`): Recorte espacial opcional para carregar apenas uma região específica do arquivo.
- **Retorno**: `Image` — Instância contendo os pixels decodificados no formato solicitado.

#### `save(file_path: str | Path, options: SaveOptions | None = None, backend: AbstractImageIO | str | None = None) -> None`
- **Descrição**: Codifica e grava a imagem no disco no caminho especificado utilizando o backend de I/O selecionado.
- **Parâmetros**:
  - `file_path` (`str | Path`): Caminho do arquivo de saída no disco.
  - `options` (`SaveOptions | None`): Objeto de configurações de compressão/qualidade. Se `None`, utiliza as opções padrão (`quality=90`, `compression_level=6`, `bg_color=(255, 255, 255)`).
  - `backend` (`AbstractImageIO | str | None`): Backend específico para a gravação (`"vips"`, `"opencv"` ou instância).
- **Retorno**: `None`.

#### `new(size: tuple[int, int], fmt: ImageFormat, color: int | tuple[int, ...] = 0, threshold_pixels: int = 4096 * 4096) -> Image` *(Class Method)*
- **Descrição**: Cria uma nova imagem em memória preenchida com uma cor constante. Se o total de pixels (`width * height`) ultrapassar `threshold_pixels`, aloca automaticamente um array empacotado em disco via **Zarr**; caso contrário, aloca um `numpy.ndarray`.
- **Parâmetros**:
{{ ... }}
- **Descrição**: Extrai a matriz NumPy da sub-região especificada convertendo automaticamente os canais de cor para a ordem **BGR / BGRA** esperada pelas funções do OpenCV (`cv2.imshow`, `cv2.imwrite`, processamento de visão).
- **Parâmetros**:
  - `region` (`Ellipsis | Region`): A sub-região espacial a ser extraída (padrão `...` para a imagem inteira).
- **Retorno**: `np.ndarray` — Matriz NumPy pronta para o OpenCV.

#### Propriedades de Dimensão e Metadados:
- `@property size -> tuple[int, int]`: Retorna `(width, height)` da imagem em pixels.
- `@property width -> int` / `@property height -> int`: Retornam a largura e a altura da imagem.
- `@property shape -> tuple[int, ...]`: Retorna a tupla de dimensões da matriz interna `(height, width, channels)`.
- `@property channels -> int`: Retorna o número de canais da imagem (ex: `4` para RGBA, `3` para RGB, `1` para GRAY).
- `@property format -> ImageFormat`: Retorna o enum `ImageFormat` associado.
- `@property has_alpha -> bool`: Retorna `True` se o formato da imagem incluir canal de transparência (Alpha).
- `@property is_zarr -> bool`: Retorna `True` se os dados estiverem armazenados em um array Zarr no disco.

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

### 2.3. Gerenciamento Global de Backends
```python
from anicrop.io import set_default_backend, get_default_backend

# Define o OpenCV como backend padrão
set_default_backend("opencv")

# Define o Pyvips como backend padrão
set_default_backend("vips")

# Consulta o backend ativo
backend_atual = get_default_backend()
```

---

## 3. Funções Utilitárias do Módulo

#### `calculate_content_rect(image: Image) -> Region`
- **Descrição**: Analisa o canal alpha da imagem e calcula o menor retângulo delimitador (*bounding box*) que engloba todos os pixels visíveis/não-transparentes (`alpha > 0`).
- **Parâmetros**:
  - `image` (`Image`): A imagem a ser analisada.
- **Lança**: `ValueError` se a imagem possuir canal alpha mas estiver totalmente transparente.
- **Retorno**: `Region` — Região delimitadora do conteúdo visível.
