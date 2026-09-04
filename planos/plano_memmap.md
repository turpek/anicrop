# Arquitetura de Paginação de Memória: Substituição de Zarr por np.memmap e Pyvips

> **Objetivo:** Estabelecer uma estratégia de paginação (*offloading*) de buffers de imagens gigantes no construtor de `Image` e em operações mutáveis (máscaras/edits), substituindo a dependência pesada do **Zarr** pelo uso conjunto de **`np.memmap`** (para buffers mutáveis em disco) e **`Pyvips`** (para streaming estático de alta performance).

---

## 1. Contexto e Motivação

Atualmente, o `anicrop` adota o `Zarr` quando um array em memória (`ndarray`) ou uma nova imagem (`Image.new`) ultrapassa o limite de memória configurado (`threshold_pixels`, por padrão >= 4096 x 4096 ou >= 8192 x 8192).

### 1.1. Gargalos do Zarr na Paginação de `ndarray`:
1. **Compressão/Descompressão Desnecessária em CPU:** O Zarr fatia o array em chunks (ex: 512 x 512) e comprime cada um (via `blosc` ou `zstd`). Em pipelines gráficos interativos, o custo de CPU para comprimir e descomprimir buffers locais temporários cria latência e engasgos.
2. **I/O Fragmentado no Disco:** Uma imagem de 10.000 x 10.000 particionada em chunks de 512 x 512 gera centenas de pequenos arquivos individuais dentro da pasta `.zarr`.
3. **Dependências Pesadas:** O ecossistema `zarr` (especialmente na versão 3.x) introduziu dependências complexas (`numcodecs`, bibliotecas assíncronas, etc.), dificultando a portabilidade e manutenção do core da biblioteca.

---

## 2. Análise Comparativa: Zarr vs Pyvips vs np.memmap

| Critério | Zarr (`zarr.Array`) | Pyvips (`pyvips.Image`) | NumPy Memmap (`np.memmap`) |
| :--- | :--- | :--- | :--- |
| **Mutabilidade (`__setitem__`)** | Sim (grava em blocos) | **Não** (imutável por design) | **Sim** (acesso e escrita direta) |
| **Velocidade de Escrita Inicial** | Lenta (compressão de chunks) | Ultra-rápida (binário C contínuo `.v`) | **Máxima** (gravação direta pelo SO) |
| **Consumo de Memória (RAM)** | Médio (cache de blocos Python) | Mínimo (streaming sob demanda C) | Mínimo (paginado pelo kernel do SO) |
| **Dependências Externas** | `zarr`, `numcodecs` | `pyvips` + biblioteca de sistema `libvips` | **Zero** (nativo do próprio `numpy`) |
| **Compatibilidade de Plataformas** | Multiplataforma | Multiplataforma | Multiplataforma (Linux, Win, Mac) |

---

## 3. Os Dois Regimes de Imagens Gigantes

Para obter o máximo desempenho de I/O e renderização sem gargalos de memória, a arquitetura divide imagens fora da RAM em dois papéis claros:

### 3.1. Regime Estático / Somente Leitura (Pipeline Pyvips)
* **Casos de Uso:** Camadas de imagem carregadas de arquivos em disco (`Image.open`), texturas de fundo e imagens de alta resolução que são apenas transformadas, fatiadas e compostas.
* **Backend:** `VipsStreamingBuffer` (`src/anicrop/io/vips.py`).
* **Vantagem:** Leitura instantânea (0 ms), sem cópia, com *mmap* e *tiling* em C gerenciado pela `libvips`.

### 3.2. Regime Mutável / Leitura e Escrita (Pipeline np.memmap)
* **Casos de Uso:**
  1. Construtor de `Image` ou `Image.new`: quando um `ndarray` em RAM ultrapassa `threshold_pixels` e precisa ser descarregado para o disco liberando a memória física.
  2. **Máscaras (`Mask`) e Camadas de Edição (`EditLayer`):** Áreas onde ferramentas de pincel, recorte ou comandos de usuário aplicam mutações diretas in-place (`mask[region] = dados`).
* **Backend Proposto:** `MemmapBuffer` (subclasse de `AbstractImageBuffer`).
* **Vantagem:** O array comporta-se exatamente como um `np.ndarray` padrão, porém os dados residem em um arquivo binário no disco gerenciado pelo cache de páginas do sistema operacional.

---

## 4. Multiplataforma e Compatibilidade com Windows

O `np.memmap` é universal e suportado em todas as plataformas onde o NumPy roda:
* **Linux e macOS:** Utiliza a chamada de sistema POSIX padrão `mmap()`.
* **Windows:** Utiliza as APIs Win32 nativas `CreateFileMapping` e `MapViewOfFile`.

### 4.1. Particularidade do Windows: Trava de Arquivo (*File Locking*)
* **Comportamento:** Diferente do Linux (onde um arquivo aberto pode ser excluído do sistema de arquivos e o kernel libera o espaço após o fechamento do processo), o Windows proíbe a remoção física (`os.remove()`) de qualquer arquivo enquanto houver um manipulador de `mmap` aberto nele.
* **Tratamento Arquitetural:**
  1. O `PersistenceManager` mantém o registro dos arquivos temporários de `memmap`.
  2. Na liberação ou limpeza do workspace (`cleanup`), deve-se fechar explicitamente a visualização (`del buffer._array` ou chamar `flush()` / coleta de lixo) antes de executar a exclusão do arquivo no disco.

---

## 5. Arquitetura de Implementação Proposta: `MemmapBuffer`

### 5.1. Estrutura do `MemmapBuffer` em `src/anicrop/buffer.py`
```python
class MemmapBuffer(AbstractImageBuffer):
    """Adaptador de buffer baseado em arquivo temporário binário mapeado em memória (np.memmap)."""

    def __init__(
        self,
        file_path: Path | str,
        shape: tuple[int, ...],
        dtype: np.dtype = np.uint8,
        mode: str = "r+",
    ) -> None:
        self._path = Path(file_path)
        self._mmap: np.memmap = np.memmap(
            self._path,
            dtype=dtype,
            mode=mode,
            shape=shape,
        )

    @classmethod
    def from_array(cls, array: np.ndarray, workspace_path: Path) -> MemmapBuffer:
        """Transfere um ndarray existente da RAM para um arquivo memmap em disco."""
        temp_file = workspace_path / f"{uuid.uuid4().hex}.dat"
        mmap_inst = np.memmap(temp_file, dtype=array.dtype, mode="w+", shape=array.shape)
        mmap_inst[...] = array
        mmap_inst.flush()
        return cls(temp_file, array.shape, array.dtype, mode="r+")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._mmap.shape

    @property
    def dtype(self) -> np.dtype:
        return self._mmap.dtype

    @property
    def ndim(self) -> int:
        return self._mmap.ndim

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._mmap[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        self._mmap[key] = value

    def flush(self) -> None:
        """Sincroniza modificações com o disco."""
        self._mmap.flush()

    def close(self) -> None:
        """Fecha o mapeamento permitindo exclusão segura no Windows."""
        if hasattr(self, "_mmap"):
            del self._mmap
```

---

## 6. Fluxo Atualizado no Construtor e Fábricas de `Image`

```text
Entrada: ndarray ou Image.new(size)
                   │
                   ▼
       width * height > threshold?
        ├── NÃO ──► ArrayBuffer (RAM)
        └── SIM  ──► Escolha do Backend:
                     ├── Imagem somente leitura (ex: Vips disponível) ──► VipsStreamingBuffer
                     └── Imagem mutável / Genérica ─────────────────────► MemmapBuffer (disco)
```

1. **Em `Image.__init__`:**
   * Se o `image` passado for um `np.ndarray` e o seu tamanho exceder `threshold_pixels`:
     * Converte transparentemente para `MemmapBuffer` no workspace gerenciado pelo `manager_global`.
     * O `ndarray` de origem em RAM pode ser desalocado pelo chamador, preservando a memória física.
2. **Em `Image.new`:**
   * Se `size` exceder `threshold_pixels`:
     * Aloca um `MemmapBuffer` inicializado com zeros ou a cor solicitada diretamente no arquivo binário em disco.

---

## 7. Roteiro de Migração e Testes

- [ ] **Fase 1:** Implementar a classe `MemmapBuffer` em `src/anicrop/buffer.py` compatível com `AbstractImageBuffer`.
- [ ] **Fase 2:** Integrar `MemmapBuffer` no `PersistenceManager` para exclusão e gerenciamento seguro de ciclo de vida em Windows/Linux.
- [ ] **Fase 3:** Atualizar `Image.__init__` e `Image.new` para substituir a instanciação de `ZarrBuffer` por `MemmapBuffer`.
- [ ] **Fase 4:** Migrar testes unitários existentes de `test_image_zarr_factory.py` para validar `MemmapBuffer`.
- [ ] **Fase 5:** Remover `zarr` e `numcodecs` das dependências em `pyproject.toml`.
