# Master Plan: Stitch CLI & Pipeline Architecture

Este documento descreve a arquitetura de alto nível para refatorar e desacoplar a ferramenta CLI de costura de frames de animes (*Stitch*), utilizando o **`anicrop`** como motor central de composição gráfica, transformações afins 2D e exportação.

---

## 1. Princípios de Design

1. **Desacoplamento de CLI e Domínio**: A camada de linha de comando (`argparse`) apenas valida e converte flags em um objeto tipado e imutável (`StitchConfig`).
2. **Eliminação de `if/else` via Registries (Strategy Pattern)**: Seleções de eixos (`VERTICAL`, `HORIZONTAL`, `MIXED`), métodos de média (`MODE`, `MEAN`) e sentidos (`POSITIVE`, `NEGATIVE`) são resolvidas via tabelas de despacho (*dispatch maps*).
3. **Pipeline Modular de Execução**: O processo de costura é construído como uma esteira de etapas (*Pipeline Steps*) configurada dinamicamente pelo `StitchPipelineBuilder`.
4. **Composição Gráfica Não-Destrutiva via `anicrop`**: A renderização final, rotação de frames, translações sub-pixel, corte de bordas e dimensionamento do canvas final são delegados ao ecossistema `Document`, `Layer`, `Composer` e `Layout`.

---

## 2. Fluxo Geral de Execução

```text
┌──────────────────┐
│  ArgumentParser  │ (CLI)
└────────┬─────────┘
         │ Converte em objeto tipado
         ▼
┌──────────────────┐
│   StitchConfig   │ (Dataclass imutável)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  StitchPipelineBuilder   │ (Lê Config e Registries)
└────────────┬─────────────┘
             │ Monta a lista de etapas
             ▼
┌──────────────────────────┐
│      StitchPipeline      │
│  ┌────────────────────┐  │
│  │ 1. ImageLoader     │  │
│  │ 2. FeatureMatcher  │  │
│  │ 3. ShiftEstimator  │  │ (Strategy do Registry)
│  │ 4. RotationStep    │  │ (Opcional se rotate > 0)
│  │ 5. BorderTrimStep  │  │ (Opcional se border > 0)
│  │ 6. AnicropComposer │  │ (Gera Document e Canvas final)
│  └────────────────────┘  │
└──────────────────────────┘
```

---

## 3. Componentes Arquiteturais

### 3.1. Objeto de Configuração (`StitchConfig`)
```python
from dataclasses import dataclass
from pathlib import Path
from anicrop.enums import AxisType, MeanMethod, SenseType

@dataclass(frozen=True)
class StitchConfig:
    files: list[Path]
    axis: AxisType
    sense: SenseType
    mean_method: MeanMethod
    num_keypoints: int = 40
    border_trim: int = 0
    rotate_threshold: float | None = None
    scale_threshold: float = 0.006
    enable_scale: bool = True
    output_path: Path | None = None
    show_window: bool = False
```

---

### 3.2. Estratégias e Registries (Eliminação de `if/else`)

```python
from typing import Protocol, Any
import numpy as np

class ShiftEstimator(Protocol):
    def estimate_shift(self, matches: list[Any], method: MeanMethod) -> tuple[float, float]:
        ...

class VerticalShiftEstimator:
    def estimate_shift(self, matches: list[Any], method: MeanMethod) -> tuple[float, float]:
        return 0.0, calculate_mean(matches, method, axis="y")

class HorizontalShiftEstimator:
    def estimate_shift(self, matches: list[Any], method: MeanMethod) -> tuple[float, float]:
        return calculate_mean(matches, method, axis="x"), 0.0

class MixedShiftEstimator:
    def estimate_shift(self, matches: list[Any], method: MeanMethod) -> tuple[float, float]:
        return calculate_mean(matches, method, axis="xy")

AXIS_ESTIMATORS: dict[AxisType, type[ShiftEstimator]] = {
    AxisType.VERTICAL: VerticalShiftEstimator,
    AxisType.HORIZONTAL: HorizontalShiftEstimator,
    AxisType.MIXED: MixedShiftEstimator,
}
```

---

### 3.3. Pipeline Context & Steps

```python
@dataclass
class StitchContext:
    config: StitchConfig
    images: list[np.ndarray] = field(default_factory=list)
    matrices: list[np.ndarray] = field(default_factory=list)
    result_document: Document | None = None

class PipelineStep(Protocol):
    def execute(self, context: StitchContext) -> None:
        ...
```

---

### 3.4. O `StitchPipelineBuilder`

```python
class StitchPipelineBuilder:
    @classmethod
    def build(cls, config: StitchConfig) -> StitchPipeline:
        pipeline = StitchPipeline()

        # Etapas do Pipeline
        pipeline.add_step(ImageLoaderStep(config.files))
        pipeline.add_step(FeatureMatchingStep(config.num_keypoints))
        
        # Despacho via Registry sem IF/ELSE
        estimator_cls = AXIS_ESTIMATORS[config.axis]
        pipeline.add_step(ShiftEstimationStep(estimator=estimator_cls(), method=config.mean_method))

        if config.rotate_threshold is not None and config.rotate_threshold > 0:
            pipeline.add_step(RotationCompensationStep(threshold=config.rotate_threshold))

        if config.border_trim > 0:
            pipeline.add_step(BorderTrimStep(border_px=config.border_trim))

        pipeline.add_step(AnicropCompositorStep())

        return pipeline
```

---

### 3.5. Integração com o Motor `anicrop`

```python
from anicrop.document import Document
from anicrop.layer import Layer
from anicrop.image import Image
from anicrop.enums import ImageFormat

class AnicropCompositorStep:
    def execute(self, context: StitchContext) -> None:
        doc = Document("StitchPanorama", width=1920, height=1080, history=False)

        for img_data, matrix in zip(context.images, context.matrices):
            layer = doc.add(Layer(Image(img_data, ImageFormat.RGBA)))
            
            # Aplica translação e rotação acumulada
            dx, dy = int(round(matrix[0, 2])), int(round(matrix[1, 2]))
            layer.transform.translate(dx, dy)
            
            if has_rotation(matrix):
                angle = extract_angle(matrix)
                layer.transform.rotate(angle)

        # Enquadra o Canvas para abarcar todo o panorama gerado
        doc.layout.fit_content(doc.canvas, doc.stack)
        context.result_document = doc
```
