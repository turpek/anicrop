from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageChops

# Garante a raiz do projeto no sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from anicrop.blend import _blend_multiply_numpy

# Tenta carregar a versão nativa em Cython/C se existir
try:
    from anicrop.native.blend import (
        blend_multiply as cy_blend_multiply,  # type: ignore[attr-defined]
    )
    _HAS_CY_MULTIPLY = True
except (ImportError, AttributeError):
    cy_blend_multiply = None
    _HAS_CY_MULTIPLY = False

# Carrega a implementação nativa de Normal como teto de referência de performance C/OpenMP
try:
    from anicrop.native.blend import blend_normal as cy_blend_normal
    _HAS_CY_NORMAL = True
except ImportError:
    cy_blend_normal = None
    _HAS_CY_NORMAL = False


@dataclass
class ScenarioResult:
    scenario: str
    resolution: str
    engine: str
    mean_ms: float
    min_ms: float
    max_ms: float
    fps: float
    throughput_mpps: float
    speedup: str = "-"
    notes: str = ""


def measure_kernel(
    func: Callable[[], Any],
    warmup: int = 3,
    iterations: int = 10,
) -> tuple[float, float, float]:
    """Mede tempo de execução médio, mínimo e máximo em milissegundos."""
    for _ in range(warmup):
        func()
        gc.collect()

    times_ms = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1_000_000.0)

    arr = np.array(times_ms)
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))


def create_test_arrays(
    height: int,
    width: int,
    base_channels: int = 4,
    overlay_channels: int = 4,
    has_alpha_variation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera matrizes aleatórias reprodutíveis com variação realista de canal alfa."""
    rng = np.random.default_rng(42)

    base = rng.integers(30, 240, size=(height, width, base_channels), dtype=np.uint8)
    overlay = rng.integers(30, 240, size=(height, width, overlay_channels), dtype=np.uint8)

    if base_channels in (2, 4) and has_alpha_variation:
        # Padrão de alfa variado na base
        base[..., -1] = rng.integers(50, 255, size=(height, width), dtype=np.uint8)

    if overlay_channels in (2, 4) and has_alpha_variation:
        # Overlay com áreas semitransparentes e áreas 100% transparentes
        overlay[..., -1] = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
        # 10% de pixels vazios (alpha == 0) para testar otimização de máscara
        zero_mask = rng.random(size=(height, width)) < 0.1
        overlay[zero_mask, -1] = 0

    return base, overlay


def run_scenario(
    name: str,
    height: int,
    width: int,
    base_channels: int = 4,
    overlay_channels: int = 4,
    opacity: float = 1.0,
    warmup: int = 3,
    iterations: int = 10,
) -> list[ScenarioResult]:
    """Executa um cenário comparando os motores disponíveis."""
    total_pixels = height * width
    res_str = f"{width}x{height} ({total_pixels / 1_000_000:.2f} MP)"
    results: list[ScenarioResult] = []

    base_src, overlay_src = create_test_arrays(
        height, width, base_channels, overlay_channels
    )

    # 1. anicrop Multiply (NumPy - Implementação atual)
    def run_numpy():
        b = base_src.copy()
        _blend_multiply_numpy(b, overlay_src, opacity)
        return b

    mean_np, min_np, max_np = measure_kernel(run_numpy, warmup=warmup, iterations=iterations)
    fps_np = 1000.0 / mean_np if mean_np > 0 else 0.0
    th_np = (total_pixels / 1_000_000.0) / (mean_np / 1000.0) if mean_np > 0 else 0.0

    results.append(
        ScenarioResult(
            scenario=name,
            resolution=res_str,
            engine="anicrop (NumPy atual)",
            mean_ms=mean_np,
            min_ms=min_np,
            max_ms=max_np,
            fps=fps_np,
            throughput_mpps=th_np,
            speedup="1.0x (Base)",
            notes="W3C Porter-Duff com Alpha",
        )
    )

    # 2. anicrop Multiply (Cython / C nativo - quando implementado)
    if _HAS_CY_MULTIPLY and cy_blend_multiply is not None:
        def run_cy():
            b = base_src.copy()
            cy_blend_multiply(b, overlay_src, opacity)
            return b

        mean_cy, min_cy, max_cy = measure_kernel(run_cy, warmup=warmup, iterations=iterations)
        fps_cy = 1000.0 / mean_cy if mean_cy > 0 else 0.0
        th_cy = (total_pixels / 1_000_000.0) / (mean_cy / 1000.0) if mean_cy > 0 else 0.0
        speedup_cy = f"{mean_np / mean_cy:.1f}x mais rápido"

        # Validação numérica de integridade
        res_np = run_numpy()
        res_c = run_cy()
        max_diff = int(np.max(np.abs(res_np.astype(np.int16) - res_c.astype(np.int16))))
        parity_note = f"Paridade C: diff máx={max_diff} LSB"

        results.append(
            ScenarioResult(
                scenario=name,
                resolution=res_str,
                engine="anicrop (Cython/C nativo)",
                mean_ms=mean_cy,
                min_ms=min_cy,
                max_ms=max_cy,
                fps=fps_cy,
                throughput_mpps=th_cy,
                speedup=speedup_cy,
                notes=parity_note,
            )
        )
    else:
        results.append(
            ScenarioResult(
                scenario=name,
                resolution=res_str,
                engine="anicrop (Cython/C planejado)",
                mean_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                fps=0.0,
                throughput_mpps=0.0,
                speedup="Pendente",
                notes="Será ativado com blend_multiply em blend.pyx",
            )
        )

    # 3. Referência Teto C: anicrop Normal (Cython/OpenMP)
    if _HAS_CY_NORMAL and cy_blend_normal is not None:
        def run_cy_normal():
            b = base_src.copy()
            cy_blend_normal(b, overlay_src, opacity)
            return b

        mean_norm, min_norm, max_norm = measure_kernel(
            run_cy_normal, warmup=warmup, iterations=iterations
        )
        fps_norm = 1000.0 / mean_norm if mean_norm > 0 else 0.0
        th_norm = (total_pixels / 1_000_000.0) / (mean_norm / 1000.0) if mean_norm > 0 else 0.0
        speedup_norm = f"{mean_np / mean_norm:.1f}x pot."

        results.append(
            ScenarioResult(
                scenario=name,
                resolution=res_str,
                engine="Referência C: Normal (Cython)",
                mean_ms=mean_norm,
                min_ms=min_norm,
                max_ms=max_norm,
                fps=fps_norm,
                throughput_mpps=th_norm,
                speedup=speedup_norm,
                notes="Teto teórico Cython OpenMP",
            )
        )

    # 4. Baseline Alternativo: Pillow ImageChops.multiply
    # Nota: Pillow multiply multiplica canais brutos (não faz Porter-Duff Over de alpha com multiply)
    try:
        pil_mode = "RGBA" if base_channels == 4 else "RGB"
        if base_channels == overlay_channels:
            pil_base = PILImage.fromarray(base_src, mode=pil_mode)
            pil_over = PILImage.fromarray(overlay_src, mode=pil_mode)

            def run_pillow():
                return ImageChops.multiply(pil_base, pil_over)

            mean_pil, min_pil, max_pil = measure_kernel(run_pillow, warmup=warmup, iterations=iterations)
            fps_pil = 1000.0 / mean_pil if mean_pil > 0 else 0.0
            th_pil = (total_pixels / 1_000_000.0) / (mean_pil / 1000.0) if mean_pil > 0 else 0.0
            speedup_pil = f"{mean_np / mean_pil:.1f}x"

            results.append(
                ScenarioResult(
                    scenario=name,
                    resolution=res_str,
                    engine="Baseline: Pillow ImageChops",
                    mean_ms=mean_pil,
                    min_ms=min_pil,
                    max_ms=max_pil,
                    fps=fps_pil,
                    throughput_mpps=th_pil,
                    speedup=speedup_pil,
                    notes="C (apenas canais brutos, sem Porter-Duff)",
                )
            )
    except Exception:
        pass

    # 5. Baseline Alternativo: OpenCV cv2.multiply (apenas para canais iguais, sem alpha blend)
    if base_channels == overlay_channels:
        def run_opencv():
            return cv2.multiply(base_src, overlay_src, scale=1.0 / 255.0)

        mean_cv, min_cv, max_cv = measure_kernel(run_opencv, warmup=warmup, iterations=iterations)
        fps_cv = 1000.0 / mean_cv if mean_cv > 0 else 0.0
        th_cv = (total_pixels / 1_000_000.0) / (mean_cv / 1000.0) if mean_cv > 0 else 0.0
        speedup_cv = f"{mean_np / mean_cv:.1f}x"

        results.append(
            ScenarioResult(
                scenario=name,
                resolution=res_str,
                engine="Baseline: OpenCV cv2.multiply",
                mean_ms=mean_cv,
                min_ms=min_cv,
                max_ms=max_cv,
                fps=fps_cv,
                throughput_mpps=th_cv,
                speedup=speedup_cv,
                notes="C++ SIMD (apenas canais brutos)",
            )
        )

    return results


def print_table(results: list[ScenarioResult]) -> None:
    headers = [
        "Cenário",
        "Motor / Backend",
        "Tempo",
        "FPS",
        "Speedup",
    ]

    rows = []
    current_scenario = ""
    for r in results:
        scen_col = r.scenario if r.scenario != current_scenario else ""
        current_scenario = r.scenario

        time_str = f"{r.mean_ms:.2f} ms" if r.mean_ms > 0 else "—"
        if r.fps <= 0:
            fps_str = "—"
        elif r.fps >= 1000:
            fps_str = f"{r.fps:.0f}"
        else:
            fps_str = f"{r.fps:.1f}"

        # Abrevia o nome do motor para caber em terminais estreitos
        engine = r.engine
        if "NumPy" in engine:
            engine_str = "anicrop (NumPy)"
        elif "Cython/C nativo" in engine:
            engine_str = "anicrop (Cython/C)"
        elif "planejado" in engine.lower():
            engine_str = "anicrop (Planejado)"
        elif "Normal" in engine:
            engine_str = "Ref: Normal (C)"
        elif "Pillow" in engine:
            engine_str = "Base: Pillow (C)"
        elif "OpenCV" in engine:
            engine_str = "Base: OpenCV (SIMD)"
        else:
            engine_str = engine[:19]

        rows.append([
            scen_col,
            engine_str,
            time_str,
            fps_str,
            r.speedup,
        ])

    col_widths = [
        max(len(row[i]) for row in [headers] + rows) + 2 for i in range(len(headers))
    ]

    separator = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_str = "|" + "|".join(f" {h}".ljust(w) for h, w in zip(headers, col_widths)) + "|"

    print(separator)
    print(header_str)
    print(separator)

    last_scen = ""
    for row in rows:
        if row[0] != "" and last_scen != "" and row[0] != last_scen:
            print(separator)
        last_scen = row[0] if row[0] != "" else last_scen
        line = "|" + "|".join(f" {cell}".ljust(w) for cell, w in zip(row, col_widths)) + "|"
        print(line)

    print(separator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark do Modo de Mesclagem MULTIPLY (NumPy vs Cython/C)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Executa versão rápida com menos iterações e resoluções reduzidas",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Número de iterações por medição (padrão: 10 normal, 3 quick)",
    )
    args = parser.parse_args()

    iterations = args.iterations or (3 if args.quick else 10)
    warmup = 1 if args.quick else 3

    print("=" * 78)
    print("  ANICROP — Benchmark Oficial do Modo de Mesclagem MULTIPLY")
    print("  Comparativo: NumPy (Atual) vs Cython/C (Planejado) vs Baselines Nativos")
    status_c = "DISPONÍVEL" if _HAS_CY_MULTIPLY else "NÃO COMPILADO (Simulado via NumPy + Teto Normal C)"
    print(f"  Status do Backend Cython MULTIPLY: {status_c}")
    print("=" * 78)
    print()

    all_results: list[ScenarioResult] = []

    # 1. 1080p RGBA sobre RGBA
    print(">>> [1/5] Executando Cenário: 1080p Full HD (RGBA sobre RGBA)...")
    all_results.extend(
        run_scenario(
            "1. 1080p RGBA/RGBA",
            1080,
            1920,
            base_channels=4,
            overlay_channels=4,
            opacity=1.0,
            warmup=warmup,
            iterations=iterations,
        )
    )

    # 2. 1080p RGBA sobre RGB (Fundo sólido)
    print(">>> [2/5] Executando Cenário: 1080p Full HD (RGBA sobre RGB Sólido)...")
    all_results.extend(
        run_scenario(
            "2. 1080p RGBA/RGB",
            1080,
            1920,
            base_channels=3,
            overlay_channels=4,
            opacity=1.0,
            warmup=warmup,
            iterations=iterations,
        )
    )

    # 3. 1080p RGBA sobre RGBA com Modulação de Opacidade (0.7)
    print(">>> [3/5] Executando Cenário: 1080p Full HD com Opacity=0.7...")
    all_results.extend(
        run_scenario(
            "3. 1080p Opac 0.7",
            1080,
            1920,
            base_channels=4,
            overlay_channels=4,
            opacity=0.7,
            warmup=warmup,
            iterations=iterations,
        )
    )

    # 4. Patch Local 512x512 RGBA
    print(">>> [4/5] Executando Cenário: Patch Local 512x512 RGBA...")
    all_results.extend(
        run_scenario(
            "4. Patch 512x512",
            512,
            512,
            base_channels=4,
            overlay_channels=4,
            opacity=1.0,
            warmup=warmup,
            iterations=iterations,
        )
    )

    # 5. 4K UHD RGBA sobre RGB (apenas se não for quick)
    if not args.quick:
        print(">>> [5/5] Executando Cenário: 4K UHD 3840x2160 (RGBA sobre RGB)...")
        all_results.extend(
            run_scenario(
                "5. 4K RGBA/RGB",
                2160,
                3840,
                base_channels=3,
                overlay_channels=4,
                opacity=1.0,
                warmup=warmup,
                iterations=max(2, iterations // 2),
            )
        )
    else:
        print(">>> [5/5] Cenário 4K UHD ignorado no modo --quick.")

    print()
    print_table(all_results)
    print()

    print("Notas Técnicas:")
    print("  • 'anicrop (NumPy)': Implementação pura com Porter-Duff e float32.")
    print("  • 'anicrop (Cython/C)': Valida paridade numérica automática quando ativo.")
    print("  • 'Ref: Normal (C)': Motor Cython com OpenMP (teto de performance em C).")
    print("  • 'Pillow'/'OpenCV': Multiplicação bruta de arrays (sem Porter-Duff de alfa).")
    print()


if __name__ == "__main__":
    main()
