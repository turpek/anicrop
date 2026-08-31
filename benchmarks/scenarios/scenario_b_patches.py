from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image as PILImage

from anicrop.canvas import Canvas
from anicrop.image import Image as ACImage
from anicrop.layer import Layer
from anicrop.render import CanvasRender
from anicrop.spatial import Region
from benchmarks.common import (
    DATA_DIR,
    BenchmarkResult,
    measure_execution,
    save_result_image,
)

# 25 posições distribuídas pelo Canvas 4K (3840x2160)
PATCH_COORDS = [(100 + (i % 5) * 750, 100 + (i // 5) * 400) for i in range(25)]


# ------------------------------------------------------------------------------
# 1. anicrop Implementation (abre com ACImage.open)
# ------------------------------------------------------------------------------
def run_anicrop(backend: str = "vips") -> Any:
    base_img = ACImage.open(DATA_DIR / "background_4k.png", backend=backend)
    base_layer = Layer(base_img)

    patch_img = ACImage.open(DATA_DIR / "small_patch.png", backend=backend)

    for x, y in PATCH_COORDS:
        base_layer.add_edit(patch_img, Region.from_rect(x, y, 200, 200))

    renderer = CanvasRender()
    canvas = Canvas.from_size(3840, 2160)
    return renderer.render_scene([base_layer], canvas)


# ------------------------------------------------------------------------------
# 2. Pillow Implementation
# ------------------------------------------------------------------------------
def run_pillow() -> Any:
    bg = PILImage.open(DATA_DIR / "background_4k.png").convert("RGBA")
    patch = PILImage.open(DATA_DIR / "small_patch.png").convert("RGBA")

    for x, y in PATCH_COORDS:
        bg.paste(patch, (x, y), patch)

    return bg


# ------------------------------------------------------------------------------
# 3. OpenCV + NumPy Implementation
# ------------------------------------------------------------------------------
def run_opencv() -> Any:
    bg = cv2.imread(str(DATA_DIR / "background_4k.png"), cv2.IMREAD_UNCHANGED)
    patch = cv2.imread(str(DATA_DIR / "small_patch.png"), cv2.IMREAD_UNCHANGED)

    pw, ph = 200, 200
    patch_bgr = patch[:, :, :3].astype(np.float32)
    patch_alpha = (patch[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    for x, y in PATCH_COORDS:
        roi = bg[y : y + ph, x : x + pw, :3].astype(np.float32)
        blended = patch_bgr * patch_alpha + roi * (1.0 - patch_alpha)
        bg[y : y + ph, x : x + pw, :3] = blended.astype(np.uint8)

    return bg


def run_benchmark(iterations: int = 10) -> list[BenchmarkResult]:
    scenario_name = "Edição por Retalhos (25 Patches em Imagem 4K)"
    dir_name = "scenario_b_patches"
    results = []

    print(f"\n--- Executando: {scenario_name} ---")

    print("  [1/4] anicrop (Pyvips Backend)...")
    res_ac_vips = run_anicrop(backend="vips")
    save_result_image(dir_name, "anicrop_vips", res_ac_vips)
    results.append(
        measure_execution(
            "anicrop (Pyvips)",
            scenario_name,
            lambda: run_anicrop(backend="vips"),
            iterations=iterations,
        )
    )

    print("  [2/4] anicrop (OpenCV Backend)...")
    res_ac_cv = run_anicrop(backend="opencv")
    save_result_image(dir_name, "anicrop_opencv", res_ac_cv)
    results.append(
        measure_execution(
            "anicrop (OpenCV)",
            scenario_name,
            lambda: run_anicrop(backend="opencv"),
            iterations=iterations,
        )
    )

    print("  [3/4] OpenCV (NumPy)...")
    res_opencv = run_opencv()
    save_result_image(dir_name, "opencv", res_opencv)
    results.append(
        measure_execution("OpenCV", scenario_name, run_opencv, iterations=iterations)
    )

    print("  [4/4] Pillow...")
    res_pillow = run_pillow()
    save_result_image(dir_name, "pillow", res_pillow)
    results.append(
        measure_execution("Pillow", scenario_name, run_pillow, iterations=iterations)
    )

    return results


if __name__ == "__main__":
    from tabulate import tabulate

    res = run_benchmark(iterations=10)
    data = [
        [r.library, f"{r.mean_ms:.2f} ms", f"{r.fps:.2f} FPS", f"{r.peak_ram_mb:.2f} MB"]
        for r in res
    ]
    print(
        "\n"
        + tabulate(
            data,
            headers=["Biblioteca", "Tempo Médio", "FPS", "Pico RAM"],
            tablefmt="grid",
        )
    )
