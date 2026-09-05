from typing import Any

import cv2
import numpy as np
import pyvips

from anicrop import Document, ImageFormat
from anicrop.buffer import MMapBuffer
from anicrop.enums import InterpMode
from anicrop.frame import CanvasFrame
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

GIGA_MMAP_PATH = DATA_DIR / "gigapixel_100mp.raw"
GIGA_TILE_PATH = DATA_DIR / "gigapixel_tile.png"


def setup_gigapixel_data() -> None:
    """Cria o tile procedural rico e o buffer MMap de 100MP em disco se não existirem."""
    if not GIGA_TILE_PATH.exists():
        tile_w, tile_h = 1024, 1024
        tile = np.zeros((tile_h, tile_w, 4), dtype=np.uint8)
        for y in range(tile_h):
            for x in range(tile_w):
                tile[y, x, 0] = x * 255 // tile_w
                tile[y, x, 1] = y * 255 // tile_h
                tile[y, x, 2] = (x + y) * 128 // (tile_w + tile_h)
                tile[y, x, 3] = 255

        cv2.circle(tile, (512, 512), 300, (255, 255, 255, 255), 10, lineType=cv2.LINE_AA)
        cv2.circle(tile, (512, 512), 150, (0, 255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(
            tile, (100, 100), (924, 924), (255, 0, 128, 255), 8, lineType=cv2.LINE_AA
        )
        cv2.putText(
            tile,
            "GIGAPIXEL 100MP",
            (200, 530),
            cv2.FONT_HERSHEY_DUPLEX,
            1.8,
            (0, 0, 0, 255),
            4,
            lineType=cv2.LINE_AA,
        )
        cv2.imwrite(str(GIGA_TILE_PATH), tile)

    if not GIGA_MMAP_PATH.exists():
        print("  - Criando imagem MMap de 100MP (10000x10000) em disco...")
        tile_bgra = cv2.imread(str(GIGA_TILE_PATH), cv2.IMREAD_UNCHANGED)
        tile_rgba = cv2.cvtColor(tile_bgra, cv2.COLOR_BGRA2RGBA)
        mmap_buf = MMapBuffer.create_empty(
            shape=(10000, 10000, 4), dtype=np.uint8, file_path=GIGA_MMAP_PATH
        )
        for y in range(0, 10000, 1024):
            for x in range(0, 10000, 1024):
                h = min(1024, 10000 - y)
                w = min(1024, 10000 - x)
                mmap_buf[y : y + h, x : x + w, :] = tile_rgba[:h, :w, :]
        mmap_buf.flush()


# ------------------------------------------------------------------------------
# 1. anicrop Implementation (MMap Out-of-Core + Patch Rendering)
# ------------------------------------------------------------------------------
def run_anicrop() -> Any:
    buf = MMapBuffer.open_existing(
        GIGA_MMAP_PATH, shape=(10000, 10000, 4), dtype=np.uint8, mode="r"
    )
    img = ACImage(buf, ImageFormat.RGBA)
    doc = Document("Gigapixel", width=10000, height=10000)
    layer = Layer(img, name="Gigapixel")
    doc.add(layer)
    layer.transform.rotate(15.0).scale(1.1, 1.1)

    view_region = Region.from_rect(3000, 3000, 4000, 4000)
    renderer = CanvasRender()
    frame = CanvasFrame(layer, doc.canvas, view_region=view_region)
    return renderer.render_area(layer, frame, interp=InterpMode.LINEAR)


def run_pyvips() -> Any:
    v_tile = pyvips.Image.new_from_file(str(GIGA_TILE_PATH))
    vimg = v_tile.replicate(10, 10).crop(0, 0, 10000, 10000)

    transformed = vimg.similarity(
        scale=1.1, angle=15.0, interpolate=pyvips.Interpolate.new("bilinear")
    )
    cx = transformed.width / 2.0
    cy = transformed.height / 2.0
    crop_x = int(round(cx - 2000))
    crop_y = int(round(cy - 2000))
    return transformed.crop(crop_x, crop_y, 4000, 4000)


def run_benchmark(iterations: int = 3) -> list[BenchmarkResult]:
    scenario_name = "Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)"
    dir_name = "scenario_d_gigapixel"
    setup_gigapixel_data()
    results = []

    print(f"\n--- Executando: {scenario_name} ---")

    print("  [1/2] anicrop (MMap Out-of-Core)...")
    res_anicrop = run_anicrop()
    save_result_image(dir_name, "anicrop", res_anicrop)
    results.append(
        measure_execution(
            "anicrop (MMap)", scenario_name, run_anicrop, iterations=iterations
        )
    )

    print("  [2/2] Pyvips (Streaming SIMD)...")
    res_pyvips = run_pyvips()
    save_result_image(dir_name, "pyvips", res_pyvips)
    results.append(
        measure_execution(
            "Pyvips",
            scenario_name,
            lambda: run_pyvips().write_to_memory(),
            iterations=iterations,
        )
    )

    return results


if __name__ == "__main__":
    from tabulate import tabulate

    res = run_benchmark(iterations=3)
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
