from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image as PILImage
import pyvips

# Previne warning de DOS attack no Pillow para imagens > 89MP
PILImage.MAX_IMAGE_PIXELS = None

from anicrop import Document, ImageFormat, Viewport
from anicrop.enums import InterpMode
from anicrop.spatial import Region
from anicrop.type import Scale
from benchmarks.common import (
    DATA_DIR,
    BenchmarkResult,
    measure_execution,
    save_result_image,
)

MOON_PATH = DATA_DIR / "moon_10k.jpg"

LAYERS_CONFIG = [
    {
        "asset": "character_1.png",
        "x": 4500,
        "y": 4200,
        "scale": 1.5,
        "rot": -10.0,
        "opacity": 1.0,
    },
    {
        "asset": "props.png",
        "x": 5200,
        "y": 4100,
        "scale": 1.8,
        "rot": 45.0,
        "opacity": 1.0,
    },
    {
        "asset": "character_2.png",
        "x": 5800,
        "y": 4600,
        "scale": 1.3,
        "rot": 15.0,
        "opacity": 0.95,
    },
]

VIEWPORT_SIZE = (3840, 2160)
FOCAL_POINT = (5000, 4500)


# ------------------------------------------------------------------------------
# 1. anicrop Implementation (Out-of-Core Zarr + Affine + Sprites + 4K Preview)
# ------------------------------------------------------------------------------
def run_anicrop() -> Any:
    doc = Document.open(MOON_PATH, name="Moon")
    moon_layer = doc[0]
    moon_layer.transform.rotate(15.0).scale(1.05, 1.05)

    for i, cfg in enumerate(LAYERS_CONFIG):
        layer = doc.load_layer(DATA_DIR / cfg["asset"], name=f"L_{i}")
        layer.transform.rotate(cfg["rot"]).scale(cfg["scale"], cfg["scale"]).translate(
            cfg["x"], cfg["y"]
        )
        layer.opacity = cfg["opacity"]

    viewport = Viewport(size=VIEWPORT_SIZE, fit_scale=1.0)
    viewport.scale = Scale(1.0, 1.0)
    canvas_w, canvas_h = doc.canvas.size
    viewport.region = Region.from_rect(
        FOCAL_POINT[0] - canvas_w / 2,
        FOCAL_POINT[1] - canvas_h / 2,
        VIEWPORT_SIZE[0],
        VIEWPORT_SIZE[1],
    )
    return doc.preview(viewport, interp=InterpMode.LINEAR)


# ------------------------------------------------------------------------------
# 2. Pyvips Implementation (Streaming Lazy + Similarity + Composite + 4K Crop)
# ------------------------------------------------------------------------------
def run_pyvips() -> Any:
    moon = pyvips.Image.new_from_file(str(MOON_PATH))
    transformed = moon.similarity(
        scale=1.05, angle=15.0, interpolate=pyvips.Interpolate.new("bilinear")
    )

    # Offset causado pela rotação/expansão da imagem da Lua
    dx = (transformed.width - 10000) / 2.0
    dy = (transformed.height - 10000) / 2.0

    comp = transformed
    for cfg in LAYERS_CONFIG:
        sprite = pyvips.Image.new_from_file(str(DATA_DIR / cfg["asset"]))
        ws, hs = sprite.width, sprite.height
        s = cfg["scale"]
        rot = cfg["rot"]

        t_sprite = sprite.similarity(
            scale=s, angle=rot, interpolate=pyvips.Interpolate.new("bilinear")
        )
        if cfg["opacity"] < 1.0:
            bands = t_sprite.bandsplit()
            alpha = bands[3] * cfg["opacity"]
            t_sprite = bands[0].bandjoin([bands[1], bands[2], alpha])

        cx = cfg["x"] + (ws * s) / 2.0
        cy = cfg["y"] + (hs * s) / 2.0
        paste_x = int(round(cx + dx - t_sprite.width / 2.0))
        paste_y = int(round(cy + dy - t_sprite.height / 2.0))
        comp = comp.composite2(t_sprite, "over", x=paste_x, y=paste_y)

    cx = transformed.width / 2.0
    cy = transformed.height / 2.0
    crop_x = int(round(cx - VIEWPORT_SIZE[0] / 2.0))
    crop_y = int(round(cy - VIEWPORT_SIZE[1] / 2.0))
    return comp.crop(crop_x, crop_y, VIEWPORT_SIZE[0], VIEWPORT_SIZE[1])


# ------------------------------------------------------------------------------
# 3. Pillow Implementation (Full 100MP Decompression + Transform + Composite)
# ------------------------------------------------------------------------------
def run_pillow() -> Any:
    moon = PILImage.open(MOON_PATH).convert("RGBA")
    orig_w, orig_h = moon.size
    s = 1.05
    rot = 15.0
    moon = moon.resize(
        (int(orig_w * s), int(orig_h * s)), resample=PILImage.Resampling.BILINEAR
    )
    moon = moon.rotate(-rot, resample=PILImage.Resampling.BILINEAR, expand=True)

    dx = (moon.width - 10000) / 2.0
    dy = (moon.height - 10000) / 2.0

    for cfg in LAYERS_CONFIG:
        sprite = PILImage.open(DATA_DIR / cfg["asset"]).convert("RGBA")
        sw, sh = sprite.size
        ss = cfg["scale"]
        srot = cfg["rot"]
        if ss != 1.0:
            sprite = sprite.resize(
                (int(sw * ss), int(sh * ss)), resample=PILImage.Resampling.BILINEAR
            )
        if srot != 0.0:
            sprite = sprite.rotate(
                -srot, resample=PILImage.Resampling.BILINEAR, expand=True
            )
        if cfg["opacity"] < 1.0:
            r, g, b, a = sprite.split()
            a = a.point(lambda p: int(p * cfg["opacity"]))
            sprite.putalpha(a)

        cx = cfg["x"] + (sw * ss) / 2.0
        cy = cfg["y"] + (sh * ss) / 2.0
        paste_x = int(round(cx + dx - sprite.width / 2.0))
        paste_y = int(round(cy + dy - sprite.height / 2.0))

        temp = PILImage.new("RGBA", moon.size, (0, 0, 0, 0))
        temp.paste(sprite, (paste_x, paste_y))
        moon = PILImage.alpha_composite(moon, temp)

    cx = moon.width / 2.0
    cy = moon.height / 2.0
    crop_box = (
        int(round(cx - VIEWPORT_SIZE[0] / 2.0)),
        int(round(cy - VIEWPORT_SIZE[1] / 2.0)),
        int(round(cx + VIEWPORT_SIZE[0] / 2.0)),
        int(round(cy + VIEWPORT_SIZE[1] / 2.0)),
    )
    return moon.crop(crop_box)


def run_benchmark(iterations: int = 3) -> list[BenchmarkResult]:
    scenario_name = "Edição e Composição em Foto 100MP Real (moon_10k.jpg)"
    dir_name = "scenario_e_real_gigapixel"
    results = []

    print(f"\n--- Executando: {scenario_name} ---")

    print("  [1/3] anicrop (Out-of-Core Zarr + 4K Preview)...")
    res_anicrop = run_anicrop()
    save_result_image(dir_name, "anicrop", res_anicrop)
    results.append(
        measure_execution("anicrop", scenario_name, run_anicrop, iterations=iterations)
    )

    print("  [2/3] Pyvips (Streaming SIMD)...")
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

    print("  [3/3] Pillow (Full 100MP Decompression)...")
    res_pillow = run_pillow()
    save_result_image(dir_name, "pillow", res_pillow)
    results.append(
        measure_execution("Pillow", scenario_name, run_pillow, iterations=iterations)
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
