from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image as PILImage

from anicrop import Document, Viewport
from anicrop.enums import InterpMode
from anicrop.spatial import Region
from anicrop.type import Scale
from benchmarks.common import (
    DATA_DIR,
    BenchmarkResult,
    measure_execution,
    save_result_image,
)

# 12 Camadas distribuídas pelo Canvas 8K (7680x4320)
SCENE_LAYERS = [
    # Camadas no ponto focal (visíveis no preview):
    {"asset": "character_1.png", "x": 5400, "y": 2520, "rot": 0.0, "scale": 1.0},
    {"asset": "props.png", "x": 5800, "y": 2550, "rot": 25.0, "scale": 0.7},
    {"asset": "character_2.png", "x": 6150, "y": 2600, "rot": -15.0, "scale": 0.8},
    # Camadas distantes espalhadas pelo mapa 8K (descartadas pelo culling):
    {"asset": "character_1.png", "x": 500, "y": 500, "rot": 10.0, "scale": 1.0},
    {"asset": "character_2.png", "x": 2000, "y": 1000, "rot": -15.0, "scale": 0.9},
    {"asset": "props.png", "x": 3500, "y": 2000, "rot": 45.0, "scale": 1.2},
    {"asset": "props.png", "x": 1000, "y": 3500, "rot": 0.0, "scale": 1.0},
    {"asset": "props.png", "x": 3000, "y": 3200, "rot": -10.0, "scale": 1.0},
    {"asset": "character_1.png", "x": 4500, "y": 3800, "rot": 15.0, "scale": 0.9},
    {"asset": "props.png", "x": 7000, "y": 3500, "rot": 0.0, "scale": 1.3},
    {"asset": "character_2.png", "x": 500, "y": 2500, "rot": 35.0, "scale": 1.0},
    {"asset": "props.png", "x": 6500, "y": 1500, "rot": -30.0, "scale": 0.8},
]

VIEWPORT_SIZE = (800, 600)
FOCAL_POINT = (6000, 3000)
ZOOM = 3.0


# ------------------------------------------------------------------------------
# 1. anicrop Implementation (Culling + Sub-pixel Patch Rendering)
# ------------------------------------------------------------------------------
def run_anicrop(backend: str = "vips") -> Any:
    doc = Document.open(DATA_DIR / "background_8k.png", name="Fundo", backend=backend)
    for i, cfg in enumerate(SCENE_LAYERS):
        l = doc.load_layer(DATA_DIR / cfg["asset"], name=f"L_{i}", backend=backend)
        l.transform.rotate(cfg["rot"]).scale(cfg["scale"], cfg["scale"]).translate(
            cfg["x"], cfg["y"]
        )

    viewport = Viewport(size=VIEWPORT_SIZE, fit_scale=1.0)
    viewport.scale = Scale(ZOOM, ZOOM)
    canvas_w, canvas_h = doc.canvas.size
    viewport.region = Region.from_rect(
        FOCAL_POINT[0] - canvas_w / 2,
        FOCAL_POINT[1] - canvas_h / 2,
        VIEWPORT_SIZE[0],
        VIEWPORT_SIZE[1],
    )
    return doc.preview(viewport, interp=InterpMode.LINEAR)


# ------------------------------------------------------------------------------
# 2. Pillow Implementation (Naive Full Composition + Crop & Resize)
# ------------------------------------------------------------------------------
def run_pillow() -> Any:
    bg = PILImage.open(DATA_DIR / "background_8k.png").convert("RGBA")
    for cfg in SCENE_LAYERS:
        sprite = PILImage.open(DATA_DIR / cfg["asset"]).convert("RGBA")
        orig_w, orig_h = sprite.size
        s = cfg["scale"]
        if s != 1.0:
            sprite = sprite.resize(
                (int(orig_w * s), int(orig_h * s)), resample=PILImage.Resampling.BILINEAR
            )
        if cfg["rot"] != 0.0:
            sprite = sprite.rotate(
                -cfg["rot"], resample=PILImage.Resampling.BILINEAR, expand=True
            )

        cx = cfg["x"] + orig_w / 2.0
        cy = cfg["y"] + orig_h / 2.0
        paste_x = int(round(cx - sprite.width / 2.0))
        paste_y = int(round(cy - sprite.height / 2.0))

        temp = PILImage.new("RGBA", bg.size, (0, 0, 0, 0))
        temp.paste(sprite, (paste_x, paste_y))
        bg = PILImage.alpha_composite(bg, temp)

    # Crop and scale to viewport
    fov_w = int(round(VIEWPORT_SIZE[0] / ZOOM))
    fov_h = int(round(VIEWPORT_SIZE[1] / ZOOM))
    crop_box = (
        FOCAL_POINT[0] - fov_w // 2,
        FOCAL_POINT[1] - fov_h // 2,
        FOCAL_POINT[0] + fov_w // 2,
        FOCAL_POINT[1] + fov_h // 2,
    )
    cropped = bg.crop(crop_box)
    return cropped.resize(VIEWPORT_SIZE, resample=PILImage.Resampling.BILINEAR)


# ------------------------------------------------------------------------------
# 3. OpenCV Implementation (Full Warp + Crop & Resize)
# ------------------------------------------------------------------------------
def run_opencv() -> Any:
    bg = cv2.imread(str(DATA_DIR / "background_8k.png"), cv2.IMREAD_UNCHANGED)
    h_bg, w_bg = bg.shape[:2]
    out_bgr = bg[:, :, :3].astype(np.float32)

    for cfg in SCENE_LAYERS:
        sprite = cv2.imread(str(DATA_DIR / cfg["asset"]), cv2.IMREAD_UNCHANGED)
        hs, ws = sprite.shape[:2]
        center = (ws / 2.0, hs / 2.0)
        M_rot = cv2.getRotationMatrix2D(center, -cfg["rot"], cfg["scale"])
        M_rot[0, 2] += cfg["x"]
        M_rot[1, 2] += cfg["y"]

        warped = cv2.warpAffine(sprite, M_rot, (w_bg, h_bg), flags=cv2.INTER_LINEAR)
        alpha = (warped[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
        out_bgr = warped[:, :, :3].astype(np.float32) * alpha + out_bgr * (1.0 - alpha)

    fov_w = int(round(VIEWPORT_SIZE[0] / ZOOM))
    fov_h = int(round(VIEWPORT_SIZE[1] / ZOOM))
    x1 = FOCAL_POINT[0] - fov_w // 2
    y1 = FOCAL_POINT[1] - fov_h // 2
    crop = out_bgr[y1 : y1 + fov_h, x1 : x1 + fov_w]
    return cv2.resize(crop, VIEWPORT_SIZE, interpolation=cv2.INTER_LINEAR).astype(
        np.uint8
    )


def run_benchmark(iterations: int = 3) -> list[BenchmarkResult]:
    scenario_name = "Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)"
    dir_name = "scenario_c_viewport"
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

    print("  [3/4] OpenCV (Full Warp + Crop)...")
    res_opencv = run_opencv()
    save_result_image(dir_name, "opencv", res_opencv)
    results.append(
        measure_execution("OpenCV", scenario_name, run_opencv, iterations=iterations)
    )

    print("  [4/4] Pillow (Full Comp + Crop)...")
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
