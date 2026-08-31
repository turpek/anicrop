from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image as PILImage
import pyvips

from anicrop import Document, ImageFormat
from anicrop.enums import InterpMode
from benchmarks.common import (
    DATA_DIR,
    BenchmarkResult,
    measure_execution,
    save_result_image,
)

LAYER_CONFIGS = [
    {
        "asset": "character_1.png",
        "x": 200,
        "y": 150,
        "scale": 1.1,
        "rot": 15.0,
        "opacity": 0.9,
    },
    {
        "asset": "character_2.png",
        "x": 1800,
        "y": 200,
        "scale": 0.9,
        "rot": -25.0,
        "opacity": 0.85,
    },
    {
        "asset": "props.png",
        "x": 800,
        "y": 900,
        "scale": 1.2,
        "rot": 45.0,
        "opacity": 1.0,
    },
    {
        "asset": "character_1.png",
        "x": 2400,
        "y": 400,
        "scale": 0.8,
        "rot": 10.0,
        "opacity": 0.75,
    },
    {
        "asset": "props.png",
        "x": 1200,
        "y": 300,
        "scale": 1.0,
        "rot": -15.0,
        "opacity": 0.95,
    },
    {
        "asset": "character_2.png",
        "x": 500,
        "y": 600,
        "scale": 1.05,
        "rot": 30.0,
        "opacity": 0.8,
    },
    {
        "asset": "props.png",
        "x": 2800,
        "y": 800,
        "scale": 1.3,
        "rot": -40.0,
        "opacity": 1.0,
    },
    {
        "asset": "character_1.png",
        "x": 1500,
        "y": 700,
        "scale": 0.95,
        "rot": 5.0,
        "opacity": 0.9,
    },
]


# ------------------------------------------------------------------------------
# 1. anicrop Implementation
# ------------------------------------------------------------------------------
def run_anicrop(interp: InterpMode = InterpMode.LINEAR) -> Any:
    doc = Document.open(DATA_DIR / "background_4k.png", name="Fundo")
    for i, cfg in enumerate(LAYER_CONFIGS):
        layer = doc.load_layer(DATA_DIR / cfg["asset"], name=f"L_{i}")
        layer.transform.rotate(cfg["rot"]).scale(cfg["scale"], cfg["scale"]).translate(
            cfg["x"], cfg["y"]
        )
        layer.opacity = cfg["opacity"]
    return doc.render(format=ImageFormat.RGBA, interp=interp)


# ------------------------------------------------------------------------------
# 2. Pillow Implementation (Compensa bounding box expandida e sentido horário)
# ------------------------------------------------------------------------------
def run_pillow(resample: Any = PILImage.Resampling.BILINEAR) -> Any:
    bg = PILImage.open(DATA_DIR / "background_4k.png").convert("RGBA")
    for cfg in LAYER_CONFIGS:
        sprite = PILImage.open(DATA_DIR / cfg["asset"]).convert("RGBA")
        orig_w, orig_h = sprite.size
        s = cfg["scale"]
        if s != 1.0:
            sprite = sprite.resize((int(orig_w * s), int(orig_h * s)), resample=resample)
        if cfg["rot"] != 0.0:
            sprite = sprite.rotate(-cfg["rot"], resample=resample, expand=True)
        if cfg["opacity"] < 1.0:
            r, g, b, a = sprite.split()
            a = a.point(lambda p: int(p * cfg["opacity"]))
            sprite.putalpha(a)

        cx = cfg["x"] + (orig_w * s) / 2.0
        cy = cfg["y"] + (orig_h * s) / 2.0
        paste_x = int(round(cx - sprite.width / 2.0))
        paste_y = int(round(cy - sprite.height / 2.0))

        temp = PILImage.new("RGBA", bg.size, (0, 0, 0, 0))
        temp.paste(sprite, (paste_x, paste_y))
        bg = PILImage.alpha_composite(bg, temp)
    return bg


# ------------------------------------------------------------------------------
# 3. OpenCV + NumPy Implementation (Sentido horário com pivô central)
# ------------------------------------------------------------------------------
def run_opencv(flag: int = cv2.INTER_LINEAR) -> Any:
    bg = cv2.imread(str(DATA_DIR / "background_4k.png"), cv2.IMREAD_UNCHANGED)
    h_bg, w_bg = bg.shape[:2]
    out_bgr = bg[:, :, :3].astype(np.float32)

    for cfg in LAYER_CONFIGS:
        sprite = cv2.imread(str(DATA_DIR / cfg["asset"]), cv2.IMREAD_UNCHANGED)
        hs, ws = sprite.shape[:2]
        center = (ws / 2.0, hs / 2.0)
        M_rot = cv2.getRotationMatrix2D(center, -cfg["rot"], cfg["scale"])
        M_rot[0, 2] += cfg["x"]
        M_rot[1, 2] += cfg["y"]

        warped = cv2.warpAffine(sprite, M_rot, (w_bg, h_bg), flags=flag)
        alpha = (warped[:, :, 3].astype(np.float32) / 255.0) * cfg["opacity"]
        alpha_3d = alpha[:, :, np.newaxis]

        out_bgr = warped[:, :, :3].astype(np.float32) * alpha_3d + out_bgr * (
            1.0 - alpha_3d
        )

    return out_bgr.astype(np.uint8)


# ------------------------------------------------------------------------------
# 4. Pyvips Implementation (Sentido horário com pivô central)
# ------------------------------------------------------------------------------
def run_pyvips(interp_name: str = "bilinear") -> Any:
    bg = pyvips.Image.new_from_file(str(DATA_DIR / "background_4k.png"))
    comp = bg
    for cfg in LAYER_CONFIGS:
        sprite = pyvips.Image.new_from_file(str(DATA_DIR / cfg["asset"]))
        ws, hs = sprite.width, sprite.height
        s = cfg["scale"]
        rot = cfg["rot"]

        transformed = sprite.similarity(
            scale=s,
            angle=rot,
            interpolate=pyvips.Interpolate.new(interp_name),
        )
        if cfg["opacity"] < 1.0:
            bands = transformed.bandsplit()
            alpha = bands[3] * cfg["opacity"]
            transformed = bands[0].bandjoin([bands[1], bands[2], alpha])

        cx = cfg["x"] + (ws * s) / 2.0
        cy = cfg["y"] + (hs * s) / 2.0
        paste_x = int(round(cx - transformed.width / 2.0))
        paste_y = int(round(cy - transformed.height / 2.0))
        comp = comp.composite2(transformed, "over", x=paste_x, y=paste_y)

    return comp


def run_benchmark(iterations: int = 5) -> list[BenchmarkResult]:
    scenario_name = "Composição 4K (8 Camadas - Interpolação Bilinear Pareada)"
    dir_name = "scenario_a_multilayers"
    results = []

    print(f"\n--- Executando: {scenario_name} ---")

    print("  [1/4] anicrop (Linear)...")
    res_anicrop = run_anicrop(InterpMode.LINEAR)
    save_result_image(dir_name, "anicrop", res_anicrop)
    results.append(
        measure_execution(
            "anicrop",
            scenario_name,
            lambda: run_anicrop(InterpMode.LINEAR),
            iterations=iterations,
        )
    )

    print("  [2/4] Pyvips (Bilinear)...")
    res_pyvips = run_pyvips("bilinear")
    save_result_image(dir_name, "pyvips", res_pyvips)
    results.append(
        measure_execution(
            "Pyvips",
            scenario_name,
            lambda: run_pyvips("bilinear").write_to_memory(),
            iterations=iterations,
        )
    )

    print("  [3/4] Pillow (Bilinear)...")
    res_pillow = run_pillow(PILImage.Resampling.BILINEAR)
    save_result_image(dir_name, "pillow", res_pillow)
    results.append(
        measure_execution(
            "Pillow",
            scenario_name,
            lambda: run_pillow(PILImage.Resampling.BILINEAR),
            iterations=iterations,
        )
    )

    print("  [4/4] OpenCV (Linear)...")
    res_opencv = run_opencv(cv2.INTER_LINEAR)
    save_result_image(dir_name, "opencv", res_opencv)
    results.append(
        measure_execution(
            "OpenCV",
            scenario_name,
            lambda: run_opencv(cv2.INTER_LINEAR),
            iterations=iterations,
        )
    )

    return results


if __name__ == "__main__":
    from tabulate import tabulate

    res = run_benchmark(iterations=5)
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
