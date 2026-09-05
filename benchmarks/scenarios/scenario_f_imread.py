import os

import cv2
import numpy as np
import pyvips
from PIL import Image as PILImage

from anicrop import Image as AnicropImage
from benchmarks.common import (
    DATA_DIR,
    BenchmarkResult,
    measure_execution,
    save_result_image,
)


def _vips_to_numpy(vips_img: pyvips.Image) -> np.ndarray:
    """Converte pyvips.Image para numpy array garantindo materialização em RAM."""
    mem = vips_img.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(
        (vips_img.height, vips_img.width, vips_img.bands)
    )
    return arr


def run_benchmark(iterations: int = 10) -> list[BenchmarkResult]:
    """Executa o benchmark de decodificação e throughput de I/O (inspirado no imread_benchmark)."""
    results: list[BenchmarkResult] = []
    dir_name = "scenario_f_imread"

    # Assets de teste
    jpg_4k = DATA_DIR / "photo_4k.jpg"
    png_4k = DATA_DIR / "background_4k.png"
    webp_4k = DATA_DIR / "photo_4k.webp"
    moon_10k = DATA_DIR / "moon_10k.jpg"

    # =========================================================================
    # 1. Decodificação JPEG 4K (3840x2160)
    # =========================================================================
    scen_jpg = "Cenário F1 - Leitura JPEG 4K"
    print(f"\n==> Executando {scen_jpg}...")

    if jpg_4k.exists():
        size_bytes = os.path.getsize(jpg_4k)
        mp_count = (3840 * 2160) / 1_000_000.0

        def read_anicrop_vips():
            img = AnicropImage.open(jpg_4k, backend="vips")
            return img[...]

        def read_anicrop_cv():
            img = AnicropImage.open(jpg_4k, backend="opencv")
            return img[...]

        def read_opencv():
            return cv2.imread(str(jpg_4k), cv2.IMREAD_UNCHANGED)

        def read_pillow():
            with PILImage.open(jpg_4k) as img:
                return np.array(img)

        def read_pyvips():
            return _vips_to_numpy(pyvips.Image.new_from_file(str(jpg_4k), access="sequential"))

        # Salvar amostras visuais
        save_result_image(dir_name, "f1_anicrop_vips", read_anicrop_vips())
        save_result_image(dir_name, "f1_opencv", read_opencv())
        save_result_image(dir_name, "f1_pillow", read_pillow())

        print("  [1/5] anicrop (Pyvips)...")
        r_av = measure_execution("anicrop (Pyvips)", scen_jpg, read_anicrop_vips, iterations=iterations)
        r_av.extra_info = f"{mp_count / (r_av.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_av.mean_ms / 1000):.1f} MB/s"
        results.append(r_av)

        print("  [2/5] anicrop (OpenCV)...")
        r_ac = measure_execution("anicrop (OpenCV)", scen_jpg, read_anicrop_cv, iterations=iterations)
        r_ac.extra_info = f"{mp_count / (r_ac.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_ac.mean_ms / 1000):.1f} MB/s"
        results.append(r_ac)

        print("  [3/5] OpenCV...")
        r_cv = measure_execution("OpenCV", scen_jpg, read_opencv, iterations=iterations)
        r_cv.extra_info = f"{mp_count / (r_cv.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_cv.mean_ms / 1000):.1f} MB/s"
        results.append(r_cv)

        print("  [4/5] Pillow...")
        r_pi = measure_execution("Pillow", scen_jpg, read_pillow, iterations=iterations)
        r_pi.extra_info = f"{mp_count / (r_pi.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_pi.mean_ms / 1000):.1f} MB/s"
        results.append(r_pi)

        print("  [5/5] Pyvips...")
        r_pv = measure_execution("Pyvips", scen_jpg, read_pyvips, iterations=iterations)
        r_pv.extra_info = f"{mp_count / (r_pv.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_pv.mean_ms / 1000):.1f} MB/s"
        results.append(r_pv)

    # =========================================================================
    # 2. Decodificação PNG 4K (3840x2160)
    # =========================================================================
    scen_png = "Cenário F2 - Leitura PNG 4K"
    print(f"\n==> Executando {scen_png}...")

    if png_4k.exists():
        size_bytes = os.path.getsize(png_4k)
        mp_count = (3840 * 2160) / 1_000_000.0

        def read_png_anicrop_vips():
            img = AnicropImage.open(png_4k, backend="vips")
            return img[...]

        def read_png_anicrop_cv():
            img = AnicropImage.open(png_4k, backend="opencv")
            return img[...]

        def read_png_opencv():
            return cv2.imread(str(png_4k), cv2.IMREAD_UNCHANGED)

        def read_png_pillow():
            with PILImage.open(png_4k) as img:
                return np.array(img)

        def read_png_pyvips():
            return _vips_to_numpy(pyvips.Image.new_from_file(str(png_4k), access="sequential"))

        print("  [1/5] anicrop (Pyvips)...")
        r_av = measure_execution("anicrop (Pyvips)", scen_png, read_png_anicrop_vips, iterations=iterations)
        r_av.extra_info = f"{mp_count / (r_av.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_av.mean_ms / 1000):.1f} MB/s"
        results.append(r_av)

        print("  [2/5] anicrop (OpenCV)...")
        r_ac = measure_execution("anicrop (OpenCV)", scen_png, read_png_anicrop_cv, iterations=iterations)
        r_ac.extra_info = f"{mp_count / (r_ac.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_ac.mean_ms / 1000):.1f} MB/s"
        results.append(r_ac)

        print("  [3/5] OpenCV...")
        r_cv = measure_execution("OpenCV", scen_png, read_png_opencv, iterations=iterations)
        r_cv.extra_info = f"{mp_count / (r_cv.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_cv.mean_ms / 1000):.1f} MB/s"
        results.append(r_cv)

        print("  [4/5] Pillow...")
        r_pi = measure_execution("Pillow", scen_png, read_png_pillow, iterations=iterations)
        r_pi.extra_info = f"{mp_count / (r_pi.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_pi.mean_ms / 1000):.1f} MB/s"
        results.append(r_pi)

        print("  [5/5] Pyvips...")
        r_pv = measure_execution("Pyvips", scen_png, read_png_pyvips, iterations=iterations)
        r_pv.extra_info = f"{mp_count / (r_pv.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_pv.mean_ms / 1000):.1f} MB/s"
        results.append(r_pv)

    # =========================================================================
    # 3. Decodificação WebP 4K (3840x2160)
    # =========================================================================
    scen_webp = "Cenário F3 - Leitura WebP 4K"
    print(f"\n==> Executando {scen_webp}...")

    if webp_4k.exists():
        size_bytes = os.path.getsize(webp_4k)
        mp_count = (3840 * 2160) / 1_000_000.0

        def read_webp_anicrop_vips():
            img = AnicropImage.open(webp_4k, backend="vips")
            return img[...]

        def read_webp_anicrop_cv():
            img = AnicropImage.open(webp_4k, backend="opencv")
            return img[...]

        def read_webp_opencv():
            return cv2.imread(str(webp_4k), cv2.IMREAD_UNCHANGED)

        def read_webp_pillow():
            with PILImage.open(webp_4k) as img:
                return np.array(img)

        def read_webp_pyvips():
            return _vips_to_numpy(pyvips.Image.new_from_file(str(webp_4k), access="sequential"))

        print("  [1/5] anicrop (Pyvips)...")
        r_av = measure_execution("anicrop (Pyvips)", scen_webp, read_webp_anicrop_vips, iterations=iterations)
        r_av.extra_info = f"{mp_count / (r_av.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_av.mean_ms / 1000):.1f} MB/s"
        results.append(r_av)

        print("  [2/5] anicrop (OpenCV)...")
        r_ac = measure_execution("anicrop (OpenCV)", scen_webp, read_webp_anicrop_cv, iterations=iterations)
        r_ac.extra_info = f"{mp_count / (r_ac.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_ac.mean_ms / 1000):.1f} MB/s"
        results.append(r_ac)

        print("  [3/5] OpenCV...")
        r_cv = measure_execution("OpenCV", scen_webp, read_webp_opencv, iterations=iterations)
        r_cv.extra_info = f"{mp_count / (r_cv.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_cv.mean_ms / 1000):.1f} MB/s"
        results.append(r_cv)

        print("  [4/5] Pillow...")
        r_pi = measure_execution("Pillow", scen_webp, read_webp_pillow, iterations=iterations)
        r_pi.extra_info = f"{mp_count / (r_pi.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_pi.mean_ms / 1000):.1f} MB/s"
        results.append(r_pi)

        print("  [5/5] Pyvips...")
        r_pv = measure_execution("Pyvips", scen_webp, read_webp_pyvips, iterations=iterations)
        r_pv.extra_info = f"{mp_count / (r_pv.mean_ms / 1000):.1f} MP/s | {(size_bytes / 1024 / 1024) / (r_pv.mean_ms / 1000):.1f} MB/s"
        results.append(r_pv)

    # =========================================================================
    # 4. Shrink-on-load (JPEG 4K -> 1080p, Shrink 2x)
    # =========================================================================
    scen_shrink = "Cenário F4 - Shrink-on-load (4K -> 1080p)"
    print(f"\n==> Executando {scen_shrink}...")

    if jpg_4k.exists():
        def shrink_anicrop_vips():
            img = AnicropImage.open(jpg_4k, backend="vips", shrink=2)
            return img[...]

        def shrink_anicrop_cv():
            img = AnicropImage.open(jpg_4k, backend="opencv", shrink=2)
            return img[...]

        def shrink_opencv():
            full = cv2.imread(str(jpg_4k), cv2.IMREAD_UNCHANGED)
            return cv2.resize(full, (1920, 1080), interpolation=cv2.INTER_AREA)

        def shrink_pillow():
            with PILImage.open(jpg_4k) as img:
                img.draft("RGB", (1920, 1080))
                return np.array(img.resize((1920, 1080), PILImage.Resampling.BOX))

        def shrink_pyvips():
            vips_img = pyvips.Image.new_from_file(str(jpg_4k), shrink=2)
            return _vips_to_numpy(vips_img)

        print("  [1/5] anicrop (Pyvips Shrink)...")
        r_av = measure_execution("anicrop (Pyvips)", scen_shrink, shrink_anicrop_vips, iterations=iterations)
        results.append(r_av)

        print("  [2/5] anicrop (OpenCV)...")
        r_ac = measure_execution("anicrop (OpenCV)", scen_shrink, shrink_anicrop_cv, iterations=iterations)
        results.append(r_ac)

        print("  [3/5] OpenCV (Read + Resize)...")
        r_cv = measure_execution("OpenCV", scen_shrink, shrink_opencv, iterations=iterations)
        results.append(r_cv)

        print("  [4/5] Pillow (Draft + Resize)...")
        r_pi = measure_execution("Pillow", scen_shrink, shrink_pillow, iterations=iterations)
        results.append(r_pi)

        print("  [5/5] Pyvips (Shrink)...")
        r_pv = measure_execution("Pyvips", scen_shrink, shrink_pyvips, iterations=iterations)
        results.append(r_pv)

    # =========================================================================
    # 5. Shrink-on-load Gigapixel / Imagem Gigante (Moon 10K -> 1280x1280, Shrink 8x)
    # =========================================================================
    scen_moon = "Cenário F5 - Shrink-on-load Gigante (10K -> 1280x1280)"
    print(f"\n==> Executando {scen_moon}...")

    if moon_10k.exists():
        def shrink_moon_anicrop_vips():
            img = AnicropImage.open(moon_10k, backend="vips", shrink=8)
            return img[...]

        def shrink_moon_opencv():
            full = cv2.imread(str(moon_10k), cv2.IMREAD_UNCHANGED)
            return cv2.resize(full, (1280, 1280), interpolation=cv2.INTER_AREA)

        def shrink_moon_pillow():
            with PILImage.open(moon_10k) as img:
                img.draft("RGB", (1280, 1280))
                return np.array(img.resize((1280, 1280), PILImage.Resampling.BOX))

        def shrink_moon_pyvips():
            vips_img = pyvips.Image.new_from_file(str(moon_10k), shrink=8)
            return _vips_to_numpy(vips_img)

        print("  [1/4] anicrop (Pyvips Shrink)...")
        r_av = measure_execution("anicrop (Pyvips)", scen_moon, shrink_moon_anicrop_vips, iterations=iterations)
        results.append(r_av)

        print("  [2/4] OpenCV (Read Full + Resize)...")
        r_cv = measure_execution("OpenCV", scen_moon, shrink_moon_opencv, iterations=iterations)
        results.append(r_cv)

        print("  [3/4] Pillow (Draft + Resize)...")
        r_pi = measure_execution("Pillow", scen_moon, shrink_moon_pillow, iterations=iterations)
        results.append(r_pi)

        print("  [4/4] Pyvips (Shrink)...")
        r_pv = measure_execution("Pyvips", scen_moon, shrink_moon_pyvips, iterations=iterations)
        results.append(r_pv)

    return results


if __name__ == "__main__":
    from tabulate import tabulate

    res = run_benchmark(iterations=5)
    data = [
        [
            r.scenario,
            r.library,
            f"{r.mean_ms:.2f} ms",
            f"{r.fps:.1f} FPS",
            f"{r.peak_ram_mb:.2f} MB",
            r.extra_info,
        ]
        for r in res
    ]
    print(
        "\n"
        + tabulate(
            data,
            headers=["Cenário", "Biblioteca", "Tempo Médio", "FPS", "Pico RAM", "Throughput"],
            tablefmt="grid",
        )
    )
