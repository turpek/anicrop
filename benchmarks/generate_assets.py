from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_gradient_background(w: int, h: int) -> np.ndarray:
    """Gera um fundo com gradiente suave RGB e canal alfa 100% opaco."""
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    b = (np.sin(xx * np.pi) * 120 + 80).astype(np.uint8)
    g = (np.cos(yy * np.pi) * 100 + 100).astype(np.uint8)
    r = ((xx + yy) * 100 + 40).astype(np.uint8)
    a = np.full((h, w), 255, dtype=np.uint8)

    return np.dstack([b, g, r, a])


def create_character_sprite(
    w: int, h: int, color_base: tuple[int, int, int]
) -> np.ndarray:
    """Gera um sprite com canal alfa suave (círculos, elipses e polígonos)."""
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Corpo (elipse)
    cv2.ellipse(
        img,
        (w // 2, int(h * 0.65)),
        (int(w * 0.35), int(h * 0.3)),
        0,
        0,
        360,
        (*color_base, 255),
        -1,
        lineType=cv2.LINE_AA,
    )
    # Cabeça (círculo)
    cv2.circle(
        img,
        (w // 2, int(h * 0.3)),
        int(w * 0.25),
        (255, 220, 200, 255),
        -1,
        lineType=cv2.LINE_AA,
    )
    # Detalhes (olhos/acessórios)
    cv2.circle(
        img,
        (int(w * 0.42), int(h * 0.28)),
        int(w * 0.04),
        (20, 20, 20, 255),
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        img,
        (int(w * 0.58), int(h * 0.28)),
        int(w * 0.04),
        (20, 20, 20, 255),
        -1,
        lineType=cv2.LINE_AA,
    )

    return img


def create_patch(w: int, h: int) -> np.ndarray:
    """Gera um pequeno retalho com gradiente e alfa suave."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.rectangle(
        img,
        (5, 5),
        (w - 5, h - 5),
        (0, 220, 100, 230),
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "PATCH",
        (10, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return img


def main() -> None:
    print("==> Gerando assets de teste para benchmark em benchmarks/data/...")

    # 1. Backgrounds
    bg_4k_path = DATA_DIR / "background_4k.png"
    if not bg_4k_path.exists():
        print("  - Gerando background 4K (3840x2160)...")
        bg_4k = create_gradient_background(3840, 2160)
        cv2.imwrite(str(bg_4k_path), bg_4k)

    bg_8k_path = DATA_DIR / "background_8k.png"
    if not bg_8k_path.exists():
        print("  - Gerando background 8K (7680x4320)...")
        bg_8k = create_gradient_background(7680, 4320)
        cv2.imwrite(str(bg_8k_path), bg_8k)

    # 2. Sprites com Alfa
    char1_path = DATA_DIR / "character_1.png"
    if not char1_path.exists():
        print("  - Gerando character_1 (1200x1600)...")
        char1 = create_character_sprite(1200, 1600, (40, 60, 220))
        cv2.imwrite(str(char1_path), char1)

    char2_path = DATA_DIR / "character_2.png"
    if not char2_path.exists():
        print("  - Gerando character_2 (1000x1400)...")
        char2 = create_character_sprite(1000, 1400, (180, 40, 60))
        cv2.imwrite(str(char2_path), char2)

    props_path = DATA_DIR / "props.png"
    if not props_path.exists():
        print("  - Gerando props (600x600)...")
        props = create_character_sprite(600, 600, (40, 180, 60))
        cv2.imwrite(str(props_path), props)

    patch_path = DATA_DIR / "small_patch.png"
    if not patch_path.exists():
        print("  - Gerando small_patch (200x200)...")
        patch = create_patch(200, 200)
        cv2.imwrite(str(patch_path), patch)

    # 3. Formatos Variados para Benchmark de I/O (JPEG, WebP, PNG)
    photo_4k_jpg = DATA_DIR / "photo_4k.jpg"
    photo_4k_webp = DATA_DIR / "photo_4k.webp"
    photo_1080p_jpg = DATA_DIR / "photo_1080p.jpg"
    photo_1080p_png = DATA_DIR / "photo_1080p.png"
    photo_1080p_webp = DATA_DIR / "photo_1080p.webp"

    bg_4k = None
    if not photo_4k_jpg.exists() or not photo_4k_webp.exists():
        bg_4k = create_gradient_background(3840, 2160)
        if not photo_4k_jpg.exists():
            print("  - Gerando photo_4k.jpg...")
            cv2.imwrite(str(photo_4k_jpg), bg_4k[..., :3], [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not photo_4k_webp.exists():
            print("  - Gerando photo_4k.webp...")
            cv2.imwrite(str(photo_4k_webp), bg_4k[..., :3], [cv2.IMWRITE_WEBP_QUALITY, 90])

    if not photo_1080p_jpg.exists() or not photo_1080p_png.exists() or not photo_1080p_webp.exists():
        bg_1080 = create_gradient_background(1920, 1080)
        if not photo_1080p_png.exists():
            print("  - Gerando photo_1080p.png...")
            cv2.imwrite(str(photo_1080p_png), bg_1080)
        if not photo_1080p_jpg.exists():
            print("  - Gerando photo_1080p.jpg...")
            cv2.imwrite(str(photo_1080p_jpg), bg_1080[..., :3], [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not photo_1080p_webp.exists():
            print("  - Gerando photo_1080p.webp...")
            cv2.imwrite(str(photo_1080p_webp), bg_1080[..., :3], [cv2.IMWRITE_WEBP_QUALITY, 90])

    print("✅ Assets de teste gerados com sucesso!")


if __name__ == "__main__":
    main()
