from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from anicrop import Document, ImageFormat, LayerCache
from anicrop.content import FitContext
from anicrop.effect import DynamicEffect
from anicrop.filter import BlurFilter
from anicrop.image import Image
from anicrop.spatial import Region

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"


def create_base_scene(history: bool = False) -> Document:
    """Cria a cena base inspirada em assets/example.py."""
    doc = Document.open(ASSETS_DIR / "background.jpg", name="Fundo", history=history)

    doc.load_layer(ASSETS_DIR / "character.png", name="Personagem")
    chapeu = doc.load_layer(ASSETS_DIR / "hat.png", name="Chapeu")

    # Redimensiona e alinha o chapéu
    chapeu.content.resize(250, 250)
    chapeu.layout.pin((520, 196), anchor_x=0.5, anchor_y=1.0)

    # Agrupa chapéu com a personagem
    heroina = doc.combine.merge("Chapeu", name="Heroina", count=1)

    # Enquadra a heroína proporcionalmente
    fit_payload = FitContext(heroina, doc.canvas).fit_contain()
    heroina.content.fit(fit_payload)
    heroina.layout.align(doc.canvas, anchor_x=0.85, anchor_y=1.0)

    # Desfoque de profundidade de campo no fundo
    fundo = doc["Fundo"]
    fundo.bind_effect(BlurFilter(radius=3.0))

    return doc


def benchmark_static_re_render(iterations: int = 15) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cenário 1: Re-renderização contínua da mesma cena (ex: Loop Interativo / UI / Viewport)."""
    # 1. Sem Cache
    doc_no_cache = create_base_scene()
    times_no_cache = []
    # Warmup
    doc_no_cache.render(format=ImageFormat.RGBA)
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        doc_no_cache.render(format=ImageFormat.RGBA)
        t1 = time.perf_counter_ns()
        times_no_cache.append((t1 - t0) / 1_000_000.0)

    # 2. Com Cache
    doc_with_cache = create_base_scene()
    cache = LayerCache()
    for layer in doc_with_cache.stack:
        cache.register(layer)

    times_cache = []
    # Warmup / 1º Frame de Bake
    doc_with_cache.render(format=ImageFormat.RGBA, cache=cache)
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        doc_with_cache.render(format=ImageFormat.RGBA, cache=cache)
        t1 = time.perf_counter_ns()
        times_cache.append((t1 - t0) / 1_000_000.0)

    arr_nc = np.array(times_no_cache)
    arr_c = np.array(times_cache)

    return (
        {
            "name": "Sem Cache (Re-render Total)",
            "mean": float(np.mean(arr_nc)),
            "min": float(np.min(arr_nc)),
            "max": float(np.max(arr_nc)),
            "fps": 1000.0 / float(np.mean(arr_nc)),
        },
        {
            "name": "Com LayerCache (Bakes O(1))",
            "mean": float(np.mean(arr_c)),
            "min": float(np.min(arr_c)),
            "max": float(np.max(arr_c)),
            "fps": 1000.0 / float(np.mean(arr_c)),
        },
    )


def benchmark_incremental_patches(patch_count: int = 10) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cenário 2: Adição sequencial de patches/retoques sobre a personagem."""
    # Gera retalhos de teste (64x64 com alpha)
    patches = []
    for i in range(patch_count):
        arr = np.full((64, 64, 4), (50 + i * 20, 200 - i * 15, 100, 220), dtype=np.uint8)
        patches.append((Image(arr, ImageFormat.RGBA), Region.from_rect(50 + (i % 3) * 60, 50 + (i // 3) * 60, 64, 64)))

    # 1. Sem Cache: acumula patches e re-renderiza tudo do zero
    doc_nc = create_base_scene()
    times_nc = []
    doc_nc.render(format=ImageFormat.RGBA)

    char_layer_nc = doc_nc["Personagem"]

    for patch_img, patch_region in patches:
        char_layer_nc.add_edit(patch_img, patch_region)
        gc.collect()
        t0 = time.perf_counter_ns()
        doc_nc.render(format=ImageFormat.RGBA)
        t1 = time.perf_counter_ns()
        times_nc.append((t1 - t0) / 1_000_000.0)

    # 2. Com Cache: patches adicionados incrementalmente
    doc_c = create_base_scene()
    cache = LayerCache()
    for layer in doc_c.stack:
        cache.register(layer)

    times_c = []
    doc_c.render(format=ImageFormat.RGBA, cache=cache)
    char_layer_c = doc_c["Personagem"]

    for patch_img, patch_region in patches:
        char_layer_c.add_edit(patch_img, patch_region)
        gc.collect()
        t0 = time.perf_counter_ns()
        doc_c.render(format=ImageFormat.RGBA, cache=cache)
        t1 = time.perf_counter_ns()
        times_c.append((t1 - t0) / 1_000_000.0)

    arr_nc = np.array(times_nc)
    arr_c = np.array(times_c)

    return (
        {
            "name": f"Sem Cache ({patch_count} Patches Acumulados)",
            "mean": float(np.mean(arr_nc)),
            "min": float(np.min(arr_nc)),
            "max": float(np.max(arr_nc)),
            "fps": 1000.0 / float(np.mean(arr_nc)),
        },
        {
            "name": "Com LayerCache (Deltas O(1))",
            "mean": float(np.mean(arr_c)),
            "min": float(np.min(arr_c)),
            "max": float(np.max(arr_c)),
            "fps": 1000.0 / float(np.mean(arr_c)),
        },
    )


def benchmark_anifuse_injection() -> tuple[dict[str, Any], dict[str, Any]]:
    """Cenário 3: Injeção externa de Bake (Fluxo Anifuse 2-Pass com alinhamento prévio)."""
    doc = create_base_scene()
    char_layer = doc["Personagem"]

    # Simula buffer pré-alinhado pelo Anifuse
    prebaked = Image.open(ASSETS_DIR / "character.png")

    cache = LayerCache()
    cache.register(doc["Fundo"])
    cache.register(doc["Heroina"])
    cache.register(char_layer)

    # Injeta diretamente o bake externo
    cache.set_baked(char_layer, prebaked)

    # Adiciona 5 patches de inpainting
    for i in range(5):
        arr = np.full((40, 40, 4), (100, 150, 200, 255), dtype=np.uint8)
        char_layer.add_edit(Image(arr, ImageFormat.RGBA), Region.from_rect(100 + i * 30, 100 + i * 30, 40, 40))

    # Mede render com cache vs render sem cache
    times_with = []
    for _ in range(10):
        gc.collect()
        t0 = time.perf_counter_ns()
        doc.render(format=ImageFormat.RGBA, cache=cache)
        t1 = time.perf_counter_ns()
        times_with.append((t1 - t0) / 1_000_000.0)

    # Sem cache equivalente
    doc_clean = create_base_scene()
    char_clean = doc_clean["Personagem"]
    for i in range(5):
        arr = np.full((40, 40, 4), (100, 150, 200, 255), dtype=np.uint8)
        char_clean.add_edit(Image(arr, ImageFormat.RGBA), Region.from_rect(100 + i * 30, 100 + i * 30, 40, 40))

    times_without = []
    for _ in range(10):
        gc.collect()
        t0 = time.perf_counter_ns()
        doc_clean.render(format=ImageFormat.RGBA)
        t1 = time.perf_counter_ns()
        times_without.append((t1 - t0) / 1_000_000.0)

    arr_w = np.array(times_with)
    arr_wo = np.array(times_without)

    return (
        {
            "name": "Sem Injeção (Re-render Completo)",
            "mean": float(np.mean(arr_wo)),
            "min": float(np.min(arr_wo)),
            "max": float(np.max(arr_wo)),
            "fps": 1000.0 / float(np.mean(arr_wo)),
        },
        {
            "name": "Com Injeção Anifuse set_baked",
            "mean": float(np.mean(arr_w)),
            "min": float(np.min(arr_w)),
            "max": float(np.max(arr_w)),
            "fps": 1000.0 / float(np.mean(arr_w)),
        },
    )


class AnifuseSeamCutEffect(DynamicEffect):
    """Simula o filtro dinâmico de corte de borda do Anifuse."""

    def __init__(self) -> None:
        super().__init__(name="AnifuseSeamCutEffect")

    def get_padding(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        if image.height > 10 and image.width > 10:
            image[:5, :, 3] = 0
        return image


def benchmark_dynamic_effect(iterations: int = 15) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cenário 4: Efeito Dinâmico Anifuse (Blur Estático 8.0 assado + Corte Dinâmico executado a cada frame)."""
    # 1. Sem Cache: recalcula Blur 8.0 pesado + corte a cada frame
    doc_nc = create_base_scene()
    fundo_nc = doc_nc["Fundo"]
    fundo_nc.clear_effects()
    fundo_nc.bind_effect(BlurFilter(radius=8.0))
    fundo_nc.add_effect(AnifuseSeamCutEffect())

    times_nc = []
    doc_nc.render(format=ImageFormat.RGBA)
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        doc_nc.render(format=ImageFormat.RGBA)
        t1 = time.perf_counter_ns()
        times_nc.append((t1 - t0) / 1_000_000.0)

    # 2. Com Cache: Blur 8.0 pesado é assado em baked_effects; apenas o DynamicEffect roda a cada frame
    doc_c = create_base_scene()
    fundo_c = doc_c["Fundo"]
    fundo_c.clear_effects()
    fundo_c.bind_effect(BlurFilter(radius=8.0))
    fundo_c.add_effect(AnifuseSeamCutEffect())

    cache = LayerCache()
    for layer in doc_c.stack:
        cache.register(layer)

    times_c = []
    doc_c.render(format=ImageFormat.RGBA, cache=cache)
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        doc_c.render(format=ImageFormat.RGBA, cache=cache)
        t1 = time.perf_counter_ns()
        times_c.append((t1 - t0) / 1_000_000.0)

    arr_nc = np.array(times_nc)
    arr_c = np.array(times_c)

    return (
        {
            "name": "Sem Cache (Blur 8.0 + DynamicEffect Recalculados)",
            "mean": float(np.mean(arr_nc)),
            "min": float(np.min(arr_nc)),
            "max": float(np.max(arr_nc)),
            "fps": 1000.0 / float(np.mean(arr_nc)),
        },
        {
            "name": "Com LayerCache (Blur 8.0 Assado + DynamicEffect Incremental)",
            "mean": float(np.mean(arr_c)),
            "min": float(np.min(arr_c)),
            "max": float(np.max(arr_c)),
            "fps": 1000.0 / float(np.mean(arr_c)),
        },
    )


def print_table(rows: list[list[str]]) -> None:
    headers = ["Cenário", "Modo de Renderização", "Tempo Médio", "FPS", "Speedup"]
    col_widths = [max(len(row[i]) for row in [headers] + rows) + 2 for i in range(len(headers))]

    separator = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_str = "|" + "|".join(f" {h}".ljust(w) for h, w in zip(headers, col_widths)) + "|"

    print(separator)
    print(header_str)
    print(separator)
    for row in rows:
        if row[0] == "---":
            print(separator)
            continue
        line = "|" + "|".join(f" {cell}".ljust(w) for cell, w in zip(row, col_widths)) + "|"
        print(line)
    print(separator)


def main() -> None:
    print("=" * 86)
    print("  ANICROP — Benchmark Oficial do Sistema de Cache de Camadas (LayerCache)")
    print("  Cenário: Fundo 1376x768 (Blur 3.0) + Heroína 1024x1024 + Chapéu (assets/example.py)")
    print("=" * 86)
    print()

    print(">>> Executando Cenário 1: Re-renderização Contínua...")
    res_s_nc, res_s_c = benchmark_static_re_render(iterations=20)
    speedup_s = res_s_nc["mean"] / res_s_c["mean"]

    print(">>> Executando Cenário 2: Edição Incremental (10 Patches Sequenciais)...")
    res_i_nc, res_i_c = benchmark_incremental_patches(patch_count=10)
    speedup_i = res_i_nc["mean"] / res_i_c["mean"]

    print(">>> Executando Cenário 3: Injeção Externa de Bake (Anifuse 2-Pass)...")
    res_a_nc, res_a_c = benchmark_anifuse_injection()
    speedup_a = res_a_nc["mean"] / res_a_c["mean"]

    print(">>> Executando Cenário 4: Efeito Dinâmico Anifuse (DynamicEffect)...")
    res_d_nc, res_d_c = benchmark_dynamic_effect(iterations=20)
    speedup_d = res_d_nc["mean"] / res_d_c["mean"]

    print()
    rows = [
        ["1. Re-render Contínuo", res_s_nc["name"], f"{res_s_nc['mean']:.2f} ms", f"{res_s_nc['fps']:.1f}", "1.0x (Base)"],
        ["", res_s_c["name"], f"{res_s_c['mean']:.2f} ms", f"{res_s_c['fps']:.1f}", f"{speedup_s:.1f}x mais rápido"],
        ["---", "", "", "", ""],
        ["2. Edição Incremental", res_i_nc["name"], f"{res_i_nc['mean']:.2f} ms", f"{res_i_nc['fps']:.1f}", "1.0x (Base)"],
        ["", res_i_c["name"], f"{res_i_c['mean']:.2f} ms", f"{res_i_c['fps']:.1f}", f"{speedup_i:.1f}x mais rápido"],
        ["---", "", "", "", ""],
        ["3. Anifuse set_baked", res_a_nc["name"], f"{res_a_nc['mean']:.2f} ms", f"{res_a_nc['fps']:.1f}", "1.0x (Base)"],
        ["", res_a_c["name"], f"{res_a_c['mean']:.2f} ms", f"{res_a_c['fps']:.1f}", f"{speedup_a:.1f}x mais rápido"],
        ["---", "", "", "", ""],
        ["4. DynamicEffect Anifuse", res_d_nc["name"], f"{res_d_nc['mean']:.2f} ms", f"{res_d_nc['fps']:.1f}", "1.0x (Base)"],
        ["", res_d_c["name"], f"{res_d_c['mean']:.2f} ms", f"{res_d_c['fps']:.1f}", f"{speedup_d:.1f}x mais rápido"],
    ]
    print_table(rows)
    print()


if __name__ == "__main__":
    main()
