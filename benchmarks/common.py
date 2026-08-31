from dataclasses import dataclass
import gc
from pathlib import Path
import time
import tracemalloc
from typing import Callable, Any
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


def save_result_image(scenario_dir_name: str, library_name: str, result: Any) -> Path:
    """Salva a imagem gerada pelo teste no diretório dedicado do cenário."""
    target_dir = OUTPUT_DIR / scenario_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    clean_name = library_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = target_dir / f"{clean_name}.png"

    if hasattr(result, "save"):
        # anicrop.image.Image ou PIL.Image
        result.save(out_path)
    elif isinstance(result, np.ndarray):
        # OpenCV numpy array
        import cv2

        cv2.imwrite(str(out_path), result)
    elif hasattr(result, "write_to_file"):
        # pyvips.Image
        result.write_to_file(str(out_path))

    return out_path


@dataclass
class BenchmarkResult:
    library: str
    scenario: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps: float
    peak_ram_mb: float
    extra_info: str = ""


def measure_peak_rss(func: Callable[[], Any], samples_interval: float = 0.001) -> float:
    """Executa func() monitorando em background o pico real de RSS do processo no SO."""
    import os
    import psutil
    import threading

    gc.collect()
    proc = psutil.Process(os.getpid())
    base_rss = proc.memory_info().rss
    peak_rss = base_rss
    stop_event = threading.Event()

    def sampler() -> None:
        nonlocal peak_rss
        while not stop_event.is_set():
            try:
                m = proc.memory_info().rss
                if m > peak_rss:
                    peak_rss = m
            except Exception:
                pass
            time.sleep(samples_interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        res = func()
    finally:
        stop_event.set()
        t.join(timeout=0.1)
        m = proc.memory_info().rss
        if m > peak_rss:
            peak_rss = m

    del res
    gc.collect()
    return max(0.0, float(peak_rss - base_rss) / (1024 * 1024))


def measure_execution(
    name: str,
    scenario_name: str,
    func: Callable[[], Any],
    warmup: int = 2,
    iterations: int = 10,
    measure_ram: bool = True,
    extra_info: str = "",
) -> BenchmarkResult:
    """Executa a função de teste com warmup, medindo tempo com perf_counter_ns e pico de RAM no SO."""
    # 1. Warmup
    for _ in range(warmup):
        func()
        gc.collect()

    # 2. Medição de Memória Real do SO (RSS)
    peak_ram_mb = 0.0
    if measure_ram:
        peak_ram_mb = measure_peak_rss(func)

    # 3. Medição de Tempo de Execução
    times_ms = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1_000_000.0)

    arr = np.array(times_ms)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return BenchmarkResult(
        library=name,
        scenario=scenario_name,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps=fps,
        peak_ram_mb=peak_ram_mb,
        extra_info=extra_info,
    )
