from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tabulate import tabulate

from benchmarks.common import DATA_DIR, BenchmarkResult
from benchmarks.generate_assets import main as generate_assets_main
from benchmarks.scenarios.scenario_a_multilayers import run_benchmark as run_scenario_a
from benchmarks.scenarios.scenario_b_patches import run_benchmark as run_scenario_b
from benchmarks.scenarios.scenario_c_viewport import run_benchmark as run_scenario_c
from benchmarks.scenarios.scenario_d_gigapixel import run_benchmark as run_scenario_d
from benchmarks.scenarios.scenario_e_real_gigapixel import (
    run_benchmark as run_scenario_e,
)

BENCHMARK_DIR = Path(__file__).parent
RESULTS_MD = BENCHMARK_DIR / "RESULTS.md"
CHART_PNG = BENCHMARK_DIR / "benchmark_comparison.png"


def plot_results(all_results: list[BenchmarkResult]) -> None:
    """Gera gráficos de barras comparativos de Tempo de Execução e Consumo de Memória."""
    scenarios = list(dict.fromkeys(r.scenario for r in all_results))
    fig, axes = plt.subplots(
        len(scenarios), 2, figsize=(14, 4 * len(scenarios)), dpi=120
    )
    if len(scenarios) == 1:
        axes = np.array([axes])

    palette = {
        "anicrop": "#2b5c8f",
        "anicrop (Zarr)": "#2b5c8f",
        "Pillow": "#e26d5c",
        "OpenCV": "#38b000",
        "Pyvips": "#7209b7",
    }

    for i, scen in enumerate(scenarios):
        scen_res = [r for r in all_results if r.scenario == scen]
        libs = [r.library for r in scen_res]
        times = [r.mean_ms for r in scen_res]
        rams = [r.peak_ram_mb for r in scen_res]
        colors = [palette.get(lib, "#888888") for lib in libs]

        # 1. Gráfico de Tempo (ms)
        ax_time = axes[i, 0]
        bars_time = ax_time.barh(libs, times, color=colors, height=0.55)
        ax_time.set_title(
            f"{scen}\nTempo de Execução Médio (menor é melhor)",
            fontsize=11,
            fontweight="bold",
        )
        ax_time.set_xlabel("Milissegundos (ms)")
        ax_time.grid(axis="x", linestyle="--", alpha=0.6)
        for bar in bars_time:
            w = bar.get_width()
            ax_time.text(
                w + (max(times) * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f} ms",
                va="center",
                fontsize=9,
            )

        # 2. Gráfico de Memória (MB)
        ax_ram = axes[i, 1]
        bars_ram = ax_ram.barh(libs, rams, color=colors, height=0.55)
        ax_ram.set_title(
            f"{scen}\nPico de Memória RAM (menor é melhor)",
            fontsize=11,
            fontweight="bold",
        )
        ax_ram.set_xlabel("Megabytes (MB)")
        ax_ram.grid(axis="x", linestyle="--", alpha=0.6)
        for bar in bars_ram:
            w = bar.get_width()
            ax_ram.text(
                w + (max(rams) * 0.02 if max(rams) > 0 else 0.5),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f} MB",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(CHART_PNG)
    print(f"📊 Gráfico comparativo salvo em: {CHART_PNG}")


def generate_markdown_report(all_results: list[BenchmarkResult]) -> None:
    """Gera um relatório completo em Markdown detalhando todas as métricas."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scenarios = list(dict.fromkeys(r.scenario for r in all_results))

    md_lines = [
        "# Relatório Oficial de Benchmarks: anicrop vs Concorrentes",
        f"\n> **Data de Execução:** {now_str}",
        "> **Ambiente:** Python 3.12+, uv virtualenv, Linux",
        "> **Competidores:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`\n",
        "## 📊 Sumário Executivo por Cenário\n",
    ]

    for scen in scenarios:
        scen_res = [r for r in all_results if r.scenario == scen]
        md_lines.append(f"### {scen}\n")
        table_data = [
            [
                f"**{r.library}**",
                f"{r.mean_ms:.2f} ms",
                f"{r.fps:.2f} FPS",
                f"{r.min_ms:.2f} ms",
                f"{r.max_ms:.2f} ms",
                f"{r.peak_ram_mb:.2f} MB",
            ]
            for r in scen_res
        ]
        headers = [
            "Biblioteca",
            "Tempo Médio",
            "Throughput (FPS)",
            "Min",
            "Max",
            "Pico RAM",
        ]
        md_lines.append(tabulate(table_data, headers=headers, tablefmt="github"))
        md_lines.append("\n")

    md_lines.append("## 📈 Gráficos Comparativos\n")
    md_lines.append("![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)\n")

    RESULTS_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"📝 Relatório Markdown salvo em: {RESULTS_MD}")


def main() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Bateria Completa de Benchmarks: anicrop vs Concorrentes[/bold cyan]"
        )
    )

    # 1. Garante que os assets de teste existem
    generate_assets_main()

    # 2. Executa todos os cenários
    all_results: list[BenchmarkResult] = []

    all_results.extend(run_scenario_a(iterations=5))
    all_results.extend(run_scenario_b(iterations=8))
    all_results.extend(run_scenario_c(iterations=3))
    all_results.extend(run_scenario_d(iterations=3))
    all_results.extend(run_scenario_e(iterations=3))

    # 3. Exibe tabela consolidada no terminal via Rich
    console.print("\n[bold green]=== RESULTADOS CONSOLIDADOS ===[/bold green]\n")
    scenarios = list(dict.fromkeys(r.scenario for r in all_results))

    for scen in scenarios:
        table = Table(title=scen, title_style="bold magenta")
        table.add_column("Biblioteca", style="cyan", no_wrap=True)
        table.add_column("Tempo Médio (ms)", justify="right", style="green")
        table.add_column("Throughput (FPS)", justify="right", style="yellow")
        table.add_column("Pico de RAM (MB)", justify="right", style="red")

        for r in [res for res in all_results if res.scenario == scen]:
            table.add_row(
                r.library,
                f"{r.mean_ms:.2f} ms",
                f"{r.fps:.2f} FPS",
                f"{r.peak_ram_mb:.2f} MB",
            )
        console.print(table)
        console.print("")

    # 4. Gera relatório Markdown e Gráficos
    generate_markdown_report(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    main()
