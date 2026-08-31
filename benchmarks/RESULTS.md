# Relatório Oficial de Benchmarks: anicrop vs Concorrentes

> **Data de Execução:** 2026-08-31 04:00:50
> **Ambiente:** Python 3.12+, uv virtualenv, Linux
> **Competidores:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`

## 📊 Sumário Executivo por Cenário

### Composição 4K (8 Camadas - Interpolação Bilinear Pareada)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 304.19 ms     | 3.29 FPS           | 296.77 ms  | 311.95 ms  | 169.00 MB  |
| **Pyvips**   | 468.01 ms     | 2.14 FPS           | 456.17 ms  | 480.25 ms  | 196.30 MB  |
| **Pillow**   | 741.97 ms     | 1.35 FPS           | 694.75 ms  | 774.94 ms  | 85.77 MB   |
| **OpenCV**   | 1466.71 ms    | 0.68 FPS           | 1455.78 ms | 1473.67 ms | 355.85 MB  |


### Edição por Retalhos (25 Patches em Imagem 4K)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|--------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop**  | 108.01 ms     | 9.26 FPS           | 101.58 ms | 116.25 ms | 0.02 MB    |
| **OpenCV**   | 98.24 ms      | 10.18 FPS          | 97.45 ms  | 101.31 ms | 0.00 MB    |
| **Pillow**   | 80.42 ms      | 12.44 FPS          | 79.48 ms  | 86.50 ms  | 0.00 MB    |


### Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 513.29 ms     | 1.95 FPS           | 507.88 ms  | 520.38 ms  | 126.57 MB  |
| **OpenCV**   | 9480.56 ms    | 0.11 FPS           | 9441.12 ms | 9517.97 ms | 1645.32 MB |
| **Pillow**   | 1270.71 ms    | 0.79 FPS           | 1269.42 ms | 1272.36 ms | 347.57 MB  |


### Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)

| Biblioteca         | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|--------------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop (Zarr)** | 115.97 ms     | 8.62 FPS           | 114.25 ms | 117.52 ms | 136.57 MB  |
| **Pyvips**         | 196.01 ms     | 5.10 FPS           | 190.33 ms | 200.65 ms | 61.05 MB   |


### Edição e Composição em Foto 100MP Real (moon_10k.jpg)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 227.29 ms     | 4.40 FPS           | 211.70 ms  | 236.80 ms  | 129.69 MB  |
| **Pyvips**   | 265.35 ms     | 3.77 FPS           | 252.79 ms  | 279.89 ms  | 210.64 MB  |
| **Pillow**   | 5777.54 ms    | 0.17 FPS           | 5759.52 ms | 5805.71 ms | 287.75 MB  |


## 📈 Gráficos Comparativos

![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)
