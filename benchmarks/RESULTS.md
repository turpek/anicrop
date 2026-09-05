# Avaliação de Desempenho do anicrop em Pipelines de Renderização 2D

> **Data de Execução:** 2026-09-05 02:10:57
> **Ambiente:** Python 3.12+, uv virtualenv, Linux
> **Bibliotecas Analisadas:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`

## 📊 Sumário Executivo por Cenário

### Composição 4K (8 Camadas - Interpolação Bilinear Pareada)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 333.60 ms     | 3.00 FPS           | 328.86 ms  | 336.76 ms  | 114.48 MB  |
| **anicrop (OpenCV)** | 261.45 ms     | 3.82 FPS           | 260.03 ms  | 263.54 ms  | 0.00 MB    |
| **Pyvips**           | 262.19 ms     | 3.81 FPS           | 254.75 ms  | 274.82 ms  | 2.29 MB    |
| **Pillow**           | 706.20 ms     | 1.42 FPS           | 651.05 ms  | 740.21 ms  | 0.00 MB    |
| **OpenCV**           | 1480.63 ms    | 0.68 FPS           | 1467.53 ms | 1492.97 ms | 355.85 MB  |


### Edição por Retalhos (25 Patches em Imagem 4K)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|----------------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop (Pyvips)** | 120.55 ms     | 8.30 FPS           | 99.52 ms  | 170.74 ms | 0.02 MB    |
| **anicrop (OpenCV)** | 106.66 ms     | 9.38 FPS           | 102.83 ms | 113.31 ms | 0.00 MB    |
| **OpenCV**           | 97.69 ms      | 10.24 FPS          | 95.67 ms  | 105.46 ms | 0.00 MB    |
| **Pillow**           | 80.07 ms      | 12.49 FPS          | 79.47 ms  | 83.19 ms  | 0.00 MB    |


### Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 515.00 ms     | 1.94 FPS           | 501.01 ms  | 538.86 ms  | 126.57 MB  |
| **anicrop (OpenCV)** | 570.22 ms     | 1.75 FPS           | 568.61 ms  | 572.02 ms  | 253.14 MB  |
| **OpenCV**           | 8643.10 ms    | 0.12 FPS           | 8624.09 ms | 8673.22 ms | 1645.06 MB |
| **Pillow**           | 1236.51 ms    | 0.81 FPS           | 1234.99 ms | 1237.69 ms | 253.00 MB  |


### Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)

| Biblioteca         | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|--------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (Zarr)** | 79.35 ms      | 12.60 FPS          | 77.79 ms | 81.96 ms | 75.77 MB   |
| **Pyvips**         | 90.10 ms      | 11.10 FPS          | 84.14 ms | 97.76 ms | 0.02 MB    |


### Edição e Composição em Foto 100MP Real (moon_10k.jpg)

| Biblioteca                | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|---------------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)**      | 143.64 ms     | 6.96 FPS           | 139.27 ms  | 148.18 ms  | 80.19 MB   |
| **anicrop (OpenCV Zarr)** | 1321.78 ms    | 0.76 FPS           | 1310.74 ms | 1341.76 ms | 667.58 MB  |
| **Pyvips**                | 94.92 ms      | 10.53 FPS          | 93.54 ms   | 96.21 ms   | 84.08 MB   |
| **Pillow**                | 5769.35 ms    | 0.17 FPS           | 5757.28 ms | 5792.52 ms | 303.74 MB  |


## 📈 Gráficos Comparativos

![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)
