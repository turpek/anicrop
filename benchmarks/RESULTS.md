# Relatório Oficial de Benchmarks: anicrop vs Concorrentes

> **Data de Execução:** 2026-08-31 04:18:46
> **Ambiente:** Python 3.12+, uv virtualenv, Linux
> **Competidores:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`

## 📊 Sumário Executivo por Cenário

### Composição 4K (8 Camadas - Interpolação Bilinear Pareada)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 312.26 ms     | 3.20 FPS           | 298.44 ms  | 343.47 ms  | 171.25 MB  |
| **anicrop (OpenCV)** | 254.57 ms     | 3.93 FPS           | 250.04 ms  | 267.72 ms  | 0.00 MB    |
| **Pyvips**           | 283.93 ms     | 3.52 FPS           | 278.58 ms  | 288.45 ms  | 78.85 MB   |
| **Pillow**           | 733.07 ms     | 1.36 FPS           | 685.96 ms  | 766.22 ms  | 0.01 MB    |
| **OpenCV**           | 1467.52 ms    | 0.68 FPS           | 1456.20 ms | 1477.73 ms | 316.42 MB  |


### Edição por Retalhos (25 Patches em Imagem 4K)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|----------------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop (Pyvips)** | 124.89 ms     | 8.01 FPS           | 101.71 ms | 253.83 ms | 0.02 MB    |
| **anicrop (OpenCV)** | 110.38 ms     | 9.06 FPS           | 102.79 ms | 127.62 ms | 0.00 MB    |
| **OpenCV**           | 97.34 ms      | 10.27 FPS          | 96.06 ms  | 101.20 ms | 0.00 MB    |
| **Pillow**           | 80.07 ms      | 12.49 FPS          | 79.43 ms  | 84.06 ms  | 0.00 MB    |


### Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 516.55 ms     | 1.94 FPS           | 500.19 ms  | 547.82 ms  | 126.57 MB  |
| **anicrop (OpenCV)** | 573.23 ms     | 1.74 FPS           | 569.32 ms  | 575.47 ms  | 253.14 MB  |
| **OpenCV**           | 8663.75 ms    | 0.12 FPS           | 8631.39 ms | 8709.66 ms | 1645.20 MB |
| **Pillow**           | 1277.42 ms    | 0.78 FPS           | 1271.33 ms | 1288.91 ms | 253.00 MB  |


### Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)

| Biblioteca         | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|--------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (Zarr)** | 79.37 ms      | 12.60 FPS          | 77.78 ms | 81.51 ms | 75.76 MB   |
| **Pyvips**         | 85.64 ms      | 11.68 FPS          | 83.10 ms | 89.21 ms | 0.02 MB    |


### Edição e Composição em Foto 100MP Real (moon_10k.jpg)

| Biblioteca                | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|---------------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)**      | 197.21 ms     | 5.07 FPS           | 188.22 ms  | 211.76 ms  | 129.13 MB  |
| **anicrop (OpenCV Zarr)** | 2397.71 ms    | 0.42 FPS           | 2378.17 ms | 2426.12 ms | 619.51 MB  |
| **Pyvips**                | 99.04 ms      | 10.10 FPS          | 93.89 ms   | 106.51 ms  | 84.08 MB   |
| **Pillow**                | 6409.69 ms    | 0.16 FPS           | 6248.83 ms | 6618.34 ms | 1834.32 MB |


## 📈 Gráficos Comparativos

![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)
