# Avaliação de Desempenho do anicrop em Pipelines de Renderização 2D

> **Data de Execução:** 2026-09-05 03:23:53
> **Ambiente:** Python 3.12+, uv virtualenv, Linux
> **Bibliotecas Analisadas:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`

## 📊 Sumário Executivo por Cenário

### Composição 4K (8 Camadas - Interpolação Bilinear Pareada)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 343.37 ms     | 2.91 FPS           | 325.42 ms  | 392.97 ms  | 182.90 MB  |
| **anicrop (OpenCV)** | 267.07 ms     | 3.74 FPS           | 260.34 ms  | 275.06 ms  | 0.00 MB    |
| **Pyvips**           | 257.18 ms     | 3.89 FPS           | 252.42 ms  | 267.29 ms  | 60.12 MB   |
| **Pillow**           | 660.32 ms     | 1.51 FPS           | 639.48 ms  | 699.54 ms  | 0.01 MB    |
| **OpenCV**           | 1483.19 ms    | 0.67 FPS           | 1476.53 ms | 1489.49 ms | 355.85 MB  |


### Edição por Retalhos (25 Patches em Imagem 4K)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|----------------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop (Pyvips)** | 113.25 ms     | 8.83 FPS           | 103.40 ms | 153.25 ms | 0.02 MB    |
| **anicrop (OpenCV)** | 115.91 ms     | 8.63 FPS           | 103.00 ms | 150.07 ms | 0.00 MB    |
| **OpenCV**           | 96.38 ms      | 10.38 FPS          | 95.30 ms  | 99.10 ms  | 0.00 MB    |
| **Pillow**           | 102.89 ms     | 9.72 FPS           | 102.18 ms | 105.79 ms | 0.00 MB    |


### Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 507.98 ms     | 1.97 FPS           | 502.59 ms  | 515.03 ms  | 126.58 MB  |
| **anicrop (OpenCV)** | 570.51 ms     | 1.75 FPS           | 565.99 ms  | 577.45 ms  | 252.24 MB  |
| **OpenCV**           | 8644.63 ms    | 0.12 FPS           | 8616.68 ms | 8665.15 ms | 1645.34 MB |
| **Pillow**           | 1266.98 ms    | 0.79 FPS           | 1250.96 ms | 1287.28 ms | 267.59 MB  |


### Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)

| Biblioteca         | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|--------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (MMap)** | 34.26 ms      | 29.19 FPS          | 33.36 ms | 35.70 ms | 230.98 MB  |
| **Pyvips**         | 87.97 ms      | 11.37 FPS          | 82.83 ms | 97.38 ms | 0.02 MB    |


### Edição e Composição em Foto 100MP Real (moon_10k.jpg)

| Biblioteca                | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|---------------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)**      | 146.66 ms     | 6.82 FPS           | 140.25 ms  | 156.15 ms  | 83.94 MB   |
| **anicrop (OpenCV MMap)** | 885.34 ms     | 1.13 FPS           | 881.39 ms  | 891.38 ms  | 1048.05 MB |
| **Pyvips**                | 99.35 ms      | 10.07 FPS          | 92.19 ms   | 109.54 ms  | 84.08 MB   |
| **Pillow**                | 6158.24 ms    | 0.16 FPS           | 6153.82 ms | 6161.71 ms | 1820.51 MB |


## 📈 Gráficos Comparativos

![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)
