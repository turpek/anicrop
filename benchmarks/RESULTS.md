# Avaliação de Desempenho do anicrop em Pipelines de Renderização 2D

> **Data de Execução:** 2026-09-06 15:50:44
> **Ambiente:** Python 3.12+, uv virtualenv, Linux
> **Bibliotecas Analisadas:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`

## 📊 Sumário Executivo por Cenário

### Composição 4K (8 Camadas - Interpolação Bilinear Pareada)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 329.35 ms     | 3.04 FPS           | 326.51 ms  | 331.39 ms  | 178.25 MB  |
| **anicrop (OpenCV)** | 255.99 ms     | 3.91 FPS           | 255.39 ms  | 256.38 ms  | 0.00 MB    |
| **Pyvips**           | 257.26 ms     | 3.89 FPS           | 248.04 ms  | 263.61 ms  | 43.96 MB   |
| **Pillow**           | 634.36 ms     | 1.58 FPS           | 632.96 ms  | 634.99 ms  | 0.00 MB    |
| **OpenCV**           | 1454.77 ms    | 0.69 FPS           | 1449.87 ms | 1458.72 ms | 348.03 MB  |


### Edição por Retalhos (25 Patches em Imagem 4K)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|----------------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop (Pyvips)** | 103.16 ms     | 9.69 FPS           | 99.54 ms  | 106.75 ms | 0.02 MB    |
| **anicrop (OpenCV)** | 103.60 ms     | 9.65 FPS           | 103.26 ms | 104.46 ms | 0.00 MB    |
| **OpenCV**           | 95.30 ms      | 10.49 FPS          | 94.83 ms  | 95.87 ms  | 0.00 MB    |
| **Pillow**           | 79.67 ms      | 12.55 FPS          | 79.39 ms  | 79.89 ms  | 0.00 MB    |


### Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|----------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)** | 519.67 ms     | 1.92 FPS           | 502.66 ms  | 542.58 ms  | 0.02 MB    |
| **anicrop (OpenCV)** | 493.43 ms     | 2.03 FPS           | 492.62 ms  | 493.97 ms  | 126.57 MB  |
| **OpenCV**           | 8115.24 ms    | 0.12 FPS           | 8065.52 ms | 8156.40 ms | 1518.78 MB |
| **Pillow**           | 1238.53 ms    | 0.81 FPS           | 1238.40 ms | 1238.70 ms | 251.60 MB  |


### Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)

| Biblioteca         | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|--------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (MMap)** | 32.07 ms      | 31.18 FPS          | 31.80 ms | 32.27 ms | 229.30 MB  |
| **Pyvips**         | 85.12 ms      | 11.75 FPS          | 79.15 ms | 94.92 ms | 0.02 MB    |


### Edição e Composição em Foto 100MP Real (moon_10k.jpg)

| Biblioteca                | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|---------------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Pyvips)**      | 189.79 ms     | 5.27 FPS           | 183.56 ms  | 195.29 ms  | 129.68 MB  |
| **anicrop (OpenCV MMap)** | 855.66 ms     | 1.17 FPS           | 854.84 ms  | 856.09 ms  | 1048.79 MB |
| **Pyvips**                | 93.75 ms      | 10.67 FPS          | 92.40 ms   | 95.27 ms   | 84.08 MB   |
| **Pillow**                | 5929.44 ms    | 0.17 FPS           | 5928.37 ms | 5930.00 ms | 1213.10 MB |


### Cenário F1 - Leitura JPEG 4K

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|----------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (Pyvips)** | 18.89 ms      | 52.94 FPS          | 18.13 ms | 19.77 ms | 0.30 MB    |
| **anicrop (OpenCV)** | 17.91 ms      | 55.84 FPS          | 17.82 ms | 18.05 ms | 0.00 MB    |
| **OpenCV**           | 15.29 ms      | 65.40 FPS          | 15.27 ms | 15.32 ms | 0.00 MB    |
| **Pillow**           | 25.35 ms      | 39.45 FPS          | 25.32 ms | 25.38 ms | 0.00 MB    |
| **Pyvips**           | 18.24 ms      | 54.82 FPS          | 18.03 ms | 18.37 ms | 0.22 MB    |


### Cenário F2 - Leitura PNG 4K

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|----------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (Pyvips)** | 80.52 ms      | 12.42 FPS          | 77.92 ms | 82.32 ms | 0.00 MB    |
| **anicrop (OpenCV)** | 80.52 ms      | 12.42 FPS          | 80.46 ms | 80.58 ms | 0.00 MB    |
| **OpenCV**           | 77.22 ms      | 12.95 FPS          | 77.16 ms | 77.28 ms | 0.00 MB    |
| **Pillow**           | 81.98 ms      | 12.20 FPS          | 81.94 ms | 82.01 ms | 0.00 MB    |
| **Pyvips**           | 79.56 ms      | 12.57 FPS          | 78.91 ms | 80.12 ms | 0.00 MB    |


### Cenário F3 - Leitura WebP 4K

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|----------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (Pyvips)** | 4.00 ms       | 250.05 FPS         | 3.69 ms  | 4.24 ms  | 0.00 MB    |
| **anicrop (OpenCV)** | 24.02 ms      | 41.64 FPS          | 23.91 ms | 24.08 ms | 0.00 MB    |
| **OpenCV**           | 21.41 ms      | 46.71 FPS          | 21.36 ms | 21.46 ms | 0.00 MB    |
| **Pillow**           | 39.30 ms      | 25.44 FPS          | 39.29 ms | 39.34 ms | 0.00 MB    |
| **Pyvips**           | 3.87 ms       | 258.44 FPS         | 3.75 ms  | 3.97 ms  | 0.02 MB    |


### Cenário F4 - Shrink-on-load (4K -> 1080p)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min      | Max      | Pico RAM   |
|----------------------|---------------|--------------------|----------|----------|------------|
| **anicrop (Pyvips)** | 2.17 ms       | 461.18 FPS         | 2.04 ms  | 2.33 ms  | 0.08 MB    |
| **anicrop (OpenCV)** | 19.25 ms      | 51.95 FPS          | 19.19 ms | 19.32 ms | 0.00 MB    |
| **OpenCV**           | 16.55 ms      | 60.41 FPS          | 16.50 ms | 16.61 ms | 0.00 MB    |
| **Pillow**           | 11.88 ms      | 84.16 FPS          | 11.85 ms | 11.90 ms | 0.01 MB    |
| **Pyvips**           | 1.57 ms       | 637.89 FPS         | 1.45 ms  | 1.79 ms  | 0.02 MB    |


### Cenário F5 - Shrink-on-load Gigante (10K -> 1280x1280)

| Biblioteca           | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|----------------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop (Pyvips)** | 2.03 ms       | 493.46 FPS         | 1.87 ms   | 2.11 ms   | 0.08 MB    |
| **OpenCV**           | 415.18 ms     | 2.41 FPS           | 413.87 ms | 415.97 ms | 566.21 MB  |
| **Pillow**           | 211.89 ms     | 4.72 FPS           | 211.79 ms | 212.00 ms | 0.00 MB    |
| **Pyvips**           | 1.57 ms       | 634.99 FPS         | 1.44 ms   | 1.71 ms   | 0.02 MB    |


## 📈 Gráficos Comparativos

![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)
