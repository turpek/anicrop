# Relatório Oficial de Benchmarks: anicrop vs Concorrentes

> **Data de Execução:** 2026-08-31 03:04:49
> **Ambiente:** Python 3.12+, uv virtualenv, Linux
> **Competidores:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`

## 📊 Sumário Executivo por Cenário

### Composição 4K (8 Camadas - Interpolação Bilinear Pareada)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 304.44 ms     | 3.28 FPS           | 301.54 ms  | 305.98 ms  | 171.15 MB  |
| **Pyvips**   | 465.70 ms     | 2.15 FPS           | 455.77 ms  | 471.52 ms  | 172.95 MB  |
| **Pillow**   | 777.79 ms     | 1.29 FPS           | 761.50 ms  | 800.96 ms  | 70.12 MB   |
| **OpenCV**   | 1472.01 ms    | 0.68 FPS           | 1460.06 ms | 1493.84 ms | 355.85 MB  |


### Edição por Retalhos (25 Patches em Imagem 4K)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|--------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop**  | 120.30 ms     | 8.31 FPS           | 100.87 ms | 200.81 ms | 0.02 MB    |
| **OpenCV**   | 96.67 ms      | 10.34 FPS          | 95.71 ms  | 99.96 ms  | 0.00 MB    |
| **Pillow**   | 81.14 ms      | 12.32 FPS          | 79.63 ms  | 86.02 ms  | 0.00 MB    |


### Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 514.05 ms     | 1.95 FPS           | 505.32 ms  | 529.01 ms  | 126.57 MB  |
| **OpenCV**   | 8633.28 ms    | 0.12 FPS           | 8602.89 ms | 8676.46 ms | 1645.29 MB |
| **Pillow**   | 1222.72 ms    | 0.82 FPS           | 1208.46 ms | 1244.73 ms | 324.61 MB  |


### Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)

| Biblioteca         | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Zarr)** | 2418.29 ms    | 0.41 FPS           | 2413.13 ms | 2427.43 ms | 476.85 MB  |
| **Pyvips**         | 91.59 ms      | 10.92 FPS          | 85.86 ms   | 102.98 ms  | 0.02 MB    |


### Edição e Composição em Foto 100MP Real (moon_10k.jpg)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 5144.41 ms    | 0.19 FPS           | 5129.72 ms | 5153.44 ms | 621.55 MB  |
| **Pyvips**   | 267.67 ms     | 3.74 FPS           | 264.69 ms  | 271.65 ms  | 210.64 MB  |
| **Pillow**   | 6195.63 ms    | 0.16 FPS           | 6173.87 ms | 6236.68 ms | 1866.30 MB |


## 📈 Gráficos Comparativos

![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)
