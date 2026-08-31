# Relatório Oficial de Benchmarks: anicrop vs Concorrentes

> **Data de Execução:** 2026-08-31 03:28:51
> **Ambiente:** Python 3.12+, uv virtualenv, Linux
> **Competidores:** `anicrop`, `Pillow`, `OpenCV` (NumPy), `Pyvips`

## 📊 Sumário Executivo por Cenário

### Composição 4K (8 Camadas - Interpolação Bilinear Pareada)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 272.73 ms     | 3.67 FPS           | 267.19 ms  | 284.50 ms  | 176.55 MB  |
| **Pyvips**   | 466.64 ms     | 2.14 FPS           | 458.60 ms  | 473.84 ms  | 153.09 MB  |
| **Pillow**   | 762.79 ms     | 1.31 FPS           | 748.96 ms  | 780.78 ms  | 0.01 MB    |
| **OpenCV**   | 1481.36 ms    | 0.68 FPS           | 1475.58 ms | 1490.44 ms | 442.86 MB  |


### Edição por Retalhos (25 Patches em Imagem 4K)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min       | Max       | Pico RAM   |
|--------------|---------------|--------------------|-----------|-----------|------------|
| **anicrop**  | 114.29 ms     | 8.75 FPS           | 103.31 ms | 163.06 ms | 0.02 MB    |
| **OpenCV**   | 97.25 ms      | 10.28 FPS          | 96.12 ms  | 102.62 ms | 0.00 MB    |
| **Pillow**   | 80.49 ms      | 12.42 FPS          | 79.62 ms  | 82.99 ms  | 0.00 MB    |


### Latência de Viewport Preview em Cena 8K (800x600 sob Zoom 3x)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 523.27 ms     | 1.91 FPS           | 507.43 ms  | 550.85 ms  | 126.59 MB  |
| **OpenCV**   | 8634.46 ms    | 0.12 FPS           | 8600.43 ms | 8688.13 ms | 1645.30 MB |
| **Pillow**   | 1269.15 ms    | 0.79 FPS           | 1261.32 ms | 1279.88 ms | 316.99 MB  |


### Processamento Gigapixel 100MP (10000x10000 c/ Rotação e Patch 4K)

| Biblioteca         | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------------|---------------|--------------------|------------|------------|------------|
| **anicrop (Zarr)** | 2424.43 ms    | 0.41 FPS           | 2417.73 ms | 2436.35 ms | 476.85 MB  |
| **Pyvips**         | 194.00 ms     | 5.15 FPS           | 190.13 ms  | 200.16 ms  | 61.05 MB   |


### Edição e Composição em Foto 100MP Real (moon_10k.jpg)

| Biblioteca   | Tempo Médio   | Throughput (FPS)   | Min        | Max        | Pico RAM   |
|--------------|---------------|--------------------|------------|------------|------------|
| **anicrop**  | 3888.80 ms    | 0.26 FPS           | 3862.69 ms | 3904.10 ms | 667.58 MB  |
| **Pyvips**   | 258.88 ms     | 3.86 FPS           | 246.12 ms  | 267.16 ms  | 210.64 MB  |
| **Pillow**   | 5781.17 ms    | 0.17 FPS           | 5772.14 ms | 5795.73 ms | 287.75 MB  |


## 📈 Gráficos Comparativos

![Gráfico Comparativo de Benchmarks](benchmark_comparison.png)
