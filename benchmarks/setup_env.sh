#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================================="
echo "Configurando ambiente de Benchmark para anicrop via uv..."
echo "========================================================="

# 1. Cria o ambiente virtual se não existir
if [ ! -d "$ROOT_DIR/.venv_bench" ]; then
    echo "==> Criando ambiente virtual em $ROOT_DIR/.venv_bench..."
    uv venv "$ROOT_DIR/.venv_bench"
fi

# 2. Instala anicrop em modo editável e os competidores
echo "==> Instalando dependências de benchmark e anicrop..."
uv pip install -e "$ROOT_DIR" --python "$ROOT_DIR/.venv_bench/bin/python"
uv pip install pillow opencv-python pyvips scikit-image psutil tabulate matplotlib rich --python "$ROOT_DIR/.venv_bench/bin/python"

# 3. Compila as extensões Cython do anicrop
echo "==> Compilando extensões nativas C/Cython..."
"$ROOT_DIR/.venv_bench/bin/python" "$ROOT_DIR/setup.py" build_ext --inplace

echo "========================================================="
echo "✅ Ambiente de benchmark configurado com sucesso!"
echo "Para executar os benchmarks:"
echo "  .venv_bench/bin/python benchmarks/run_all.py"
echo "========================================================="
