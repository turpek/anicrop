# Makefile para o projeto anicrop

# Extrai os argumentos adicionais passados após o alvo principal (ex: make test -v -k foo)
RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(RUN_ARGS):;@:)

# Declara que os alvos não são arquivos.
.PHONY: install build build-ext clean-ext test test_speed test-cov mypy format lint push-dev pull-dev sync-main push-main

# Instala o pacote em modo editável e as dependências de desenvolvimento.
install:
	uv sync
	uv pip install -e .
	$(MAKE) build-ext

# Compila as extensões C/Cython diretamente in-place.
build: build-ext

build-ext:
	uv run python setup.py build_ext --inplace

# Limpa binários compilados C/Cython e diretórios de build temporários.
clean-ext:
	rm -rf build src/anicrop/native/*.so src/anicrop/native/blend.c

# Roda a suíte de testes com o pytest.
test:
	uv run pytest $(RUN_ARGS)

# Roda a suíte de testes com o pytest excluindo os mais lentos.
test_speed:
	uv run ruff format tests/
	uv run pytest -m "not slow" $(RUN_ARGS)

# Roda os testes e gera um relatório de cobertura HTML na pasta 'htmlcov/'.
test-cov:
	uv run pytest --cov=anicrop --cov-report=html $(RUN_ARGS)

# Roda o checador de tipos Mypy no código-fonte.
mypy:
	uv run mypy src $(RUN_ARGS)

# Roda a verificação de formatação e linter com ruff.
lint:
	uv run ruff check .

# Formata o código com ruff format.
format:
	uv run ruff format .

# ==============================================================================
# Fluxo de Sincronização Git Multi-PC (dev <-> main)
# ==============================================================================

# Envia todas as alterações da branch dev para o GitHub
push-dev:
	@echo "==> Enviando branch dev para o GitHub..."
	git push origin dev

# Atualiza a branch dev a partir do GitHub (para rodar no outro PC)
pull-dev:
	@echo "==> Atualizando branch dev a partir do GitHub..."
	git pull origin dev

# Sincroniza apenas o código de produção da branch 'dev' para a 'main' (mantendo a main limpa)
sync-main:
	@echo "==> Sincronizando código de produção com a branch main..."
	@git checkout main
	@git checkout dev -- src/ tests/ README.md assets/ pyproject.toml setup.py Makefile
	@git commit -m "release: sincroniza codigo de producao da dev" || echo "Nenhuma nova alteracao para commitar na main"
	@git checkout dev
	@echo "==> Sincronização concluída! Retornado para a branch dev."

# Envia a branch main limpa para o GitHub
push-main:
	@echo "==> Enviando branch main para o GitHub..."
	git push origin main
