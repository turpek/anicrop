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

# Formata o código com autopep8 e corrige linter com ruff
format:
	uv run ruff check . --fix
	uv run autopep8 --in-place --recursive --max-line-length 89 --ignore E501,E402,W503,W504 src/ tests/ assets/

# ==============================================================================
# Fluxo de Sincronização Git Multi-PC (dev <-> main com sync-point)
# ==============================================================================

# Envia todas as alterações da branch dev para o GitHub
push-dev:
	@echo "==> Enviando branch dev para o GitHub..."
	git push origin dev

# Atualiza a branch dev a partir do GitHub (para rodar no outro PC)
pull-dev:
	@echo "==> Atualizando branch dev a partir do GitHub..."
	git pull origin dev

# Sincroniza commits de produção da 'dev' para a 'main' individualmente (histórico limpo e semântico)
sync-main:
	@echo "==> Sincronizando commits de produção com a branch main..."
	@LAST_POINT=$$(git log -50 --format="%b" main 2>/dev/null | grep -oE '(sync-point: [a-f0-9]+|\(cherry picked from commit [a-f0-9]+\))' | head -1 | grep -oE '[a-f0-9]{7,40}'); \
	if [ -n "$$LAST_POINT" ]; then \
		RANGE="$$LAST_POINT..dev"; \
	else \
		RANGE="main..dev"; \
	fi; \
	COMMITS=$$(git log $$RANGE --oneline --reverse -- src/ tests/ README.md pyproject.toml setup.py Makefile uv.lock .gitignore .python-version assets/ | cut -d' ' -f1); \
	if [ -z "$$COMMITS" ]; then \
		echo "Nenhuma alteração de produção para sincronizar."; \
	else \
		echo "Commits de produção a sincronizar:"; \
		git log $$RANGE --oneline --reverse -- src/ tests/ README.md pyproject.toml setup.py Makefile uv.lock .gitignore .python-version assets/; \
		git checkout main || exit 1; \
		SUCCESS=1; \
		for C in $$COMMITS; do \
			echo "--> Aplicando $$C na main..."; \
			NON_PROD=$$(git diff-tree --no-commit-id --name-only -r $$C | grep -vE '^(src/|tests/|README\.md|pyproject\.toml|setup\.py|Makefile|uv\.lock|\.gitignore|\.python-version|assets/)' || true); \
			if [ -z "$$NON_PROD" ]; then \
				git cherry-pick -x $$C || { echo "Falha ao aplicar $$C"; git cherry-pick --abort; SUCCESS=0; break; }; \
			else \
				git cherry-pick -n $$C || { echo "Falha no cherry-pick de $$C"; git cherry-pick --abort; SUCCESS=0; break; }; \
				git rm -rf --ignore-unmatch $$NON_PROD >/dev/null 2>&1 || true; \
				git commit -C $$C --no-edit >/dev/null 2>&1 || true; \
				git commit --amend -m "$$(git log -1 --format=%B $$C)" -m "(cherry picked from commit $$C)" >/dev/null 2>&1 || true; \
			fi; \
		done; \
		git checkout dev; \
		if [ $$SUCCESS -eq 1 ]; then \
			echo "==> Sincronização concluída com sucesso! Retornado para a branch dev."; \
		else \
			echo "==> Erro durante a sincronização. Verifique o status da main."; exit 1; \
		fi; \
	fi





# Envia a branch main limpa para o GitHub
push-main:
	@echo "==> Enviando branch main para o GitHub..."
	git push origin main

