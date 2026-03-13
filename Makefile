.PHONY: test lint format docs

test:
	uv run pytest tests/ --nbmake notebook/view_results_arena.ipynb notebook/view_results_maze.ipynb

lint:
	uv run ruff check
	uv run black placecell --diff

format:
	uv run black placecell
	uv run ruff check --fix

docs:
	uv run sphinx-build -b html docs docs/_build/html
