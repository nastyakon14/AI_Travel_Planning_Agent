.PHONY: test up down

test:
	python -m pytest tests/ -v

up:
	docker compose up --build

down:
	docker compose down
