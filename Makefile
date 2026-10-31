.PHONY: install test demo api train lint

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev,api]"

test:
	. .venv/bin/activate && pytest -q

demo:
	. .venv/bin/activate && intervallum-demo --users 20 --kus 16 --tasks 80 --attempts 20

api:
	. .venv/bin/activate && uvicorn intervallum_ml.api:app --reload

train:
	. .venv/bin/activate && intervallum-train --tasks examples/tasks.json --attempts examples/attempts.json --out examples/model.json
