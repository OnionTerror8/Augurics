.PHONY: ingest diagnose gates test

test:
	pytest

ingest:
	python -m engine.data.ingest

diagnose:
	python -m engine.backtest.coverage

gates:
	python -m engine.backtest.gates
