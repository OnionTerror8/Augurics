.PHONY: ingest diagnose gates test push-data pull-results

test:
	pytest

ingest:
	python -m engine.data.ingest

diagnose:
	python -m engine.backtest.coverage

gates:
	python -m engine.backtest.gates

# Cloud provisioning (task 2.1 comparator batch) -- see PROVISIONING.md.
# REMOTE is user@host, e.g. `make push-data REMOTE=root@1.2.3.4`.
push-data:
	@test -n "$(REMOTE)" || (echo "usage: make push-data REMOTE=user@host" && exit 1)
	ssh $(REMOTE) 'mkdir -p augurics-engine/data/m5'
	rsync -avzP --progress data/m5/augurics.duckdb $(REMOTE):augurics-engine/data/m5/augurics.duckdb

pull-results:
	@test -n "$(REMOTE)" || (echo "usage: make pull-results REMOTE=user@host" && exit 1)
	rsync -avzP --progress $(REMOTE):augurics-engine/data/fold_cache/comparator/results/ data/fold_cache/comparator/results/
