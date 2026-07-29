.PHONY: ingest diagnose gates test push-data pull-results validate-kaggle morning-report pull-kaggle

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

# Kaggle results validation (task 2.1/2.3) -- see KAGGLE_PROVISIONING.md.
# COMPARATOR_ZIP and/or CHALLENGER_ZIP are paths to the two notebooks'
# downloaded output archives; either or both may be given.
validate-kaggle:
	python -m engine.backtest.validate_kaggle \
		$(if $(COMPARATOR_ZIP),--comparator-zip $(COMPARATOR_ZIP)) \
		$(if $(CHALLENGER_ZIP),--challenger-zip $(CHALLENGER_ZIP))

# The morning report (task 2.1/2.3): fold-1 predictive-agreement,
# batch health, bounded NUTS coefficient diff, challenger grid summary.
# Run only after validate-kaggle reports clean. Read-only: never
# merges results or evaluates a gate.
morning-report:
	python -m engine.backtest.kaggle_morning_report

# One-command Kaggle retrieval (task 2.1/2.3) -- see KAGGLE_PROVISIONING.md.
# NOTEBOOK_A/NOTEBOOK_B default to this project's two batch notebooks
# (identified via `kaggle kernels list --mine`); override on the
# command line if the slugs ever change, e.g.
#   make pull-kaggle NOTEBOOK_A=onionterror8/some-other-slug
NOTEBOOK_A ?= onionterror8/notebook-a
NOTEBOOK_B ?= onionterror8/notebook-b

pull-kaggle:
	python -m engine.backtest.pull_kaggle_cli --notebook-a $(NOTEBOOK_A) --notebook-b $(NOTEBOOK_B)
	$(MAKE) validate-kaggle COMPARATOR_ZIP=/tmp/kaggle_a/augurics_results.zip CHALLENGER_ZIP=/tmp/kaggle_b/augurics_results.zip
