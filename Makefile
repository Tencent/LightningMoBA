PYTHON ?= python3
ARTIFACTS ?= dist/*

.PHONY: clean_dist create_dist check_dist upload_package test test-cpu test-block8 test-topk

test: test-cpu test-block8 test-topk

test-cpu:
	$(PYTHON) -m pytest -q tests/test_public_interface.py -m "not cuda"

test-block8:
	$(PYTHON) -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0), "test-block8 requires an SM90 GPU"'
	$(PYTHON) -m pytest -q tests/test_public_interface.py tests/test_block8_migration.py tests/test_block8_extension.py tests/test_block8_numerics.py -m cuda

test-topk:
	$(PYTHON) -c 'import torch; assert torch.cuda.is_available(), "test-topk requires a CUDA GPU"'
	$(PYTHON) -m pytest -q tests/test_topk_correctness.py

clean_dist:
	rm -rf -- dist/

create_dist:
	LIGHTNING_MOBA_SKIP_CUDA_BUILD=TRUE $(PYTHON) setup.py sdist

check_dist:
	$(PYTHON) -m twine check $(ARTIFACTS)

upload_package: check_dist
	$(PYTHON) -m twine upload --non-interactive $(ARTIFACTS)
