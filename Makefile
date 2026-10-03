# Convenience targets - each maps to one step of RUNBOOK.md
WINNER ?= lora_r16
setup:      ; ./env/setup_vm.sh
download:   ; ./scripts/01_download_base.sh
data:       ; ./scripts/02_prepare_data.sh
baseline:   ; ./scripts/03_baseline_eval.sh
train:      ; ./scripts/04_train_all.sh
eval:       ; ./scripts/05_merge_and_eval.sh
quantize:   ; ./scripts/06_quantize_and_eval.sh models/$(WINNER)-merged $(WINNER)
bench:      ; ./scripts/07_benchmark.sh $(WINNER)
report:     ; ./scripts/08_report.sh
test:       ; .venv-eval/bin/python tests/test_prompt_parity.py
.PHONY: setup download data baseline train eval quantize bench report test
