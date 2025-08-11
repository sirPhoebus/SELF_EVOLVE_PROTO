# Hierarchical Reasoning Model (HRM)

A self-evolving deep neural network that performs hierarchical reasoning with Adaptive Computation Time (ACT) and can autonomously mine recent AI research (arXiv), extract concepts, and propose architecture updates.

This repository includes:
- A modular Hierarchical Reasoning Model implemented in PyTorch with transformer blocks, rotary embeddings (RoPE), RMSNorm, SwiGLU MLPs, and ACT-style halting.
- A self-evolution pipeline that fetches recent papers, extracts reasoning-related concepts, maps them to configuration actions, and proposes config updates that can be applied automatically.
- A CLI test harness to run a minimal forward/backward pass and to trigger self-evolution, apply proposals, and persist artifacts.


## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Training Smoke Test](#training-smoke-test)
  - [Self-Evolution (Dry-Run)](#self-evolution-dry-run)
  - [Apply Evolution + Save Report + Export Config](#apply-evolution--save-report--export-config)
  - [Train Using Exported Config](#train-using-exported-config)
- [Model Architecture](#model-architecture)
- [Self-Evolution Pipeline](#self-evolution-pipeline)
- [CLI Usage](#cli-usage)
- [Configuration Notes](#configuration-notes)
- [Troubleshooting](#troubleshooting)
- [Extensibility](#extensibility)
- [Development Notes](#development-notes)


## Features
- __Hierarchical Reasoning__: High-level and low-level recurrent reasoning modules with multi-step ACT halting.
- __Modern Transformer Blocks__: Rotary positional embeddings, attention with FlashAttention (if installed) or PyTorch SDPA fallback, RMSNorm, SwiGLU.
- __Self-Evolution__: Fetches recent arXiv papers, extracts concepts, plans config updates, and can apply changes to instantiate an evolved model.
- __Reproducible CLI__: Minimal runnable script to validate forward/backward and to invoke self-evolution end-to-end. Offers exporting/using configs, controlling ACT steps, and saving reports.


## Repository Structure
- `hrm.py` — Top-level model `HierarchicalReasoningModel_ACTV1`, ACT loop, config plumbing, and `self_evolve()` integration.
- `models/` — Core model components
  - `models/layers.py` — Linear/embedding layers, attention, RoPE, RMSNorm, SwiGLU. Includes a PyTorch SDPA fallback when FlashAttention is unavailable.
  - `models/losses.py` — Loss head `ACTLossHead` to run the ACT loop, compute language modeling and halting losses, and aggregate metrics.
  - `models/common.py` — Common helpers (e.g., truncated normal init).
  - `models/evolve/` — Self-evolution pipeline modules:
    - `literature.py` — `ArxivMiner` to fetch recent papers (via `arxiv` package or HTTP fallback using `requests`).
    - `concepts.py` — Extract concepts and map them into architecture update actions.
    - `introspect.py` — Extract current model config from a live model instance.
    - `planner.py` — Plan a modest, safe set of config updates with rationale.
    - `apply.py` — Helpers to instantiate a new model from a modified config (if needed by callers).
    - `pipeline.py` — `SelfEvolvePipeline` orchestrating the full process and returning a structured report.
- `scripts/test_self_evolve.py` — CLI test harness to:
  - run a minimal forward/backward pass,
  - invoke self-evolution in dry-run or applied mode,
  - save reports and export evolved configs,
  - train using a provided config, and
  - control ACT steps or iterate until halt.
- `requirements.txt` — Python dependencies.


## Installation
Prerequisites:
- Python 3.9+
- Windows, Linux, or macOS
- Optional GPU with CUDA for acceleration

Set up a virtual environment and install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```
Notes:
- `arxiv` is conditionally enabled on Windows for Python >= 3.9. If not available, the HTTP fallback is used automatically via `requests`.
- FlashAttention is optional. If not installed, attention falls back to PyTorch `scaled_dot_product_attention` (see `models/layers.py`).


## Quickstart
All commands run from the repo root `HRM/` with the virtual environment activated.

### Training Smoke Test
Runs a single or few ACT steps to validate forward/backward:
```powershell
python -m scripts.test_self_evolve --no-evolve --act-steps 2
```
Add `--until-halt` to iterate ACT until all sequences halt:
```powershell
python -m scripts.test_self_evolve --no-evolve --until-halt
```

### Self-Evolution (Dry-Run)
Fetch recent papers, score concepts, plan updates, but do NOT change the model:
```powershell
python -m scripts.test_self_evolve --no-train
```

### Apply Evolution + Save Report + Export Config
Apply the proposed changes (dry_run=False), save a full JSON report, and export the resulting `new_config` for later reuse:
```powershell
python -m scripts.test_self_evolve --no-train --apply-evolve \
  --save-report artifacts\evolve_report.json \
  --export-config artifacts\hrm_evolved_config.json
```
Outputs:
- `artifacts\evolve_report.json` — Papers, concepts, actions, rationale, old_config, and when applied, `new_config`.
- Printed config diff and a validation that the evolved model instantiates successfully.

### Train Using Exported Config
Use a saved config JSON for training, skip arXiv:
```powershell
python -m scripts.test_self_evolve --use-config artifacts\hrm_evolved_config.json --no-evolve --act-steps 3
```
Or until all sequences halt:
```powershell
python -m scripts.test_self_evolve --use-config artifacts\hrm_evolved_config.json --no-evolve --until-halt
```


## Model Architecture
- `HierarchicalReasoningModel_ACTV1` in `hrm.py` implements a hierarchical, recurrent reasoning process.
- ACT-style halting allows variable computation per example, controlled by `halt_max_steps` and exploration probability.
- Transformer components are in `models/layers.py`:
  - Attention uses FlashAttention if available, otherwise PyTorch SDPA fallback.
  - Rotary positional embeddings (`apply_rotary_pos_emb()`), RMSNorm (`RMSNorm`), and SwiGLU MLPs.
- Loss and ACT control via `models/losses.py` (`ACTLossHead`).

Key config fields (non-exhaustive):
- Sequence/model sizes: `seq_len`, `hidden_size`, `expansion`, `num_heads`, `vocab_size`.
- Hierarchical structure: `H_layers`, `L_layers`, `H_cycles`, `L_cycles`.
- ACT parameters: `halt_max_steps`, `halt_exploration_prob`.
- Position encodings: `pos_encodings` (e.g., `"rope"`), `rope_theta`.
- Training dtype: `forward_dtype` (set to `"float32"` for CPU-friendly tests).


## Self-Evolution Pipeline
Files in `models/evolve/`:
- `literature.py` — `ArxivMiner.fetch()` gets recent papers given `query_terms`, `days_back`, and `max_results`. Uses the `arxiv` package or an HTTP fallback via `requests`.
- `concepts.py` — `score_concepts()` yields concept objects with scores; `map_concepts_to_actions()` converts them into config actions (e.g., `{"param": "H_layers", "delta": +1}`).
- `introspect.py` — `model_config_dict()` returns the active model’s configuration.
- `planner.py` — `plan_updates()` prioritizes a modest, safe set of parameter updates, returning `Proposal(changes, rationale)`.
- `pipeline.py` — `SelfEvolvePipeline.run()` orchestrates the above and returns a `report` dictionary containing:
  - `papers`, `concepts` (as dictionaries), `actions`, `rationale`, and `old_config`.
  - When `dry_run=False`, also `new_config` with changes applied.

The model integrates this via `HierarchicalReasoningModel_ACTV1.self_evolve()` in `hrm.py`.


## CLI Usage
The script `scripts/test_self_evolve.py` supports:

Training:
- `--no-evolve` — Skip arXiv step.
- `--act-steps N` — Number of ACT iterations (default: 1).
- `--until-halt` — Keep iterating until all sequences halt.
- `--use-config PATH` — Load a JSON config for training/evolution instead of the built-in test config.

Self-evolution:
- `--no-train` — Skip training.
- `--query-terms ...` — Custom search terms (default: reasoning/hierarchical/transformer/RL/ACT).
- `--days-back N` — Recency window for arXiv.
- `--max-results N` — Maximum papers to fetch.
- `--apply-evolve` — Apply the planned changes (dry_run=False) and print a config diff.
- `--save-report PATH` — Save the full report JSON.
- `--export-config PATH` — Write `new_config` to JSON (requires `--apply-evolve`).

Code evolution (LLM-driven):
- `--code-evolve` — Run LLM-driven code evolution to produce a proposal JSON.
- `--use-report PATH` — Use an existing evolve report JSON as input to code evolution.
- `--save-proposal PATH` — Save the code-evolution proposal (JSON if parsed; otherwise raw text).
- `--llm-provider NAME` — LLM provider (e.g., `ollama`, `openai`, `lmstudio` [OpenAI-compatible]).
- `--llm-model NAME` — Model name for the provider (e.g., `llama3.1:8b`).
- `--llm-host URL` — Base URL for the provider (Ollama defaults to `http://localhost:11434`).
- `--llm-timeout SEC` — Timeout for LLM requests.
- `--llm-temperature FLOAT` — Sampling temperature.
- `--llm-max-tokens N` — Max tokens to generate.
- `--summary-bytes N` — Max bytes per file to include in summarization.
- `--summary-basic` — Summarize only key files.

Example:
```powershell
python -m scripts.test_self_evolve --code-evolve \
  --use-report artifacts\evolve_report.json \
  --save-proposal artifacts\code_proposal.json \
  --llm-provider lmstudio --llm-host http://localhost:1234 --llm-model openai/gpt-oss-20b
```

Basic sizes:
- `--batch-size`, `--seq-len`, `--vocab-size`, `--num-puzzle-ids` — A minimal set to define toy data.


## Configuration Notes
- The test harness builds a small config by default (see `build_small_config()` in `scripts/test_self_evolve.py`).
- When applying evolution (`--apply-evolve`), the script prints a diff (`old -> new`) and validates that the evolved model can be instantiated.
- You can persist `new_config` with `--export-config` and reuse it via `--use-config` for training or further evolution runs.


## Troubleshooting
- __CPU/GPU device mismatch__: Fixed internally by allocating carry tensors on the same device as inputs (`hrm.py`). If you pass your own tensors, ensure consistent devices.
- __FlashAttention not installed__: The code automatically falls back to PyTorch `scaled_dot_product_attention` in `models/layers.py`. CPU-only runs are supported.
- __arXiv issues on Windows__: `requirements.txt` conditionally enables `arxiv` for Python >= 3.9. If unavailable, the HTTP fallback will be used transparently via `requests`.
- __Network failures / rate-limits__: Re-run later or reduce `--max-results`. You may configure proxies via environment variables if needed.
- __ACT metrics show count=0__: With `halt_max_steps > 1`, a single ACT call may not halt sequences. Use `--act-steps N` > 1 or `--until-halt`.


## Extensibility
- __Concept extraction__: Replace or expand `score_concepts()` to use more advanced NLP or LLM summarization.
- __Action mapping__: Enhance `map_concepts_to_actions()` to cover more hyperparameters or structural changes.
- __Planning__: Adjust `plan_updates()` to change the number/strength of changes, add constraints, or include multi-objective criteria.
- __Application__: Integrate weight transfer or state mapping when structure changes are significant.


## Development Notes
- Code style favors modularity and explicit configuration plumbing.
- Initialization utilities (`models/common.py`) use truncated normal; attention/MLP blocks prefer numerically stable operations.
- The CLI implements reproducible evolution-testing and training smoke tests.

## Environment-based Configuration (.env on Windows)
 The continuous runner `scripts/continuous_runner.py` auto-loads a `.env` file from the project root using `python-dotenv` (or a built-in fallback). This lets you configure all run parameters without setting PowerShell variables.
 
 Steps:
 - Copy `.env.example` to `.env` in the repo root.
 - Edit values as needed. All keys prefixed with `HRM_` map to `RUN_CONFIG` keys in `scripts/continuous_runner.py`.
 - Run the runner normally (no special PowerShell `set` needed):
 
 ```powershell
 python -m scripts.continuous_runner
 ```
 
 Example `.env` for a short smoke test:
 
 ```env
 HRM_MAX_RUNTIME_SEC=900
 HRM_MAX_CYCLES=3
 HRM_CYCLE_INTERVAL_SEC=60
 HRM_TRAIN_STEPS_PER_CYCLE=2
 HRM_APPLY_EVOLVE=true
 HRM_LLM_PROVIDER=lmstudio
 HRM_LLM_HOST=http://localhost:1234
 HRM_LLM_MODEL=openai/gpt-oss-20b
 LLM_MAX_RETRIES=3
 ARXIV_MAX_RETRIES=3
 ```
 
 Example `.env` for a 24h run:
 
 ```env
 HRM_MAX_RUNTIME_SEC=86400
 HRM_CYCLE_INTERVAL_SEC=300
 HRM_TRAIN_STEPS_PER_CYCLE=5
 HRM_APPLY_EVOLVE=true
 HRM_LLM_PROVIDER=lmstudio
 HRM_LLM_HOST=http://localhost:1234
 HRM_LLM_MODEL=openai/gpt-oss-20b
 LLM_MAX_RETRIES=3
 ARXIV_MAX_RETRIES=3
 ```
 
 Optional:
 - `HRM_RUN_CONFIG_JSON=C:\\path\\to\\my_run_config.json` to overlay an additional JSON config.
 - Install `matplotlib` to emit `plots.png` automatically at run end.
 
 Additional keys (environment overlay):
 - Timeboxing/schedule: `HRM_MAX_RUNTIME_SEC`, `HRM_MAX_CYCLES`, `HRM_CYCLE_INTERVAL_SEC`, `HRM_CHECKPOINT_FILENAME`.
 - Training per cycle: `HRM_TRAIN_STEPS_PER_CYCLE`, `HRM_BATCH_SIZE`, `HRM_SEQ_LEN`, `HRM_VOCAB_SIZE`, `HRM_NUM_PUZZLE_IDS`, `HRM_UNTIL_HALT`.
 - Self-evolution: `HRM_APPLY_EVOLVE`, `HRM_QUERY_TERMS`, `HRM_QUERY_ROTATE_TERMS`, `HRM_DAYS_BACK`, `HRM_MAX_RESULTS`, `HRM_GLOBAL_SEEN_PAPERS_PATH`.
 - LLM/code-evolution: `HRM_LLM_PROVIDER`, `HRM_LLM_MODEL`, `HRM_LLM_HOST`, `HRM_LLM_TIMEOUT`, `HRM_LLM_TEMPERATURE`, `HRM_LLM_MAX_TOKENS`.
 - Summarization: `HRM_SUMMARY_BASIC`, `HRM_SUMMARY_BYTES`.
 - Visualization: `HRM_GENERATE_PLOTS`, `HRM_PLOTS_FILENAME`.
 - Resource thresholds: `HRM_MIN_DISK_FREE_GB`, `HRM_MAX_MEMORY_PCT`.
 - Autopatch: `HRM_AUTO_APPLY_PATCHES`, `HRM_AUTOPATCH_MAX_FILES`, `HRM_AUTOPATCH_STRICT`, `HRM_SMOKETEST_CMD`.
 
 Retry/backoff knobs:
 - LLM: `LLM_MAX_RETRIES`, `LLM_BACKOFF_BASE_SEC`, `LLM_BACKOFF_JITTER_SEC`.
 - arXiv: `ARXIV_MAX_RETRIES`, `ARXIV_BACKOFF_BASE_SEC`, `ARXIV_BACKOFF_JITTER_SEC`.
 
 Provider secrets/hosts:
 - `OPENAI_API_KEY`, `OLLAMA_HOST`, `LMSTUDIO_HOST`, `LMSTUDIO_API_KEY`.
 
 Synthetic data generation (optional):
  - `HRM_SYNTHETIC_DATA_ENABLE` — Enable LLM-generated synthetic training batches.
  - `HRM_SYNTHETIC_DATA_PROVIDER`, `HRM_SYNTHETIC_DATA_MODEL`, `HRM_SYNTHETIC_DATA_HOST`.
  - `HRM_SYNTHETIC_DATA_TIMEOUT`, `HRM_SYNTHETIC_DATA_TEMPERATURE`, `HRM_SYNTHETIC_DATA_MAX_TOKENS`.
  - `HRM_SYNTHETIC_DATA_SYSTEM`, `HRM_SYNTHETIC_DATA_PROMPT` — Override system/prompt templates.
  Notes: When synthetic keys are not set, values fall back to `HRM_LLM_*` then `LLM_*` defaults.
  
  Note: The runner sets a per-run `HRM_SEEN_PAPERS_PATH` for deduplication; set `HRM_GLOBAL_SEEN_PAPERS_PATH` to dedup across runs.
 
 If you have any questions or want additional features, feel free to open an issue or request enhancements.
