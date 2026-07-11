# v4 reproduction commands

Run from this directory in PowerShell.

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m lspgmoe.pipeline build-data --config configs\v4_main.yaml
.venv\Scripts\python.exe -m lspgmoe.pipeline run --config configs\v4_main.yaml
.venv\Scripts\python.exe -m lspgmoe.pipeline baselines --config configs\v4_main.yaml
.venv\Scripts\python.exe -m lspgmoe.pipeline smoke --config configs\v4_smoke.yaml
.venv\Scripts\python.exe -m lspgmoe.pipeline ablations --config configs\v4_main.yaml
```

The smoke configuration is an engineering check. It uses one seed, three OOF folds, and eight epochs. The main configuration is the manuscript run: five seeds, five source-group OOF folds, and up to 1000 epochs with validation early stopping. The current manuscript must not use smoke outputs as final numbers. `smoke`, `baselines`, `ablations`, and `run` read the current frozen processed tables; they do not rebuild or erase external sources. Run `build-data` only for an intentional legacy bootstrap, and it refuses to overwrite processed tables that already contain external sources.

Every experiment writes input hashes, software/GPU information, training logs, predictions, metrics, and bootstrap results under `outputs/`. The raw v3.4 CSV remains outside the new processed tables and is never overwritten.
