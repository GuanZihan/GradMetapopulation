# Grad-Metapopulation

Everything needed to run the demos is already included under `Data/`. Clone the repo, install dependencies, then run the shell scripts from the **repository root** (paths in `Configs/` are relative to that directory).

## Setup

Use Python 3.10+ (3.11 recommended). From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists only libraries imported by this codebase. For a specific CUDA build of PyTorch, follow [pytorch.org](https://pytorch.org/get-started/locally/) and install `torch` before or after the other packages.

The training scripts call PyTorch on **GPU device index `0`** (`-d 0`). If you have no GPU, edit the `python src/main.py ...` line in the script you run and replace `-d 0` with `-d cpu`.

## What is included

- **Bogotá (forecasting demo):** Transaction-side inputs are **not** the real financial data (they cannot be shared). The repo includes **synthetic tensors** produced with `torch.randn` (see `src/privatize_bogota_lap.py`). All other demo files are already in `Data/`.
- **United States:** We ship a **forecasting** demo and a separate **nowcasting** demo, both with data in this repo.

No extra downloads or data preparation steps are required for these commands.

## Run the demos

Run **exactly one** of the following from the repository root:

**Bogotá — forecasting (multiple laps):**

```bash
bash scripts/run_forecasting_bogota.sh
```

**United States — forecasting:**

```bash
bash scripts/run_forecasting_usa.sh
```

**United States — nowcasting:**

```bash
bash scripts/run_nowcasting_usa.sh
```

Outputs (figures, logs, checkpoints) are written under project directories such as `Figure-Prediction/`, `Figures/`, and `Results/` according to each config.

## Optional: regenerate synthetic Bogotá transaction tensors

Only if you want to resample the random placeholder (not required to run the demos):

```bash
python src/privatize_bogota_lap.py --moving_window 0
```

Repeat with other `--moving_window` values to match the laps used in `scripts/run_forecasting_bogota.sh`.
