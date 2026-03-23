#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Differentially private input perturbation for the 'hospitalizedIncrease' column
using the Laplace mechanism with *data-range* sensitivity (Δ = max - min),
per your specification. We also report the accumulated ε and δ across all
timestamps released (basic composition).

IMPORTANT PRIVACY NOTE:
- Classical Laplace DP uses a *data-independent* global sensitivity (e.g., Δ=1
  for per-person counts under event-level DP). Here you requested Δ = max - min
  computed *from the dataset itself*. This choice can leak some information
  (since Δ depends on the data). The code implements it anyway, with clear
  comments, so you can decide if it fits your threat model.
"""

import os
import numpy as np
import pandas as pd

# ==== CONFIGURATION ====
INPUT_CSV = "Data/US/online/2021-01-09_0_moving_202101.csv"  # your dataset path
OUTPUT_CSV = os.path.splitext(INPUT_CSV)[0] + "_dp2.csv"

COLUMN = "inIcuCurrently"  # column to privatize
EPSILON = 0.1                    # per-timestamp ε
DELTA = 1e-3                     # per-timestamp δ (not used by Laplace noise itself)
SEED = 42                        # set RNG seed for reproducibility; remove to randomize

# Add as a new column (recommended so you keep the original column as-is)
ADD_AS_NEW_COLUMN = False
NEW_COLUMN_NAME = COLUMN + "_dp"

# Optional post-processing: clip negatives to 0 for count-like data
CLIP_NONNEGATIVE = True

# Optional: round to integers after adding noise (post-processing preserves DP)
ROUND_TO_INT = False

# ==== LOAD DATA ====
df = pd.read_csv(INPUT_CSV)

if COLUMN not in df.columns:
    raise KeyError(f"Column '{COLUMN}' not found in CSV. Columns are: {list(df.columns)}")

# Coerce to numeric (unexpected strings -> NaN) and record which rows are valid
col = pd.to_numeric(df[COLUMN], errors="coerce")
valid_mask = ~col.isna()
valid_values = col[valid_mask].astype(float).to_numpy()

if valid_values.size == 0:
    raise ValueError(f"No valid numeric entries found in column '{COLUMN}'.")

# ==== DATA-RANGE SENSITIVITY (as requested) ====
# Δ = max - min computed on the (non-NaN) column.
# WARNING: This is data-dependent; prefer a known global bound if available.
data_min = float(np.min(valid_values))
data_max = float(np.max(valid_values))
SENSITIVITY = data_max - data_min

# If all values are identical, Δ = 0; no noise is needed to match this definition.
# Add a tiny epsilon to avoid division-by-zero in degenerate cases.
if SENSITIVITY == 0:
    SENSITIVITY = 0.0

# Scale parameter for Laplace: b = Δ / ε
scale = (SENSITIVITY / EPSILON) if EPSILON > 0 else np.inf

# ==== NOISE SAMPLING & PERTURBATION ====
rng = np.random.default_rng(SEED)
if SENSITIVITY == 0.0:
    # No variation in data per your Δ definition -> zero noise (outputs equal inputs)
    privatized = valid_values.copy()
else:
    noise = rng.laplace(loc=0.0, scale=scale, size=valid_values.shape[0])
    privatized = valid_values + noise

# Optional post-processing for counts
if CLIP_NONNEGATIVE:
    privatized = np.maximum(privatized, 0.0)
if ROUND_TO_INT:
    privatized = np.rint(privatized).astype(int)

# ==== WRITE BACK ====
out_series = pd.Series(index=col.index, dtype=float)
out_series.loc[valid_mask] = privatized
# Preserve NaN positions (e.g., if original had non-numeric/missing)
out_series.loc[~valid_mask] = np.nan

if ADD_AS_NEW_COLUMN:
    df[NEW_COLUMN_NAME] = out_series
else:
    df[COLUMN] = out_series

df.to_csv(OUTPUT_CSV, index=False)

# ==== PRIVACY ACCOUNTING (BASIC COMPOSITION) ====
# Each timestamp (row) you publish costs (ε, δ). Publishing T rows costs:
#   ε_total = T * ε
#   δ_total = T * δ
# This is a simple (pessimistic) bound; tighter bounds require advanced composition.
T = int(valid_mask.sum())
epsilon_total_basic = T * EPSILON
delta_total_basic = T * DELTA

# ==== LOGGING ====
print("=== Differential Privacy (Laplace, input perturbation) ===")
print(f"Input file:  {INPUT_CSV}")
print(f"Output file: {OUTPUT_CSV}")
print(f"Column privatized: '{COLUMN}' -> '{NEW_COLUMN_NAME if ADD_AS_NEW_COLUMN else COLUMN}'")
print(f"Entries privatized (non-NaN): T = {T}")
print(f"Requested per-row (ε, δ): ({EPSILON}, {DELTA})")
print(f"Data-range sensitivity Δ = max - min = {SENSITIVITY:.6g} (min={data_min:.6g}, max={data_max:.6g})")
print(f"Laplace scale b = Δ/ε = {scale:.6g}")
print("--- Composition (basic) over T rows ---")
print(f"Accumulated ε_total = T * ε = {epsilon_total_basic:.6g}")
print(f"Accumulated δ_total = T * δ = {delta_total_basic:.6g}")
print("NOTE: Laplace mechanism itself is (ε, 0)-DP; δ is tracked here per your request and for")
print("      composition bookkeeping. Consider advanced composition/accountants for tighter bounds.")
