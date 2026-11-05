# Prediction CSV tools

Two small command-line tools for a **saved prediction CSV**. They do not take a model, train, run inference or extract features.

| tool | what it does | needs |
|---|---|---|
| `check_prediction_manifest.py` | Checks that the CSV has the same images, dates and labels as a split file (e.g. `data/discharge/val.csv`). | Python standard library |
| `evaluate_predictions.py` | Scores the CSV with the original `day_scores()` of `baseline/train_linear.py` at commit `4f0ef92`. | numpy, pandas |

Copy the whole `tools/predictions/` folder. `evaluate_predictions.py` will not run without `vendor/train_linear_4f0ef92.py`.

## Input

Required columns: `image_id`, `date` (YYYY-MM-DD), `discharge_lps`, `pred_discharge_lps`.
- Both discharge columns are in **L/s**.
- Evaluate **one site per run**. If a `site` column is present, it must hold a single non-empty value. Without it, make sure all rows come from one site; the tool does not guess the site from `image_id`.
- `baseline/results/predictions_<split>.csv` written by `baseline/train_linear.py` already has these columns.

## Use (run from the repository root)

On macOS/Linux use `python3` if `python` is not Python 3. **`--out` must be a directory that does not exist yet**; the tools never overwrite.

On Windows without long-path support, keep the full output file paths under 260 characters. A longer path makes writing the output fail.

**Step 1: check against the split file first**

```
python tools/predictions/check_prediction_manifest.py --predictions baseline/results/predictions_val.csv --reference data/discharge/val.csv --expected-reference-blob eb5951c6cb52c7582e4209b07180e896e5a89edc --out runs/check_val
```

Expected output: `{"status": "required_match", "differences_rows": 0}`, exit code 0.
`--expected-reference-blob` is optional. With it, the tool refuses a `val.csv` that is not the 4f0ef92 version (git blob `eb5951c6…`).

**Step 2: compute the scores**

```
python tools/predictions/evaluate_predictions.py --input baseline/results/predictions_val.csv --out runs/eval_val
```

This prints the metrics. For the committed validation predictions: `days` 364, `within_x2` 0.9752747252747253 (355 of 364 days), `r2_log` 0.9470870…, `rmse_log` 0.2977664…, `spearman` 0.9746276…

`runs/` is only an example output location; do not commit it.

## Outputs

`check_prediction_manifest.py` → `--out/`
- `summary.json`: status, id counts, per-field results and empty-value counts.
- `differences.csv`: **one row per difference; look here when the status is not `required_match`.** Issue types are `missing_in_predictions`, `extra_in_predictions`, `field_mismatch` and `empty_in_predictions_when_reference_present`.
- A field that is empty on both sides (e.g. `fewshot` in `val.csv`) is counted in both `empty_in_*` counts and is not a difference.

`evaluate_predictions.py` → `--out/`
- `metrics.json`: `days`, `within_x2`, `r2_log`, `rmse_log`, `spearman` as returned by `day_scores()`. A non-finite value (e.g. R² with only one day) is written as `null`, never as 0.
- `daily_scores.csv`: one row per date with `y` and `p` (daily means on the log1p scale) and `hit` (|p − y| ≤ ln 2).
- `evaluation_record.json`: input hash, row and date counts, site check, function source, unit conversion and runtime.

## Exit codes

| code | `check_prediction_manifest.py` | `evaluate_predictions.py` |
|---|---|---|
| 0 | `required_match` | scores written |
| 1 | valid input, differences listed in `differences.csv` | — |
| 2 | invalid input, reference blob mismatch, or `--out` already exists | invalid input or `--out` already exists (no output written) |

## Small example (constructed data, not research data)

`examples/` holds hand-made rows with `demo_*` ids.

```
python tools/predictions/check_prediction_manifest.py --predictions tools/predictions/examples/example_predictions_matching.csv --reference tools/predictions/examples/example_reference.csv --out runs/example_check
python tools/predictions/evaluate_predictions.py --input tools/predictions/examples/example_predictions_matching.csv --out runs/example_eval
python tools/predictions/check_prediction_manifest.py --predictions tools/predictions/examples/example_predictions.csv --reference tools/predictions/examples/example_reference.csv --out runs/example_mismatch
```

Expected results:
1. Exit 0 with `required_match`.
2. Exit 0 with `days` 2 and `within_x2` 1.0.
3. Exit 1 with `partial_or_mismatch`. `runs/example_mismatch/differences.csv` lists `demo_3` as `missing_in_predictions` and `demo_4` as `extra_in_predictions`. A label written as `10.00` instead of `10.0`, or a different row order, is not a difference.

## Scoring rule is fixed to 4f0ef92

`evaluate_predictions.py` does not reimplement the metric. It reads `day_scores()` verbatim from `vendor/train_linear_4f0ef92.py`, an unmodified copy of `baseline/train_linear.py` at commit `4f0ef92506ba00ad6eeda7ab27be7cdf7fc72009` (git blob `2a461fd31c7d941bb7ab882a18db2f5b13a324bb`). The tool checks the file's git blob before use and then runs only that one function. torch and matplotlib are not imported, and nothing is trained.

If `baseline/train_linear.py` changes later, this tool keeps the 4f0ef92 rule; it does not follow automatically. Updating the rule means replacing the vendor file and its hash on purpose.

## Tests (constructed program tests; no images, no models)

```
python tools/predictions/tests/run_manifest_tests.py runs/test_manifest
python tools/predictions/tests/run_evaluate_checks.py baseline/results/predictions_val.csv tools/predictions/tests/metrics_reference_val_4f0ef92.json runs/test_evaluate
```

- The first command runs 27 constructed cases.
- The second runs 17 constructed cases, plus an acceptance run on the committed validation predictions: 1810 rows, 364 days and 355 hit days, compared with the validation metrics of `baseline/results/metrics.json` (copied into `tests/metrics_reference_val_4f0ef92.json`).
- Both print JSON and exit 0 when everything passes.

## What a match does not show

A `required_match` and a set of scores show only that the CSV agrees with the split file and was scored with the 4f0ef92 rule. They do **not** show that the discharge labels are true, or that the model actually read the matching images. The `sha256` column is compared only as declared text; image bytes are not read.
