"""Evaluate a saved prediction CSV with the shared repository's original day_scores() (v2).

Usage: python evaluate_predictions.py --input PREDICTIONS.csv --out NEW_OUTPUT_DIR

One call evaluates one set of predictions from ONE site. Required columns: image_id, date,
discharge_lps, pred_discharge_lps (both discharge columns in L/s). If a site column exists it must be
non-empty and hold a single value; without it the caller must guarantee a single site (the site is
never guessed from image_id).
The metric code is NOT reimplemented here: day_scores() is read verbatim from
vendor/train_linear_4f0ef92.py (research_river_bu @ 4f0ef92, baseline/train_linear.py, git blob
2a461fd3...), whose git blob hash is checked before use. Only that one function is executed; the
module itself is not imported, because its top-level imports pull in torch/matplotlib (training).

This wrapper only: validates the input, converts pred_discharge_lps to log1p (day_scores expects a
log1p-scale prediction and takes discharge_lps in L/s itself), calls day_scores, and writes
metrics.json, daily_scores.csv and evaluation_record.json into a new directory. Metrics that the
original function returns as non-finite are written as JSON null and listed in the record; all JSON is
written strictly (no NaN/Infinity tokens) and re-parsed strictly before it is saved.
Invalid input is rejected with exit code 2; rows are never dropped, filled or clipped.
"""
import argparse
import ast
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "train_linear_4f0ef92.py"
SOURCE = {
    "repository": "https://github.com/arpita-chowdhury-osu/research_river_bu",
    "commit": "4f0ef92506ba00ad6eeda7ab27be7cdf7fc72009",
    "path": "baseline/train_linear.py",
    "git_blob_sha1": "2a461fd31c7d941bb7ab882a18db2f5b13a324bb",
    "function": "day_scores(rows, pred)",
}
REQUIRED = ["image_id", "date", "discharge_lps", "pred_discharge_lps"]


class InputError(Exception):
    pass


def git_blob_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_day_scores():
    """Return the original day_scores, executed from its verbatim source segment."""
    data = VENDOR.read_bytes()
    if git_blob_sha1(data) != SOURCE["git_blob_sha1"]:
        raise RuntimeError(f"{VENDOR} does not match git blob {SOURCE['git_blob_sha1']}")
    text = data.decode("utf-8")
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "day_scores")
    segment = ast.get_source_segment(text, node)
    namespace = {"np": np, "pd": pd}  # the only names the original function uses
    exec(compile(segment, f"{SOURCE['path']}@{SOURCE['commit'][:7]}", "exec"), namespace)
    info = dict(SOURCE, vendored_file=str(VENDOR.relative_to(HERE)).replace("\\", "/"),
                vendored_file_sha256=hashlib.sha256(data).hexdigest(),
                source_lines=[node.lineno, node.end_lineno],
                extracted_segment_sha256=hashlib.sha256(segment.encode("utf-8")).hexdigest(),
                extraction="ast.get_source_segment of the unmodified file; executed with np=numpy, pd=pandas")
    return namespace["day_scores"], info


def _reject_constant(token):
    raise ValueError(f"non-strict JSON constant {token}")


def write_strict_json(path, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    json.loads(text, parse_constant=_reject_constant)
    path.write_text(text, encoding="utf-8")


def null_non_finite(metrics):
    """Replace non-finite metric values by None (JSON null); values are not changed otherwise."""
    clean, undefined = {}, []
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            clean[k] = None
            undefined.append({"metric": k, "returned_value": repr(v)})
        else:
            clean[k] = v
    return clean, undefined


def read_and_validate(path):
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise InputError("input CSV is empty (no header and no data rows)")
    if len(raw) == 0:
        raise InputError("input CSV has no data rows")
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise InputError(f"missing required column(s): {missing_cols}")
    for c in REQUIRED + (["site"] if "site" in raw.columns else []):
        blank = raw[c].str.strip().eq("")
        if blank.any():
            raise InputError(f"column {c} has {int(blank.sum())} empty value(s), first at data row {int(blank.idxmax()) + 1}")
    if "site" in raw.columns and raw.site.nunique() != 1:
        raise InputError(f"one site per evaluation; site column has {raw.site.nunique()} values: {sorted(raw.site.unique())[:5]}")
    dup = raw.image_id.duplicated(keep=False)
    if dup.any():
        raise InputError(f"image_id not unique: {sorted(raw.image_id[dup].unique())[:5]}")
    dates = pd.to_datetime(raw.date, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise InputError(f"invalid date(s) (expected YYYY-MM-DD): {raw.date[dates.isna()].unique()[:5].tolist()}")
    values = {}
    for c in ["discharge_lps", "pred_discharge_lps"]:
        v = pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(v)
        if bad.any():
            raise InputError(f"column {c} has {int(bad.sum())} non-numeric/non-finite value(s): {raw[c][bad].unique()[:5].tolist()}")
        values[c] = v
    if (values["discharge_lps"] < 0).any():
        raise InputError(f"discharge_lps has {int((values['discharge_lps'] < 0).sum())} negative value(s); log1p label invalid")
    if (values["pred_discharge_lps"] <= -1).any():
        raise InputError(f"pred_discharge_lps has {int((values['pred_discharge_lps'] <= -1).sum())} value(s) <= -1; log1p undefined")
    rows = raw.copy()
    rows["discharge_lps"] = values["discharge_lps"]
    rows["pred_discharge_lps"] = values["pred_discharge_lps"]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    src, out = args.input.resolve(), args.out.resolve()
    try:
        if out.exists():
            raise InputError(f"output directory already exists, refusing to overwrite: {out}")
        if not src.is_file():
            raise InputError(f"input file not found: {src}")
        rows = read_and_validate(src)
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    day_scores, function_info = load_day_scores()
    pred_log1p = np.log1p(rows.pred_discharge_lps.to_numpy(dtype=float))
    daily, returned = day_scores(rows, pred_log1p)
    metrics, undefined = null_non_finite(returned)

    out.mkdir(parents=True)
    write_strict_json(out / "metrics.json", metrics)
    daily.to_csv(out / "daily_scores.csv", index_label="date", date_format="%Y-%m-%d")
    data = src.read_bytes()
    negative = rows.pred_discharge_lps < 0
    has_site = "site" in rows.columns
    record = {
        "input": {"path": str(src), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "rows": int(len(rows)), "unique_image_ids": int(rows.image_id.nunique()),
                  "dates": int(rows.date.nunique()), "columns": list(rows.columns)},
        "validation": {"required_columns": REQUIRED, "rows_dropped_filled_or_clipped": 0,
                       "site_column_present": has_site,
                       "site": rows.site.iloc[0] if has_site else None,
                       "site_note": "single site checked from site column" if has_site else
                       "no site column; caller must ensure the predictions come from one site (not inferred from image_id)",
                       "negative_predictions_above_minus_one": int(negative.sum()),
                       "negative_prediction_image_ids": rows.image_id[negative].tolist()[:50]},
        "unit_conversion": {"pred_for_function": "numpy.log1p(pred_discharge_lps), pred in L/s",
                            "discharge_lps": "passed in L/s unchanged; day_scores applies log1p itself",
                            "no_expm1_applied": True},
        "function_source": function_info,
        "metrics_undefined_or_non_finite": undefined,
        "metrics_null_policy": "non-finite values returned by the original day_scores are written as JSON null; never replaced by 0",
        "outputs": ["metrics.json", "daily_scores.csv", "evaluation_record.json"],
        "json_serialization": "json.dumps(allow_nan=False), re-parsed with NaN/Infinity rejected before writing",
        "not_checked": ["truth or quality of discharge labels", "how predictions were produced",
                        "site identity beyond the site column, year and split scope of the input", "image files"],
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "argv": sys.argv, "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    }
    write_strict_json(out / "evaluation_record.json", record)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    if negative.any():
        print(f"NOTE: {int(negative.sum())} prediction(s) between -1 and 0 L/s were evaluated as-is", file=sys.stderr)
    if undefined:
        print(f"NOTE: non-finite metric(s) written as null: {[u['metric'] for u in undefined]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
