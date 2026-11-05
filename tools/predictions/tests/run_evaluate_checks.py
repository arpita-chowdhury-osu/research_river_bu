"""Acceptance run on the 2024 HOPB validation predictions plus a few constructed program tests (v2).

Usage (from the repository root):
  python tools/predictions/tests/run_evaluate_checks.py baseline/results/predictions_val.csv \
      tools/predictions/tests/metrics_reference_val_4f0ef92.json NEW_CHECK_DIR [V1_ACCEPTANCE_DIR]
The 1810 / 364 / 355 numbers are acceptance expectations for that one file only; the tool itself
does not hard-code them. Rejection tests pass only with exit code exactly 2. All JSON outputs are
re-parsed strictly (NaN/Infinity rejected). Constructed tests are program checks, not research cases.
"""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "evaluate_predictions.py"
TOL = 2e-6
COLS = ("image_id", "date", "discharge_lps", "pred_discharge_lps")


def run(inp, out):
    p = subprocess.run([sys.executable, "-I", "-B", str(TOOL), "--input", str(inp), "--out", str(out)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _reject_constant(token):
    raise ValueError(f"non-strict JSON constant {token}")


def strict_load(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def write_csv(path, rows, header=COLS):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    src, ref_path, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    v1_dir = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
    root.mkdir(parents=False, exist_ok=False)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    results = {"input_sha256": before, "acceptance": {}, "constructed": []}

    # ---- 2024 acceptance ----
    acc_dir = root / "acceptance_2024"
    code, stdout, stderr = run(src, acc_dir)
    assert code == 0, stderr
    metrics = strict_load(acc_dir / "metrics.json")
    record = strict_load(acc_dir / "evaluation_record.json")
    with (acc_dir / "daily_scores.csv").open(newline="") as f:
        daily = list(csv.DictReader(f))
    ref = json.loads(ref_path.read_text())["reference_val"]
    hits = sum(r["hit"] == "True" for r in daily)
    comparison = {}
    for k, v in ref.items():
        diff = metrics[k] - v
        tol = 0 if k in ("days", "within_x2") else TOL
        comparison[k] = {"tool": metrics[k], "reference": v, "difference": diff, "tolerance": tol, "pass": abs(diff) <= tol}
    acc = {
        "rows_used": record["input"]["rows"], "dates_out": len(daily), "day_hits": hits,
        "expect_rows_1810": record["input"]["rows"] == 1810, "expect_days_364": len(daily) == 364,
        "expect_hits_355": hits == 355, "negative_predictions": record["validation"]["negative_predictions_above_minus_one"],
        "site_check": {k: record["validation"][k] for k in ("site_column_present", "site")},
        "metrics_undefined_or_non_finite": record["metrics_undefined_or_non_finite"],
        "strict_json_parse": True, "metrics_vs_reference": comparison,
    }
    acc["pass"] = (acc["expect_rows_1810"] and acc["expect_days_364"] and acc["expect_hits_355"]
                   and not acc["metrics_undefined_or_non_finite"] and all(c["pass"] for c in comparison.values()))
    if v1_dir:
        same = {n: hashlib.sha256((acc_dir / n).read_bytes()).hexdigest() == hashlib.sha256((v1_dir / n).read_bytes()).hexdigest()
                for n in ("metrics.json", "daily_scores.csv")}
        acc["byte_identical_to_v1"] = same
        acc["pass"] = acc["pass"] and all(same.values())
    results["acceptance"] = acc

    # ---- constructed program tests ----
    t = root / "constructed"
    t.mkdir()

    def case(name, rows, expect_ok, check=None, header=COLS, raw_bytes=None):
        path = t / f"{name}.csv"
        if raw_bytes is not None:
            path.write_bytes(raw_bytes)
        else:
            write_csv(path, rows, header)
        out_dir = t / f"{name}_out"
        code, _, err = run(path, out_dir)
        ok = code == 0 if expect_ok else code == 2
        detail = err.strip().splitlines()[-1] if err.strip() else ""
        if expect_ok and ok:
            try:
                strict_load(out_dir / "metrics.json"), strict_load(out_dir / "evaluation_record.json")
            except ValueError as e:
                ok, detail = False, f"strict JSON parse failed: {e}"
        if not expect_ok and out_dir.exists():
            ok, detail = False, "output directory created for rejected input"
        if ok and check:
            ok, detail = check(out_dir)
        results["constructed"].append({"case": name, "expected": "accepted (exit 0)" if expect_ok else "rejected (exit 2)",
                                       "exit_code": code, "pass": ok, "detail": detail})

    def log_mean_check(out_dir):
        with (out_dir / "daily_scores.csv").open(newline="") as f:
            d = next(csv.DictReader(f))
        y_log, p_log = (math.log1p(0) + math.log1p(99)) / 2, (math.log1p(9) + math.log1p(9)) / 2
        y_lin = math.log1p((0 + 99) / 2)
        ok = abs(float(d["y"]) - y_log) < 1e-12 and abs(float(d["p"]) - p_log) < 1e-12 and abs(float(d["y"]) - y_lin) > 0.1
        return ok, f"y={d['y']} (log-mean {y_log:.6f}, L/s-mean would be {y_lin:.6f}); p={d['p']}; hit={d['hit']}"

    def negative_check(out_dir):
        n = strict_load(out_dir / "evaluation_record.json")["validation"]["negative_predictions_above_minus_one"]
        return n == 1, f"negative_predictions_above_minus_one={n}"

    def site_check(out_dir):
        v = strict_load(out_dir / "evaluation_record.json")["validation"]
        return v["site_column_present"] and v["site"] == "HOPB", f"site_column_present={v['site_column_present']} site={v['site']}"

    def nonfinite_check(out_dir):
        m = strict_load(out_dir / "metrics.json")
        listed = sorted(u["metric"] for u in strict_load(out_dir / "evaluation_record.json")["metrics_undefined_or_non_finite"])
        text = (out_dir / "metrics.json").read_text()
        ok = (m["r2_log"] is None and m["spearman"] is None and listed == ["r2_log", "spearman"]
              and m["days"] == 1 and math.isfinite(m["rmse_log"]) and math.isfinite(m["within_x2"])
              and "NaN" not in text and "Infinity" not in text)
        return ok, f"metrics={m}; listed_non_finite={listed}"

    case("same_day_log_mean", [["a", "2024-01-01", "0", "9"], ["b", "2024-01-01", "99", "9"],
                               ["c", "2024-01-02", "5", "5"]], True, log_mean_check)
    case("negative_pred_reported", [["a", "2024-01-01", "1", "-0.5"], ["b", "2024-01-02", "2", "2"]], True, negative_check)
    case("single_site_column_ok", [["a", "HOPB", "2024-01-01", "1", "1"], ["b", "HOPB", "2024-01-02", "2", "3"]], True,
         site_check, header=("image_id", "site", "date", "discharge_lps", "pred_discharge_lps"))
    case("single_day_non_finite_metrics_null", [["a", "2024-01-01", "10", "20"], ["b", "2024-01-01", "30", "20"]], True, nonfinite_check)
    case("empty_file", [], False, raw_bytes=b"")
    case("header_only_zero_rows", [], False)
    case("multiple_sites", [["a", "HOPB", "2024-01-01", "1", "1"], ["b", "OKSR", "2024-01-02", "1", "1"]], False,
         header=("image_id", "site", "date", "discharge_lps", "pred_discharge_lps"))
    case("blank_site", [["a", "HOPB", "2024-01-01", "1", "1"], ["b", "", "2024-01-02", "1", "1"]], False,
         header=("image_id", "site", "date", "discharge_lps", "pred_discharge_lps"))
    case("duplicate_id", [["a", "2024-01-01", "1", "1"], ["a", "2024-01-02", "1", "1"]], False)
    case("missing_column", [["a", "2024-01-01", "1"]], False, header=("image_id", "date", "discharge_lps"))
    case("missing_value", [["a", "2024-01-01", "", "1"]], False)
    case("non_finite_pred", [["a", "2024-01-01", "1", "inf"]], False)
    case("non_numeric_label", [["a", "2024-01-01", "abc", "1"]], False)
    case("pred_le_minus_one", [["a", "2024-01-01", "1", "-1"]], False)
    case("negative_label", [["a", "2024-01-01", "-0.1", "1"]], False)
    case("invalid_date", [["a", "2024-02-30", "1", "1"]], False)
    code, _, err = run(t / "same_day_log_mean.csv", t / "same_day_log_mean_out")
    results["constructed"].append({"case": "refuse_existing_output_dir", "expected": "rejected (exit 2)", "exit_code": code,
                                   "pass": code == 2, "detail": err.strip().splitlines()[-1] if err.strip() else ""})

    results["input_unchanged"] = hashlib.sha256(src.read_bytes()).hexdigest() == before
    results["all_pass"] = acc["pass"] and all(c["pass"] for c in results["constructed"]) and results["input_unchanged"]
    text = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    json.loads(text, parse_constant=_reject_constant)
    (root / "check_results.json").write_text(text, encoding="utf-8")
    print(text)
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
