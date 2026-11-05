"""Constructed program tests for check_prediction_manifest.py, plus acceptance conditions of the real 2024 run.

Usage (from the repository root):
  python tools/predictions/tests/run_manifest_tests.py NEW_TEST_DIR [ACCEPTANCE_DIR REFERENCE_CSV]
The small reference below is hand-written test data, not research data. Expected exit codes, issue
types and ids are written here by hand, not copied from the tool's own output.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "check_prediction_manifest.py"
REF_HEADER = ["image_id", "site", "date", "time_local", "split", "discharge_lps", "n_records", "sha256"]
REF_ROWS = [
    ["img_a", "HOPB", "2024-01-01", "11:00:00", "val", "10.5", "1", "aa11"],
    ["img_b", "HOPB", "2024-01-01", "12:00:00", "val", "1.0", "1", "bb22"],
    ["img_c", "HOPB", "2024-01-02", "11:00:00", "val", "0", "1", "cc33"],
    ["img_d", "HOPB", "2024-01-03", "11:00:00", "val", "250.25", "1", "dd44"],
]
PRED_HEADER = REF_HEADER + ["pred_discharge_lps"]


def pred_rows(rows=None):
    return [r + ["5"] for r in (rows if rows is not None else REF_ROWS)]


def write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        w.writerows(rows)


def run(pred, ref, out, extra=()):
    p = subprocess.run([sys.executable, "-I", "-B", str(TOOL), "--predictions", str(pred), "--reference", str(ref),
                        "--out", str(out), *extra], capture_output=True, text=True)
    return p.returncode, p.stderr.strip()


def issues(out):
    with (out / "differences.csv").open(newline="", encoding="utf-8") as f:
        return sorted((r["issue_type"], r["image_id"], r["field"]) for r in csv.DictReader(f))


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=False, exist_ok=False)
    ref = root / "reference_small.csv"
    write(ref, REF_HEADER, REF_ROWS)
    results = []

    def case(name, expect_code, pred_header=PRED_HEADER, rows=None, expect_issues=None, check=None, ref_path=None, extra=()):
        pred = root / f"{name}.csv"
        if rows == "EMPTY_FILE":
            pred.write_bytes(b"")
        else:
            write(pred, pred_header, rows if rows is not None else pred_rows())
        out = root / f"{name}_out"
        code, err = run(pred, ref_path or ref, out, extra)
        ok, detail = code == expect_code, err.splitlines()[-1] if err else ""
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").exists() else None
        if ok and expect_issues is not None:
            got = issues(out)
            ok, detail = got == sorted(expect_issues), f"issues={got}"
        if ok and check:
            ok, detail = check(summary, out)
        if ok and expect_code == 2 and summary is not None:
            ok = summary["status"] == "invalid_input" and not (out / "differences.csv").exists()
        results.append({"case": name, "expected_exit": expect_code, "exit_code": code, "pass": ok, "detail": detail})

    shuffled = [REF_ROWS[i] for i in (3, 1, 0, 2)]
    case("identical", 0, expect_issues=[])
    case("shuffled_order", 0, rows=pred_rows(shuffled), expect_issues=[],
         check=lambda s, o: (s["comparison"]["row_order_differs"] is True, "row_order_differs=True"))
    case("one_missing", 1, rows=pred_rows(REF_ROWS[:3]), expect_issues=[("missing_in_predictions", "img_d", "image_id")])
    case("one_extra", 1, rows=pred_rows(REF_ROWS + [["img_e", "HOPB", "2024-01-04", "11:00:00", "val", "3", "1", "ee55"]]),
         expect_issues=[("extra_in_predictions", "img_e", "image_id")])
    swapped = [r[:] for r in REF_ROWS]
    swapped[2][0] = "img_x"
    case("same_count_wrong_id", 1, rows=pred_rows(swapped),
         expect_issues=[("extra_in_predictions", "img_x", "image_id"), ("missing_in_predictions", "img_c", "image_id")])
    changed = [r[:] for r in REF_ROWS]
    changed[3][5] = "250.26"
    case("label_changed", 1, rows=pred_rows(changed), expect_issues=[("field_mismatch", "img_d", "discharge_lps")])
    redated = [r[:] for r in REF_ROWS]
    redated[1][2] = "2024-01-02"
    case("date_changed", 1, rows=pred_rows(redated), expect_issues=[("field_mismatch", "img_b", "date")])
    text_num = [r[:] for r in REF_ROWS]
    text_num[1][5] = "1.0000"
    text_num[2][5] = "0.0"
    case("label_text_format_only", 0, rows=pred_rows(text_num), expect_issues=[])
    site_conf = [r[:] for r in REF_ROWS]
    site_conf[0][1] = "OKSR"
    case("optional_site_conflict", 1, rows=pred_rows(site_conf), expect_issues=[("field_mismatch", "img_a", "site")])
    sha_conf = [r[:] for r in REF_ROWS]
    sha_conf[2][7] = "CC33"
    sha_conf[3][7] = "ffff"
    case("optional_sha256_conflict_case_insensitive", 1, rows=pred_rows(sha_conf),
         expect_issues=[("field_mismatch", "img_d", "sha256")])
    blank_opt = [r[:] for r in REF_ROWS]
    blank_opt[0][4] = ""
    case("optional_value_empty_in_predictions", 1, rows=pred_rows(blank_opt),
         expect_issues=[("empty_in_predictions_when_reference_present", "img_a", "split")])
    minimal = [[r[0], r[2], r[5], "5"] for r in REF_ROWS]

    def not_provided(s, o):
        np_ = s["optional_fields_not_provided"]
        return (sorted(np_) == sorted(["site", "split", "slot", "time_local", "filename", "url", "fewshot", "n_records", "sha256"]),
                f"not_provided={np_}")
    case("optional_fields_absent", 0, pred_header=["image_id", "date", "discharge_lps", "pred_discharge_lps"],
         rows=minimal, expect_issues=[], check=not_provided)
    neg = pred_rows()
    neg[0][-1] = "-0.5"
    case("negative_pred_above_minus_one_warning", 0, rows=neg, expect_issues=[],
         check=lambda s, o: (s["warnings"]["negative_predictions_above_minus_one"] == 1, "warning count 1"))
    dup = pred_rows(REF_ROWS + [REF_ROWS[0]])
    case("duplicate_id_predictions", 2, rows=dup)
    dup_ref = root / "reference_dup.csv"
    write(dup_ref, REF_HEADER, REF_ROWS + [REF_ROWS[1]])
    case("duplicate_id_reference", 2, ref_path=dup_ref)
    case("empty_file", 2, rows="EMPTY_FILE")
    case("header_only", 2, rows=[])
    case("missing_pred_column", 2, pred_header=REF_HEADER, rows=[r[:] for r in REF_ROWS])
    nonfin = pred_rows()
    nonfin[1][-1] = "nan"
    case("non_finite_prediction", 2, rows=nonfin)
    dom = pred_rows()
    dom[1][-1] = "-1"
    case("prediction_outside_log1p_domain", 2, rows=dom)
    amb = pred_rows()
    amb[2][2] = "01/02/2024"
    case("ambiguous_date_format", 2, rows=amb)
    case("reference_blob_mismatch", 2, extra=("--expected-reference-blob", "0" * 40))
    code, err = run(root / "identical.csv", ref, root / "identical_out")
    results.append({"case": "refuse_existing_output_dir", "expected_exit": 2, "exit_code": code, "pass": code == 2,
                    "detail": err.splitlines()[-1] if err else ""})

    # ---- v2: empty-value counts over common ids ----
    fs_ref = root / "reference_fewshot.csv"
    fs_header = ["image_id", "date", "discharge_lps", "fewshot", "split"]
    fs_rows = [["e1", "2024-01-01", "1", "", "val"], ["e2", "2024-01-01", "2", "", "val"], ["e3", "2024-01-02", "3", "", ""]]
    write(fs_ref, fs_header, fs_rows)
    fs_pred_header = fs_header + ["pred_discharge_lps"]

    def counts(field, exp):
        def check(s, o):
            got = {k: s["comparison"]["optional_fields"][field][k] for k in exp}
            return got == exp, f"{field} {got}"
        return check

    case("v2_both_sides_empty", 0, pred_header=fs_pred_header, ref_path=fs_ref,
         rows=[r + ["1"] for r in fs_rows], expect_issues=[],
         check=counts("fewshot", {"empty_in_predictions": 3, "empty_in_reference": 3,
                                  "empty_in_predictions_when_reference_present": 0, "mismatches": 0}))
    one_side = [r[:] for r in fs_rows]
    one_side[0][4] = ""
    case("v2_prediction_empty_reference_present", 1, pred_header=fs_pred_header, ref_path=fs_ref,
         rows=[r + ["1"] for r in one_side],
         expect_issues=[("empty_in_predictions_when_reference_present", "e1", "split")],
         check=counts("split", {"empty_in_predictions": 2, "empty_in_reference": 1,
                                "empty_in_predictions_when_reference_present": 1, "mismatches": 0}))
    ref_empty = [r[:] for r in fs_rows]
    ref_empty[2][4] = "val"
    case("v2_reference_empty_prediction_present", 1, pred_header=fs_pred_header, ref_path=fs_ref,
         rows=[r + ["1"] for r in ref_empty], expect_issues=[("field_mismatch", "e3", "split")],
         check=counts("split", {"empty_in_predictions": 0, "empty_in_reference": 1,
                                "empty_in_predictions_when_reference_present": 0, "mismatches": 1}))
    case("v2_empty_counts_common_ids_only", 1, pred_header=fs_pred_header, ref_path=fs_ref,
         rows=[r + ["1"] for r in fs_rows[:2]] + [["e9", "2024-01-03", "4", "", "", "1"]],
         expect_issues=[("extra_in_predictions", "e9", "image_id"), ("missing_in_predictions", "e3", "image_id")],
         check=counts("fewshot", {"empty_in_predictions": 2, "empty_in_reference": 2,
                                  "empty_in_predictions_when_reference_present": 0}))

    report = {"constructed": results, "constructed_all_pass": all(r["pass"] for r in results),
              "note_v2": "case optional_value_empty_in_predictions now expects issue_type "
                         "empty_in_predictions_when_reference_present (renamed in v2; same rows flagged)"}
    if len(sys.argv) > 3:
        acc, ref_csv = Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
        s = json.loads((acc / "summary.json").read_text(encoding="utf-8"))
        with ref_csv.open(newline="", encoding="utf-8") as f:
            rr = list(csv.DictReader(f))
        cond = {"reference_blob_match": s["reference"]["blob_check"] == "match"
                and s["reference"]["git_blob_sha1"] == "eb5951c6cb52c7582e4209b07180e896e5a89edc",
                "reference_1810_unique_ids": s["reference"]["unique_image_ids"] == 1810 == s["reference"]["rows"],
                "reference_364_dates": s["reference"]["dates"] == 364,
                "reference_scope_HOPB_2024_val": {r["site"] for r in rr} == {"HOPB"} and {r["split"] for r in rr} == {"val"}
                and {r["date"][:4] for r in rr} == {"2024"},
                "predictions_sha256": s["predictions"]["sha256"] == "3a9326656a3d43b5584c4f16161cdbed2563b1055aa51f12dd7673af26a274eb",
                "status": s["status"], "differences_rows": s["differences_rows"]}
        report["acceptance_2024_conditions"] = cond
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (root / "test_results.json").write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["constructed_all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
