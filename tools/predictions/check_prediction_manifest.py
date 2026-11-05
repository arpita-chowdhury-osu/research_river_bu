"""Check that a prediction CSV covers the same images, dates and labels as a fixed validation manifest.

Usage: python check_prediction_manifest.py --predictions PRED.csv --reference VAL.csv --out NEW_DIR
                                           [--expected-reference-blob GIT_BLOB_SHA1]

This is a pre-check before evaluation. It does not compute model metrics, does not judge whether the
labels are true, and cannot prove that a prediction was produced from the matching image.
Neither input file is modified, filled, filtered or reordered; alignment happens in memory only.

Exit codes: 0 = required_match, 1 = partial_or_mismatch (valid inputs, differences listed),
            2 = invalid_input or reference-source error (no comparison result claimed).

Comparison rules (declared before any comparison; not tuned on results):
  * image_id: exact string after stripping surrounding whitespace; must be non-empty and unique in each file.
  * date: must match YYYY-MM-DD and be a real calendar date in both files; compared as that string.
  * discharge_lps: parsed as float; equal if |ref - pred| <= 1e-10 (absolute; relative tolerance 0).
  * optional fields, compared only if present in both files:
      n_records -> integer value; sha256 -> lower-case hex text (declared image hash only);
      site, split, slot, time_local, filename, url, fewshot -> exact text after stripping whitespace.
    Absent from the prediction file -> not_provided (not checked, not an error).
    Empty-value counts (v2) are over common ids only: empty_in_predictions / empty_in_reference count
    every empty value on that side (both-empty counts on both sides and is not a difference);
    empty_in_predictions_when_reference_present counts prediction-empty with reference value present
    (each such row is a difference). Missing/extra ids are reported separately.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import date as _date
from pathlib import Path

PRED_REQUIRED = ["image_id", "date", "discharge_lps", "pred_discharge_lps"]
REF_REQUIRED = ["image_id", "date", "discharge_lps"]
LABEL_ABS_TOL = 1e-10
LABEL_REL_TOL = 0.0
OPTIONAL = {
    "site": "text", "split": "text", "slot": "text", "time_local": "text", "filename": "text",
    "url": "text", "fewshot": "text", "n_records": "integer", "sha256": "hex_lower",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DIFF_COLUMNS = ["issue_type", "image_id", "field", "reference_value", "prediction_value", "detail"]


class InvalidInput(Exception):
    pass


def git_blob_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _reject_constant(token):
    raise ValueError(f"non-strict JSON constant {token}")


def write_strict_json(path, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    json.loads(text, parse_constant=_reject_constant)
    path.write_text(text, encoding="utf-8")


def read_table(path, required, role):
    """Read a CSV as text rows (original values kept) and validate identity/required fields."""
    data = path.read_bytes()
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise InvalidInput(f"{role}: file is empty (no header)")
    columns = [c.strip() for c in reader.fieldnames]
    if len(set(columns)) != len(columns):
        raise InvalidInput(f"{role}: duplicate column names")
    missing = [c for c in required if c not in columns]
    if missing:
        raise InvalidInput(f"{role}: missing required column(s) {missing}")
    rows = [{k.strip(): (v if v is not None else "") for k, v in r.items() if k is not None} for r in reader]
    if not rows:
        raise InvalidInput(f"{role}: no data rows")
    ids, seen, dups = [], set(), set()
    for i, r in enumerate(rows, start=1):
        for c in required:
            if r.get(c, "").strip() == "":
                raise InvalidInput(f"{role}: empty {c} at data row {i}")
        iid = r["image_id"].strip()
        if iid in seen:
            dups.add(iid)
        seen.add(iid)
        ids.append(iid)
        d = r["date"].strip()
        if not DATE_RE.match(d):
            raise InvalidInput(f"{role}: date '{r['date']}' at data row {i} is not YYYY-MM-DD")
        try:
            _date.fromisoformat(d)
        except ValueError:
            raise InvalidInput(f"{role}: date '{r['date']}' at data row {i} is not a calendar date")
        for c in [c for c in ("discharge_lps", "pred_discharge_lps") if c in required]:
            try:
                v = float(r[c])
            except ValueError:
                raise InvalidInput(f"{role}: {c} '{r[c]}' at data row {i} is not numeric")
            if not math.isfinite(v):
                raise InvalidInput(f"{role}: {c} '{r[c]}' at data row {i} is not finite")
            if c == "discharge_lps" and v < 0:
                raise InvalidInput(f"{role}: discharge_lps {r[c]} at data row {i} is negative (log1p domain)")
            if c == "pred_discharge_lps" and v <= -1:
                raise InvalidInput(f"{role}: pred_discharge_lps {r[c]} at data row {i} is <= -1 (log1p domain)")
    if dups:
        raise InvalidInput(f"{role}: image_id not unique ({len(dups)} duplicated, e.g. {sorted(dups)[:3]})")
    info = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "git_blob_sha1": git_blob_sha1(data),
            "bytes": len(data), "rows": len(rows), "unique_image_ids": len(seen),
            "dates": len({r["date"].strip() for r in rows}), "columns": columns}
    return rows, ids, info


def compare_optional(kind, ref_value, pred_value):
    r, p = ref_value.strip(), pred_value.strip()
    if kind == "integer":
        try:
            return int(r) == int(p), None
        except ValueError:
            return r == p, "not an integer; compared as text"
    if kind == "hex_lower":
        return r.lower() == p.lower(), None
    return r == p, None


def compare(ref_rows, ref_ids, pred_rows, pred_ids, ref_cols, pred_cols):
    diffs = []
    ref_by = {i: r for i, r in zip(ref_ids, ref_rows)}
    pred_by = {i: r for i, r in zip(pred_ids, pred_rows)}
    missing = [i for i in ref_ids if i not in pred_by]
    extra = [i for i in pred_ids if i not in ref_by]
    for i in missing:
        diffs.append(["missing_in_predictions", i, "image_id", i, "", "id in reference, absent from predictions"])
    for i in extra:
        diffs.append(["extra_in_predictions", i, "image_id", "", i, "id in predictions, absent from reference"])
    common = [i for i in ref_ids if i in pred_by]
    pred_order_common = [i for i in pred_ids if i in ref_by]
    field_counts = {"date": 0, "discharge_lps": 0}
    for i in common:
        r, p = ref_by[i], pred_by[i]
        if r["date"].strip() != p["date"].strip():
            field_counts["date"] += 1
            diffs.append(["field_mismatch", i, "date", r["date"], p["date"], "YYYY-MM-DD string comparison"])
        a, b = float(r["discharge_lps"]), float(p["discharge_lps"])
        if abs(a - b) > LABEL_ABS_TOL + LABEL_REL_TOL * abs(a):
            field_counts["discharge_lps"] += 1
            diffs.append(["field_mismatch", i, "discharge_lps", r["discharge_lps"], p["discharge_lps"],
                          f"abs diff {abs(a - b):.6g} > {LABEL_ABS_TOL}"])
    optional = {}
    for f, kind in OPTIONAL.items():
        if f not in pred_cols:
            optional[f] = {"status": "not_provided", "comparison": kind}
            continue
        if f not in ref_cols:
            optional[f] = {"status": "not_in_reference", "comparison": kind}
            continue
        mism = empty_pred = empty_ref = empty_pred_ref_present = 0
        notes = set()
        for i in common:
            rv, pv = ref_by[i].get(f, ""), pred_by[i].get(f, "")
            p_empty, r_empty = pv.strip() == "", rv.strip() == ""
            empty_pred += p_empty
            empty_ref += r_empty
            if p_empty and not r_empty:
                empty_pred_ref_present += 1
                diffs.append(["empty_in_predictions_when_reference_present", i, f, rv, pv,
                              "prediction value empty, reference value present"])
                continue
            same, note = compare_optional(kind, rv, pv)
            if note:
                notes.add(note)
            if not same:
                mism += 1
                diffs.append(["field_mismatch", i, f, rv, pv, f"{kind} comparison" + (f"; {note}" if note else "")])
        optional[f] = {"status": "checked", "comparison": kind, "count_scope": "common_ids",
                       "mismatches": mism, "empty_in_predictions": empty_pred, "empty_in_reference": empty_ref,
                       "empty_in_predictions_when_reference_present": empty_pred_ref_present, "notes": sorted(notes)}
    return diffs, {
        "missing_in_predictions": len(missing), "extra_in_predictions": len(extra), "common_ids": len(common),
        "id_sets_equal": not missing and not extra,
        "row_order_differs": pred_order_common != common,
        "required_field_mismatches": field_counts, "optional_fields": optional,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expected-reference-blob", default=None,
                    help="git blob SHA-1 the reference file must have (source check)")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        print(f"ERROR: output directory already exists, refusing to overwrite: {out}", file=sys.stderr)
        return 2
    summary = {"tool": "check_prediction_manifest.py",
               "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Same image_ids, dates and labels (and provided optional metadata) as the reference manifest. "
                        "Not a label-truth check; does not show predictions came from the matching images; "
                        "sha256 column = declared image hash only, image bytes not read; no model metrics computed.",
               "rules": {"image_id": "exact after stripping whitespace; unique and non-empty in both files",
                         "date": "YYYY-MM-DD calendar date required in both; string compare after stripping whitespace",
                         "discharge_lps": {"abs_tol": LABEL_ABS_TOL, "rel_tol": LABEL_REL_TOL},
                         "optional": OPTIONAL, "normalisation": "surrounding whitespace stripped for comparison only; files unchanged"}}
    try:
        for p in (args.predictions, args.reference):
            if not p.is_file():
                raise InvalidInput(f"file not found: {p}")
        ref_rows, ref_ids, ref_info = read_table(args.reference.resolve(), REF_REQUIRED, "reference")
        if args.expected_reference_blob and ref_info["git_blob_sha1"] != args.expected_reference_blob.lower():
            raise InvalidInput(f"reference git blob {ref_info['git_blob_sha1']} != expected {args.expected_reference_blob}")
        pred_rows, pred_ids, pred_info = read_table(args.predictions.resolve(), PRED_REQUIRED, "predictions")
    except InvalidInput as e:
        out.mkdir(parents=True)
        summary.update({"status": "invalid_input", "comparison_performed": False, "error": str(e),
                        "expected_reference_blob": args.expected_reference_blob})
        write_strict_json(out / "summary.json", summary)
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    diffs, result = compare(ref_rows, ref_ids, pred_rows, pred_ids, ref_info["columns"], pred_info["columns"])
    negative = [i for i, r in zip(pred_ids, pred_rows) if float(r["pred_discharge_lps"]) < 0]
    status = "required_match" if not diffs else "partial_or_mismatch"
    out.mkdir(parents=True)
    with (out / "differences.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DIFF_COLUMNS)
        w.writerows(diffs)
    summary.update({
        "status": status, "comparison_performed": True,
        "reference": dict(ref_info, expected_git_blob_sha1=args.expected_reference_blob,
                          blob_check="match" if args.expected_reference_blob else "not_requested"),
        "predictions": pred_info,
        "comparison": result,
        "differences_rows": len(diffs),
        "optional_fields_not_provided": [k for k, v in result["optional_fields"].items() if v["status"] == "not_provided"],
        "warnings": {"negative_predictions_above_minus_one": len(negative), "examples": negative[:20],
                     "note": "allowed by the evaluation contract (pred > -1); not a manifest difference"},
    })
    write_strict_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "differences_rows")}, allow_nan=False))
    return 0 if status == "required_match" else 1


if __name__ == "__main__":
    sys.exit(main())
