"""Download the images of one or more splits from the PhenoCam archive (standard library only).

Usage:
  python download.py --split train val test test_oksr --out data/images
  python download.py --split val --limit 50          # quick try

Images are saved as <out>/<site>/<YYYY>/<filename> and checked against the sha256 in the split CSV.
Re-running skips files that are already present and correct, so an interrupted download can simply be restarted.
"""
import argparse
import csv
import hashlib
import pathlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
SPLITS = ["train", "val", "test", "test_oksr"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(row, out, retries=4):
    dest = out / row["site"] / row["date"][:4] / row["filename"]
    if dest.exists() and sha256(dest) == row["sha256"]:
        return "skipped", row["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(row["url"], headers={"User-Agent": "river-phenocam-discharge/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("checksum mismatch")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return "downloaded", row["filename"]
        except Exception as e:  # network error or bad file: wait and retry
            if attempt == retries - 1:
                return f"failed ({e})", row["filename"]
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--out", default=str(HERE / "data" / "images"))
    ap.add_argument("--data", default=str(HERE / "data" / "discharge"), help="folder with the split CSVs")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of each split")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (please keep this small)")
    args = ap.parse_args()

    rows = []
    for split in args.split:
        with open(pathlib.Path(args.data) / f"{split}.csv", newline="") as f:
            rows += list(csv.DictReader(f))[: args.limit]
    out = pathlib.Path(args.out)
    print(f"{len(rows)} images from {', '.join(args.split)} -> {out}", flush=True)

    counts, failed = {"downloaded": 0, "skipped": 0}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, r, out) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            status, name = fut.result()
            if status.startswith("failed"):
                failed.append((name, status))
            else:
                counts[status] += 1
            if i % 200 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  downloaded {counts['downloaded']}, already present {counts['skipped']}, "
                      f"failed {len(failed)}", flush=True)
    for name, status in failed:
        print("FAILED", name, status)
    if failed:
        print(f"{len(failed)} images failed; run the same command again to retry them.")
        sys.exit(1)
    print("done: all images present and verified.")


if __name__ == "__main__":
    main()
