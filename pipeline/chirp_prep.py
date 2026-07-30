# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Prep a CHIRP re-ID subset for local embedding: parse Annotation.csv, pick a split, subsample
frames per tracklet, and emit (a) a manifest and (b) an rclone files-from list (frames + the
per-video-territory masks_ring.csv needed for band-masking).

CHIRP ships its OWN standardized Annotation.csv (identity + closed/disjointed/open splits + dates),
so we parse it directly -- no extra dataset library needed. Downstream: fetch the listed files
locally (e.g. `rclone copy --files-from`), then pipeline/chirp_embed.py.

Annotation.csv columns: Video,Tracklet,id,Territory,Year,img,UnqTrack,ClosedSetSplit,
DisjointedSetSplit,OpenSetSplit,Date. `img` is relative to the ReID/ root
(e.g. data/{id}/{video_terr}/Tracklet-NNN/frame.jpg); masks_ring.csv sits at data/{id}/{video_terr}/.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotation", default="chirp_work/ReID/Annotation.csv")
    ap.add_argument("--split-col", default="DisjointedSetSplit",
                    help="ClosedSetSplit | DisjointedSetSplit | OpenSetSplit")
    ap.add_argument("--splits", default="Query,Gallery", help="comma list of split values to keep")
    ap.add_argument("--frames-per-tracklet", type=int, default=8,
                    help="evenly-spaced frames sampled per 25-frame tracklet (0 = all)")
    ap.add_argument("--out-manifest", default="chirp_work/manifest.json")
    ap.add_argument("--out-files", default="chirp_work/files_from.txt")
    args = ap.parse_args()

    keep = set(args.splits.split(","))
    col = args.split_col
    by_tr = defaultdict(list)  # UnqTrack -> rows
    with open(args.annotation) as f:
        for r in csv.DictReader(f):
            if r[col] in keep:
                by_tr[r["UnqTrack"]].append(r)

    def subsample(rows):
        rows = sorted(rows, key=lambda r: r["img"])
        n = args.frames_per_tracklet
        if n <= 0 or n >= len(rows):
            return rows
        step = len(rows) / n
        return [rows[int(i * step)] for i in range(n)]

    manifest, files = [], set()
    per_split = defaultdict(int)
    for tr, rows in by_tr.items():
        for r in subsample(rows):
            manifest.append({"img": r["img"], "id": r["id"], "unqtrack": r["UnqTrack"],
                             "video": r["Video"], "territory": r["Territory"],
                             "split": r[col], "date": r["Date"]})
            files.add(r["img"])
            files.add(str(PurePosixPath(r["img"]).parents[1] / "masks_ring.csv"))
            per_split[r[col]] += 1

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(args.out_manifest, "w"))
    Path(args.out_files).write_text("\n".join(sorted(files)) + "\n")

    n_ids = len({m["id"] for m in manifest})
    n_tr = len({m["unqtrack"] for m in manifest})
    n_mask = sum(1 for p in files if p.endswith("masks_ring.csv"))
    print(f"split {col} in {keep}: {n_tr} tracklets, {n_ids} individuals")
    print(f"  frames selected: {len(manifest)}  ({dict(per_split)})")
    print(f"  files to fetch: {len(files)} ({len(files)-n_mask} frames + {n_mask} masks_ring.csv)")
    print(f"  wrote {args.out_manifest} and {args.out_files}")


if __name__ == "__main__":
    main()
