# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Evaluate CHIRP re-ID from frame embeddings: aggregate per tracklet, then query->gallery
retrieval (rank-1 + mAP) on the disjointed split, optionally under the territorial-gallery
constraint (the setting where MegaDescriptor scored ~0.28 Top-1).

Input: an embed dir from pipeline/chirp_embed.py (frame_emb.npy + frame_meta.json with
unqtrack/id/split/video). This is pure geometry on frozen or trained embeddings; no labels leak
(query and gallery tracklets are disjoint by construction of the split).
"""
import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def aggregate(emb, meta, agg):
    """per-tracklet embedding: max or mean over frames, then L2-normalize."""
    by_tr = defaultdict(list)
    for i, m in enumerate(meta):
        by_tr[m["unqtrack"]].append(i)
    tr_ids, X, info = [], [], []
    for tr, idx in by_tr.items():
        V = emb[idx]
        v = V.max(0) if agg == "max" else V.mean(0)
        v = v / (np.linalg.norm(v) + 1e-9)
        tr_ids.append(tr); X.append(v)
        m0 = meta[idx[0]]
        info.append({"id": m0["id"], "split": m0["split"], "video": m0.get("video", "")})
    return tr_ids, np.array(X, "float32"), info


def average_precision(ranked_ids, true_id):
    rel = np.array([r == true_id for r in ranked_ids])
    if not rel.any():
        return 0.0
    cum = np.cumsum(rel)
    prec = cum / (np.arange(len(rel)) + 1)
    return float((prec * rel).sum() / rel.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emb", required=True, help="dir with frame_emb.npy + frame_meta.json")
    ap.add_argument("--agg", choices=["max", "mean"], default="max")
    ap.add_argument("--query-split", default="Query")
    ap.add_argument("--gallery-split", default="Gallery")
    ap.add_argument("--possible-birds", default="",
                    help="PossibleBirds_Territory.csv to enable the territorial-gallery constraint")
    ap.add_argument("--among", action="store_true",
                    help="open-set mode: all-vs-all leave-self-out retrieval among ALL tracklets in the "
                         "emb dir (e.g. Unknown_Test individuals never seen in head training)")
    ap.add_argument("--cross-video", action="store_true",
                    help="in --among mode, exclude same-video tracklets from each query's gallery "
                         "(honest cross-session retrieval; kills the same-session near-dup inflation)")
    args = ap.parse_args()

    emb = np.load(Path(args.emb) / "frame_emb.npy").astype("float32")
    meta = json.load(open(Path(args.emb) / "frame_meta.json"))
    tr_ids, X, info = aggregate(emb, meta, args.agg)

    if args.among:
        gid = np.array([d["id"] for d in info])
        vids = np.array([d.get("video", "") for d in info])
        r1 = ap_sum = 0.0
        scored = 0
        for qi in range(len(tr_ids)):
            mask = np.ones(len(tr_ids), bool); mask[qi] = False
            if args.cross_video:
                mask &= vids != vids[qi]
            # only score queries that actually have a retrievable same-id match under this gallery
            if not (gid[mask] == info[qi]["id"]).any():
                continue
            order = np.argsort(-(X[mask] @ X[qi]))
            ranked = gid[mask][order]
            r1 += int(ranked[0] == info[qi]["id"])
            ap_sum += average_precision(ranked, info[qi]["id"])
            scored += 1
        mode = "cross-video (honest cross-session)" if args.cross_video else "all-vs-all (same-session allowed)"
        print(f"open-set among {len(tr_ids)} tracklets | {len(set(gid))} individuals | agg={args.agg} | {mode}")
        print(f"  rank-1 = {r1/scored:.3f}   mAP = {ap_sum/scored:.3f}   (scored {scored} findable queries)")
        return

    q = [i for i, d in enumerate(info) if d["split"] == args.query_split]
    g = [i for i, d in enumerate(info) if d["split"] == args.gallery_split]
    gid = np.array([info[i]["id"] for i in g])
    gvec = X[g]
    print(f"tracklets: {len(q)} query / {len(g)} gallery | {len(set(gid))} gallery ids | agg={args.agg}")

    possible = None
    if args.possible_birds:
        possible = {}
        with open(args.possible_birds) as f:
            for r in csv.DictReader(f):
                possible[r["Video"]] = set(ast.literal_eval(r["PossibleBirds"]))

    def run(constrained):
        r1 = r3 = ap_sum = n = miss_true = 0
        for i in q:
            qv, qid, qvid = X[i], info[i]["id"], info[i]["video"]
            mask = np.ones(len(g), bool)
            if constrained and possible is not None:
                cand = possible.get(qvid, set())
                mask = np.array([gg in cand for gg in gid])
                if not mask.any():
                    continue
                if qid not in cand:
                    miss_true += 1        # true id excluded by the territorial prior -> unfindable
            sims = gvec[mask] @ qv
            order = np.argsort(-sims)
            ranked = gid[mask][order]
            r1 += int(ranked[0] == qid)
            r3 += int(qid in ranked[:3])
            ap_sum += average_precision(ranked, qid)
            n += 1
        return r1 / n, r3 / n, ap_sum / n, n, miss_true

    print("\n-- unconstrained gallery (all birds) --")
    r1, r3, mAP, n, _ = run(False)
    print(f"  Top-1 = {r1:.3f}   Top-3 = {r3:.3f}   mAP = {mAP:.3f}   (n={n})")
    if possible is not None:
        print("\n-- territorial-gallery constraint --")
        r1c, r3c, mAPc, nc, miss = run(True)
        print(f"  Top-1 = {r1c:.3f}   Top-3 = {r3c:.3f}   mAP = {mAPc:.3f}   (n={nc}, true-id-excluded-by-prior={miss})")


if __name__ == "__main__":
    main()
