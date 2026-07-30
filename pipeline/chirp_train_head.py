# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""Train a supervised re-ID projection head (ArcFace) on FROZEN CHIRP frame embeddings, then apply
it to a query/gallery embedding set so pipeline/chirp_eval.py can score the trained metric.

The backbone stays frozen (embeddings are precomputed by chirp_embed.py); we only learn a small
head 1536->512->128 (L2-norm) with an additive-angular-margin (ArcFace) classifier over the real
bird ids from the Train split. This is the core method-validation: with genuine identity labels
(from the colour bands, which the frames are masked to hide), can DINOv2/3 + a head actually
separate individuals -- and beat frozen + MegaDescriptor's ~0.28?

Requires torch; runs on CUDA or CPU. Frame-level training (each frame carries its tracklet's id);
eval-side aggregation stays in chirp_eval.py.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(d):
    emb = np.load(Path(d) / "frame_emb.npy").astype("float32")
    meta = json.load(open(Path(d) / "frame_meta.json"))
    return emb, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-emb", required=True, help="frozen frame embeddings of the Train split")
    ap.add_argument("--apply-emb", required=True, help="frozen Query+Gallery embeddings to transform")
    ap.add_argument("--out", required=True, help="dir for transformed embeddings (+ head.pt)")
    ap.add_argument("--dim-hidden", type=int, default=512)
    ap.add_argument("--dim-out", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--arc-margin", type=float, default=0.3)
    ap.add_argument("--arc-scale", type=float, default=30.0)
    ap.add_argument("--max-tracklets-per-id", type=int, default=0,
                    help="cap training tracklets per identity (samples-per-individual scaling curve; 0=all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    Xtr, mtr = load(args.train_emb)
    if args.max_tracklets_per_id:
        from collections import defaultdict
        tr_by_id, seen = defaultdict(list), set()
        for m in mtr:
            k = (m["id"], m["unqtrack"])
            if k not in seen:
                seen.add(k); tr_by_id[m["id"]].append(m["unqtrack"])
        keep = set()
        for cid, trs in tr_by_id.items():
            keep.update(sorted(trs)[:args.max_tracklets_per_id])
        idxs = [i for i, m in enumerate(mtr) if m["unqtrack"] in keep]
        Xtr, mtr = Xtr[idxs], [mtr[i] for i in idxs]
        print(f"subsampled to <={args.max_tracklets_per_id} tracklets/id: {len(Xtr)} frames", flush=True)
    ids = sorted({m["id"] for m in mtr})
    id2i = {c: i for i, c in enumerate(ids)}
    ytr = np.array([id2i[m["id"]] for m in mtr])
    print(f"train frames={len(Xtr)}  identities={len(ids)}  in_dim={Xtr.shape[1]}", flush=True)

    head = nn.Sequential(
        nn.Linear(Xtr.shape[1], args.dim_hidden), nn.BatchNorm1d(args.dim_hidden), nn.GELU(),
        nn.Linear(args.dim_hidden, args.dim_out),
    ).to(dev)
    W = nn.Parameter(torch.randn(len(ids), args.dim_out, device=dev))  # ArcFace class prototypes
    opt = torch.optim.Adam(list(head.parameters()) + [W], lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr).to(dev)
    m, s = args.arc_margin, args.arc_scale
    n = len(Xt)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            z = nn.functional.normalize(head(Xt[idx]), dim=1)
            Wn = nn.functional.normalize(W, dim=1)
            cos = z @ Wn.t()                                   # cosine logits
            theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            target = torch.zeros_like(cos)
            target.scatter_(1, yt[idx].unsqueeze(1), 1.0)
            logits = s * torch.cos(theta + m * target)          # add margin on the true class only
            loss = ce(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  arcface loss {tot / n:.4f}", flush=True)

    # apply to query/gallery frozen embeddings -> transformed dir chirp_eval can read directly
    Xap, map_ = load(args.apply_emb)
    head.eval()
    with torch.no_grad():
        Z = nn.functional.normalize(head(torch.from_numpy(Xap).to(dev)), dim=1).cpu().numpy()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frame_emb.npy", Z)
    json.dump(map_, open(out / "frame_meta.json", "w"))
    torch.save(head.state_dict(), out / "head.pt")
    print(f"saved transformed {Z.shape} + head.pt -> {out}", flush=True)


if __name__ == "__main__":
    main()
