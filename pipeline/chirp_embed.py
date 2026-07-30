# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "torchvision", "timm", "pillow"]
# ///
"""Embed CHIRP re-ID frames with a frozen backbone, optionally BAND-MASKING the colour leg rings.

The rings ARE the identity label (bird ids are ring codes), so an appearance model must not see
them or it just reads the label. We rasterize the per-frame ring polygons from masks_ring.csv and
black them out before embedding, so an appearance model can't cheat by reading the label. `--mask
none` gives the leakage baseline for the ablation.

Requires timm + torch; runs on CUDA or CPU. Backbones: dinov2_giant / dinov3_large / dinov3_7b /
megadescriptor (or any timm model name). Output: <out>/frame_emb.npy [N,d] (L2-normed) +
frame_meta.json [{unqtrack, id, split, img}], aligned.
"""
import argparse
import ast
import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw

import timm


ALIASES = {
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_giant": "vit_giant_patch14_dinov2.lvd142m",
    "dinov3_large": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_7b": "vit_7b_patch16_dinov3.lvd1689m",
}


def build(backbone, device):
    if backbone == "megadescriptor":
        m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        tf = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    else:
        name = ALIASES.get(backbone, backbone)
        m = timm.create_model(name, pretrained=True, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=m)
        tf = timm.data.create_transform(**cfg)
    return m.eval().to(device), tf


def _contours(obj):
    """yield flat [x0,y0,x1,y1,...] coordinate lists from CHIRP's nested polygon encoding."""
    if obj and isinstance(obj[0], (int, float)):
        yield obj
    else:
        for x in obj:
            yield from _contours(x)


class PolyMasker:
    """Lazily loads a per-video-territory polygon-mask CSV and fills either INSIDE the polygons
    (rings -> hide the identity label) or OUTSIDE them (invert=True -> background removal), with
    `fill` (black or gray). Background removal + gray fill matches datasets whose crops are
    already segmented (bird-on-gray)."""
    def __init__(self, root, csv_name, class_prefix, fill, invert=False):
        self.root, self.csv_name, self.cls = Path(root), csv_name, class_prefix
        self.fill, self.invert = tuple(fill), invert
        self.cache = {}
        self.hit = self.miss = 0

    def _table(self, img_rel):
        csv_rel = str(PurePosixPath(img_rel).parents[1] / self.csv_name)
        if csv_rel not in self.cache:
            d = {}
            p = self.root / csv_rel
            if p.exists():
                with open(p) as f:
                    for r in csv.DictReader(f):
                        if r.get("Class", "").startswith(self.cls):
                            d.setdefault(r["img"], []).append(r["mask"])
            self.cache[csv_rel] = d
        return self.cache[csv_rel]

    def apply(self, im, img_rel):
        masks = self._table(img_rel).get(img_rel, [])
        if not masks:
            self.miss += 1
            return im            # no polygon -> leave frame unchanged (a coverage gap)
        self.hit += 1
        m = Image.new("L", im.size, 0)
        draw = ImageDraw.Draw(m)
        for ms in masks:
            try:
                poly = ast.literal_eval(ms)
            except (ValueError, SyntaxError):
                continue
            for flat in _contours(poly):
                if len(flat) >= 6:
                    draw.polygon([(flat[i], flat[i+1]) for i in range(0, len(flat) - 1, 2)], fill=255)
        arr = np.array(im)
        sel = (np.array(m) == 0) if self.invert else (np.array(m) > 0)
        arr[sel] = np.array(self.fill, arr.dtype)
        return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True, help="local mirror root (dir containing data/...)")
    ap.add_argument("--backbone", default="dinov2_giant")
    ap.add_argument("--mask", choices=["rings", "none"], default="rings")
    ap.add_argument("--remove-bg", action="store_true",
                    help="remove background (keep only bird via masks.csv) to match segmented crops")
    ap.add_argument("--fill", choices=["black", "gray"], default="black",
                    help="fill colour for masked/removed regions (gray=128 matches segmented crops)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    root = Path(args.root)
    fill = (0, 0, 0) if args.fill == "black" else (128, 128, 128)
    ring_masker = PolyMasker(root, "masks_ring.csv", "Ring", fill) if args.mask == "rings" else None
    bg_remover = PolyMasker(root, "masks.csv", "bird", fill, invert=True) if args.remove_bg else None
    print(f"{len(manifest)} frames | backbone={args.backbone} | mask={args.mask} | "
          f"remove_bg={args.remove_bg} | fill={args.fill}", flush=True)

    model, tf = build(args.backbone, args.device)
    if args.half:
        model = model.to(torch.bfloat16)

    def load(rec):
        im = Image.open(root / rec["img"]).convert("RGB")
        if bg_remover is not None:
            im = bg_remover.apply(im, rec["img"])       # remove background first
        if ring_masker is not None:
            im = ring_masker.apply(im, rec["img"])      # then hide the ring label
        return tf(im)

    embs = []
    for i in range(0, len(manifest), args.batch):
        chunk = manifest[i:i + args.batch]
        batch = torch.stack([load(r) for r in chunk]).to(args.device)
        if args.half:
            batch = batch.to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(batch)
        embs.append(F.normalize(z.float(), dim=1).cpu().numpy())
        if i % (args.batch * 25) == 0:
            print(f"  {i}/{len(manifest)}", flush=True)
    embs = np.concatenate(embs)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frame_emb.npy", embs)
    meta = [{"unqtrack": r["unqtrack"], "id": r["id"], "split": r["split"],
             "video": r["video"], "territory": r["territory"], "img": r["img"]}
            for r in manifest]
    json.dump(meta, open(out / "frame_meta.json", "w"))
    if ring_masker is not None:
        print(f"ring-mask coverage: {ring_masker.hit}/{ring_masker.hit + ring_masker.miss} frames had ≥1 ring", flush=True)
    if bg_remover is not None:
        print(f"bg-removal coverage: {bg_remover.hit}/{bg_remover.hit + bg_remover.miss} frames had a bird mask", flush=True)
    print(f"saved {embs.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
