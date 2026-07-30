# Individual bird re-identification on CHIRP with a frozen DINOv2 backbone

Can you tell one **Siberian jay** from another from a photo, **without reading its colour leg-bands**?
This repo benchmarks a simple appearance-only recipe on the
[CHIRP dataset](https://github.com/alexhang212/CHIRP_Dataset) and finds that a **frozen
[DINOv2](https://github.com/facebookresearch/dinov2) backbone outperforms
[MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384)** (the wildlife re-ID foundation
model CHIRP uses as its appearance baseline) -- and that a lightweight head trained on top of the
*frozen* backbone **matches full fine-tuning** at a fraction of the cost.

> **Scope & honesty.** This is appearance-only individual ID -- **not** band-reading, and it is a
> hard, unsolved problem. CHIRP's colour-ring reader (CORVID) hits Top-1 **0.69**; every appearance
> method here is far below that. The claim is narrow and specific: *among appearance-only methods,
> this recipe is the strongest tested*, and DINOv2 is a better off-the-shelf backbone than
> MegaDescriptor for it. It does **not** claim to solve individual identification from appearance.

## Method

- **Frozen backbone.** DINOv2-giant (ViT-g/14) -- no fine-tuning, just precomputed features. (We also
  tested DINOv3-large/7B and MegaDescriptor; DINOv2-giant wins -- architecture matters more than scale
  or recency here.)
- **Band-masking.** The bird IDs literally *are* their colour-ring codes, so the rings are the label.
  We rasterize CHIRP's per-frame ring segmentation and black it out before embedding, so an
  appearance model can't cheat by reading the bands. (Masking barely moves DINOv2 -- it doesn't lean
  on the rings anyway -- which is the point: the numbers are honest appearance ID, ~0.01 leakage.)
- **Tracklet aggregation.** Max-over-frames per 25-frame tracklet.
- **Optional head.** A small ArcFace projection head (1536->512->128, L2-norm) trained on the labelled
  Train split -- the backbone stays frozen. Costs minutes; matches full backbone fine-tuning.

## Results

CHIRP video re-ID, **disjointed split**, all 25 frames/tracklet, evaluated in an identical pipeline
(same crops, masking, aggregation, gallery protocol). MegaDescriptor was re-run *through this pipeline*
and reproduces its published within-territory Top-1 (0.31 -> 0.326 here), which validates the harness --
so the same-pipeline comparison is apples-to-apples.

**Within-territory gallery** (the primary setting; Top-1 = rank-1):

| method | Top-1 | Top-3 | mAP |
|---|---|---|---|
| MegaDescriptor -- *this pipeline* | 0.326 | 0.440 | 0.434 |
| **DINOv2-giant, frozen** | **0.382** | **0.553** | **0.455** |
| **DINOv2-giant + ArcFace head** *(mean+/-std, 5 seeds)* | **0.420 +/- 0.008** | **0.583 +/- 0.010** | **0.481 +/- 0.004** |
| *reference (CHIRP paper)* MegaDescriptor, pretrained | 0.31 | -- | -- |
| *reference (CHIRP paper)* MegaDescriptor, fine-tuned | 0.41 | -- | -- |
| *reference (CHIRP paper)* CORVID (**reads the bands**) | 0.69 | -- | -- |

And the gap holds across gallery difficulties (Top-1, DINOv2-giant frozen vs MegaDescriptor, this pipeline):

| gallery | MegaDescriptor | DINOv2-giant frozen |
|---|---|---|
| within-territory | 0.326 | **0.382** |
| within-territory + neighbours | 0.131 | **0.191** |
| all birds | 0.052 | **0.072** |

**Takeaways:** (1) same-pipeline, frozen DINOv2-giant beats MegaDescriptor on every metric and every
gallery setting; (2) the frozen-backbone + head recipe (0.420 +/- 0.008) **matches the CHIRP paper's
*fine-tuned* MegaDescriptor (0.41)** without touching the backbone; (3) all appearance methods remain
far below the band-reader -- appearance-only individual corvid ID is not solved.

> Cross-pipeline Top-3 is **not** comparable (the CMC computation differs -- our-pipeline MegaDescriptor
> Top-3 is 0.440, not the paper's 0.62); only the same-pipeline columns are directly comparable.

## Reproduce

1. **Get CHIRP** (CC BY-SA 4.0): [dataset DOI 10.17617/3.GVO4LG](https://doi.org/10.17617/3.GVO4LG).
   You need the `ReID/` directory (`Annotation.csv`, `data/`, per-tracklet `masks_ring.csv`,
   `PossibleBirds_Territory.csv`).
2. **Manifest + file list** for a split:
   ```
   uv run pipeline/chirp_prep.py --annotation ReID/Annotation.csv \
     --split-col DisjointedSetSplit --splits Query,Gallery --frames-per-tracklet 0 \
     --out-manifest manifest.json --out-files files.txt
   ```
3. **Fetch** the frames in `files.txt` locally (paths are relative to `ReID/`).
4. **Embed** (any CUDA or CPU torch):
   ```
   uv run pipeline/chirp_embed.py --manifest manifest.json --root ReID_local \
     --backbone dinov2_giant --mask rings --out emb/
   ```
5. **Evaluate** (Top-1/Top-3/mAP, +/- territorial gallery constraint, open-set `--among`):
   ```
   uv run pipeline/chirp_eval.py --emb emb/ --possible-birds ReID/PossibleBirds_Territory.csv
   ```
6. **(optional) Train the head** on the Train split and re-evaluate:
   ```
   uv run pipeline/chirp_train_head.py --train-emb emb_train/ --apply-emb emb/ --out emb_head/
   ```

Requirements: Python >=3.11, `torch`, `timm`, `numpy`, `pillow`. Each script declares its own deps in
a [uv](https://docs.astral.sh/uv/) inline header, so `uv run pipeline/<script>.py` just works; or
`pip install torch timm numpy pillow` and run with `python`.

## Caveats

- **Appearance-only, and modest** -- see the scope note. This is the weakest-signal regime; the
  band-reader dominates.
- The backbone comparison is off-the-shelf DINOv2-giant vs off-the-shelf MegaDescriptor. The
  *fine-tuned* MegaDescriptor number is from the CHIRP paper (we did not fine-tune anything).
- Numbers are full-frame (all 25/tracklet); the head is reported mean+/-std over 5 seeds. We report the
  **disjointed** split across three gallery settings; the scripts also run the **closed-set** split
  (`--split-col ClosedSetSplit`) and **open-set** retrieval (`chirp_eval --among`).

## Attribution & license

- **Data:** CHIRP dataset -- Chan, Singhal, Kocahan, Meltzer, Lubrano, Warrington, Griesser, Kano &
  Naik, CVPR 2026. [Dataset](https://github.com/alexhang212/CHIRP_Dataset) (CC BY-SA 4.0). Please cite
  the authors if you use the data.
- **Models:** [DINOv2](https://github.com/facebookresearch/dinov2) (Meta),
  [MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384) (BVRA), loaded via
  [`timm`](https://github.com/huggingface/pytorch-image-models).
- **This code:** MIT -- see [LICENSE](LICENSE).
