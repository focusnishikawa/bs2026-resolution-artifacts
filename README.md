# bs2026-resolution — reproducibility artifacts

Data manifests, aggregated results and analysis code for the study
**"Which input resolution gives high identification accuracy at the shortest processing time,
and what does the evaluation setup change about that answer?"**
(14 input resolutions x 6 architectures x 2 training regimes x 30 seeds, with on-device
measurements on an NVIDIA Jetson Orin Nano 8 GB).

The study trains 2,520 checkpoints, produces 5,040 evaluation conditions and measures
56 Orin configurations plus 126 parts of the split ViT-L chains.

**The images themselves are not redistributed** (see [Data licensing](#data-licensing)).
What is published here is enough to rebuild the exact image set: per-image source ID,
source URL, SHA-256, group ID and the split membership under both splits.

---

## 1. Contents

| Path | What it is |
|---|---|
| `data/manifest.csv` | **12,519 rows, one per image.** Source, ID, URL, SHA-256, group ID (two rules) and membership in both splits |
| `data/splits/{train,val,test}.csv` | Image-wise stratified split used for the main experiments (8,761 / 1,876 / 1,882) |
| `data/splits_group/{train,val,test}.csv` | Group-disjoint split used for the leakage check (8,763 / 1,880 / 1,876) |
| `data/wikimedia_crawl_urls.json` | Wikimedia Commons URLs in acquisition order, per species (see [Sources](#3-image-sources)) |
| `results/*.json` | Every aggregate the paper's tables and figures are built from (see [Results](#5-results-files)) |
| `train/*.py`, `train/*.sh` | Training, evaluation, aggregation, statistics and figure generation |
| `edge_scripts/*` | Orin-side scripts: ONNX export, graph splitting, TensorRT builds, latency and accuracy measurement |

Per-image predictions (6 logits + argmax for every image, for all seeds and configurations;
4,172 CSV files, 490 MB uncompressed) are **not in this repository** — they are attached to the
Zenodo record as `predictions_per_image.tar.gz`.

## 2. Dataset

Six-way identification of Japanese raptors. 12,519 images.

| Key | Common name | Scientific name |
|---|---|---|
| `Inuwashi` | Golden eagle | *Aquila chrysaetos* |
| `Ojirowashi` | White-tailed eagle | *Haliaeetus albicilla* |
| `Oowashi` | Steller's sea eagle | *Haliaeetus pelagicus* |
| `Kumataka` | Mountain hawk-eagle | *Nisaetus nipalensis* |
| `Ootaka` | Northern (Eurasian) goshawk | *Astur gentilis* (syn. *Accipiter gentilis*) |
| `Chuhi` | Eastern marsh harrier | *Circus spilonotus* |

Splits are stratified by species. The image-wise split is the one used for all headline
results; the group-disjoint split is used only for the leakage check in the limitations
section.

## 3. Image sources

`source_kind` in the manifest, with the URL pattern used to reach the original:

| `source_kind` | Images | Source and URL |
|---|---:|---|
| `inaturalist` | 4,865 | iNaturalist observation: `https://www.inaturalist.org/observations/<source_id>` |
| `own_photo` | 4,578 | Photographed by the authors; crops of source photographs. `source_id` is the source-photograph (camera file) ID. Not redistributed |
| `own_video` | 1,000 | Frames the authors extracted from their own video. `source_id` is the 8-hex video ID; **33 videos** |
| `legacy_internal` | 846 | Earlier internal collections (`v5_ds_beta_bird` 496, `v5_train_v5_inat` 120, `v4_gt` 50, `inat_3113/3119/3160` 150, `v5_gt_critical` 1). Serial or UUID file names; the original URL is not recoverable |
| `wikimedia` | 746 | Wikimedia Commons: `https://commons.wikimedia.org/`. See below |
| `own_video_noid` | 339 | Frames from the authors' own video whose file names carry no video ID |
| `birds525` | 88 | BIRDS 525 species dataset: `https://www.kaggle.com/datasets/gpiosenka/100-bird-species` |
| `macaulay` | 57 | Macaulay Library asset: `https://macaulaylibrary.org/asset/<source_id>` |

**5,756 of 12,519 images (46.0%) carry a directly resolvable source URL**; SHA-256 is given
for all 12,519.

**Wikimedia.** File names are `wm_<first 12 hex of the SHA-1 of the downloaded bytes>.jpg`, so
the mapping from URL to file is reproduced by fetching the URLs in
`data/wikimedia_crawl_urls.json` and hashing the bytes. That log records 4,470 URLs visited
across four species; 746 images survived de-duplication, a minimum-dimension filter (300 px)
and the later dataset assembly.

## 4. Group definitions and residual leakage

Groups are inferred from file names. Two rules are given in the manifest.

- `group_id` (**v1**) — the rule used in the experiments (`train/make_group_split.py`):
  crops of one source photograph (camera ID), one iNaturalist observation, one Macaulay
  asset, all video frames of a species that carry no video ID, otherwise one file per group.
- `group_id_v2` — **v1 plus the 8-hex video ID** carried by the 1,000 frames in
  `bird_frames_pure_oowashi_20260504/`. Under v1 those frames matched no video pattern and
  became one group each.

Images belonging to a group that spans more than one split (the definition used in the paper):

| Split | Rule | Straddling groups | test | val |
|---|---|---:|---:|---:|
| image-wise | v1 | 24 | 61 (3.24%) | 58 (3.09%) |
| image-wise | v2 | 57 | 207 (11.00%) | 212 (11.30%) |
| group-disjoint | v1 | **0** | 0 | 0 |
| group-disjoint | v2 | 33 | 162 (8.64%) | 161 (8.56%) |

So the group-disjoint split removes every straddle that rule v1 can see, but **does not
separate the 33 videos**, whose frames are near-duplicates: within one video the median gap
between consecutive retained frames is 3.34 s and 278 of 967 consecutive pairs are within 1 s.
Rebuilding the split under v2 and re-running the 16 boundary configurations over 30 seeds is
left as future work; the numbers above are stated in the paper's limitations section.

`results/data_provenance.json` holds all of this, plus per-species and per-split counts by
source, under two counting conventions (`in_straddling_group_*`, the paper's definition, and
`sharing_train_*`, which counts only groups that also contain training images).

## 5. Results files

| File | Contents |
|---|---|
| `summary_v2.json` | Server FP32 test accuracy, all 84 configurations x 30 seeds, both regimes, with `per_seed` |
| `summary_val.json` | The same on validation (used for selection) |
| `summary_deploy.json` / `summary_deploy_val.json` | Accuracy of the engine actually deployed on the Orin (test / validation) |
| `summary_group.json` | Boundary 16 configurations x 30 seeds under the group-disjoint split |
| `final_tables*.json` | Latency, preprocessing and derived tables (`_trt10_maxn` is the current environment) |
| `optimal_n*.json`, `target_sweep.json` | Configuration selection under time budgets and accuracy targets |
| `seed_stability.json` | Sign agreement over k-seed subsets |
| `uncertainty.json` | Seed standard deviations and group bootstrap intervals |
| `class_metrics.json` | Per-class recall and macro-F1 for the central configurations |
| `wallclock_trt10.json` | Continuous-execution wall-clock of the deployed chains |
| `data_provenance.json` | Source, split and group statistics (Sections 3 and 4) |

## 6. Models and training

| Key | Identifier | Pretrained weights |
|---|---|---|
| `mnv4` | `mobilenetv4_conv_medium.e500_r256_in1k` (timm) | ImageNet-1k |
| `effb0` | `efficientnet_b0.ra_in1k` (timm) | ImageNet-1k |
| `resnet50` | `torchvision:resnet50:IMAGENET1K_V2` | ImageNet-1k V2 |
| `vit_small` | `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm) | IN-21k, fine-tuned on IN-1k |
| `dinov2_l` | `vit_large_patch14_dinov2` (timm) | `dinov2_vitl14_pretrain.pth` |
| `dinov3_l` | `vit_large_patch16_dinov3` (timm) | `dinov3_vitl16_lvd1689m.safetensors` |

AdamW, weight decay 0.05, label smoothing 0.1. Separate backbone / head learning rates:
5e-4 / 5e-3 (CNNs), 1e-4 / 1e-3 (ViT-S/16), 5e-5 / 5e-4 (ViT-L-scale). Epoch budget 160 with
batch 256 for the CNNs and ViT-S/16, and 60 with batch 64 and layer-wise decay 0.75 for the
ViT-L-scale models. The weights of the best validation epoch are kept, with early stopping
after 20 (CNN), 25 (ViT-S/16) or 10 (ViT-L) epochs without improvement.

**Seeds 42-71 (30 seeds)** vary initialisation, data order and augmentation sampling.
**The data split is held fixed across seeds.** Resolutions are N = 16, 32, ..., 224 in steps
of 16 (14 levels). Regime A evaluates the N=224 checkpoint of each model and seed at every
resolution; regime B evaluates the checkpoint trained at that resolution.

Resolution note: DINOv2-L is patch-14, so the requested N is rounded to a multiple of 14 and
the manifest's tables report `N'`, the side length actually fed to the model. DINOv3-L is
patch-16 and `N' = N`.

## 7. Measurement setup

**Jetson Orin Nano 8 GB**, JetPack 6.2 (L4T R36.4.3), TensorRT 10.3.0, power mode
`MAXN_SUPER` (CPU 1.73 GHz, GPU 1,020 MHz). Every configuration is the **median of 30 runs**;
one run is `trtexec --warmUp=2000 --iterations=300 --avgRuns=100` from a fresh process. The
split ViT-L chains are measured per part with `--iterations=200 --avgRuns=50` and reported as
the sum of the per-part medians over 126 parts; `wallclock_trt10.json` additionally reports
the continuous execution of whole chains.

Precision as deployed: FP16 for the CNNs and ViT-S/16; DINOv2-L mixed (FP32 on the last part
only, because outlier activations exceed the FP16 range); DINOv3-L FP32 throughout. Server
measurements use an NVIDIA H200 with TensorRT 10.7 and are medians of three runs.

**Verification applied to every engine** (a build that silently degenerates can be faster than
a correct one): all 1,882 test images are run through it and the number of distinct outputs is
checked (a degenerate engine returns one), argmax agreement with server FP32 is recorded, and
the task metric is recomputed on the device rather than inferred from agreement.

## 8. Reproducing

```bash
# 1. Rebuild the image set from the manifest (iNaturalist, Macaulay, Wikimedia, BIRDS 525).
#    Verify each file against the SHA-256 column; 6,763 images have no public URL.
# 2. Regenerate the provenance statistics
python3 train/make_data_manifest.py --sha256 <sha256sum output> \
    --index <raptor_subset_index.csv> --splits data/splits --splits-group data/splits_group

# 3. Training / evaluation (one model x resolution x seed)
python3 train/train_res.py --model resnet50 --res 112 --seed 42

# 4. Aggregation, selection and figures
python3 train/analyze_all.py --src trt10-maxn
python3 train/optimal_n.py   --src trt10-maxn
python3 train/make_figs.py
```

Numbers in the paper are produced from the JSON files in `results/`; the scripts above
regenerate those JSONs from raw measurements.

## Data licensing

The code in this repository is MIT-licensed (`LICENSE`).

**No images are redistributed.** iNaturalist photographs carry per-photograph licences chosen
by their observers, Macaulay Library assets may not be redistributed, Wikimedia Commons files
carry their own licences and attribution requirements, and the authors' own photographs and
video frames are not released here. The manifest publishes identifiers, URLs and hashes so
that the same image set can be reassembled from the original sources under their own terms.

`data/manifest.csv` and the split CSVs are released under CC BY 4.0.

## Citation

The paper is under submission; this section will be updated with the DOI on publication.
