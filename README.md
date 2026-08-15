# Diffusion module

A conditional 3D denoising diffusion model for terrain-adaptive Minecraft settlement
generation, trained under a single shared recipe across four datasets, in both binary
(occupancy) and categorical (block2vec embedding) representations. It replaces a
GAN-based generator (vox2vox) (https://github.com/avdstaaij/terrain-adaptive-pcgml-in-minecraft) that was found to suffer from discriminator collapse and
an L1/discrete-embedding mismatch on structurally complex datasets — see
`references.md` for the full account and every paper-attributed method used here.

Background and design rationale: [`thesis_guideline.md`](thesis_guideline.md) ·
[`references.md`](references.md) (every method used here, with its paper and exact
code location) · [`diffusion_code_guide.md`](diffusion_code_guide.md) (a from-scratch
walkthrough of how the code works).

## Setup

Requires Python ≥3.9 and a CUDA-capable GPU (training/generation will run on CPU but is
impractically slow for anything beyond a smoke test).

```bash
python3 -m pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) for exact pinned versions (PyTorch nightly
for current CUDA support, `scipy` for connected-component auxiliary losses, `NBT` /
`anvil-parser` for Minecraft world/region file editing, `wandb` for experiment
tracking).

### Data prerequisites

This module trains on preprocessed ("stacked") paired terrain/settlement datasets and
block2vec embeddings — it does not generate these itself. Each config file's
`dataset_path_binary` / `dataset_path_categorical` / `embedding_path` / `props_path`
fields point at these inputs:

- **Stacked dataset** (`.npy`): a `[N, 2, D, H, W]` array of N paired samples, where
  index 0 along axis 1 is the terrain volume and index 1 is the settlement volume.
  Binary datasets store 0/1 occupancy; categorical datasets store block-ID integers.
- **block2vec embeddings** (`.npy`): a `[vocab_size, emb_dim]` array mapping each
  block ID to its learned embedding vector (`emb_dim=32` throughout this recipe).
- **`properties.json`**: per-dataset metadata (`chunkOffset`, `buildAreaChunkSize`,
  `sampleCounts`) used to convert between sample indices and world coordinates.
- **A source Minecraft world**: used by `generate.py` to read live conditioning
  terrain and as the base world that generated settlements get placed into.

These come from a companion terrain-adaptive-Minecraft-PCG pipeline (dataset
generation, preprocessing, and block2vec training) — this module only consumes their
output, in the formats above.

## File structure

```
.
├── model.py                  3D U-Net denoiser (FiLM, optional attention/self-conditioning)
├── diffusion.py               noise schedule, forward process, DDPM/DDIM sampling
├── train_diffusion.py         training loop, all loss/weighting terms
├── generate.py                unified inference + world-placement + render script
├── numpy_dataset.py            paired-volume PyTorch Dataset (mmap-backed, picklable)
├── configs/                   one YAML per experiment (see "Which config?" below)
├── requirements.txt
├── thesis_guideline.md        thesis structure guide against the IML grading rubric
├── references.md              every paper-attributed method used here, verified against code
└── diffusion_code_guide.md    line-by-line build-it-yourself walkthrough
```

## Training

```bash
python3 train_diffusion.py --config configs/<name>.yaml
```

Useful flags: `--epochs N` overrides the config's epoch count; `--resume path/to/ckpt.pt`
resumes from a checkpoint; `--no-wandb` disables experiment tracking; `--wandb-name`
overrides the run name. Checkpoints are saved every `checkpoint_freq` epochs (and always
at the final epoch) to `<output_dir>/emb<N>_ch<N>_T<N>_lr<...>_bs<N>_ep<N>/`.

### Which config?

There are 44 config files in `configs/` — most are ablations from developing the
recipe, not the recipe itself. **The 8 configs that produced the actual reported
results** are:

| Dataset | Mode | Config |
|---|---|---|
| Wall | categorical | `mikes_angels_wall_cat_smoothweight_w50_minsnr_coarseshape_falsepositive_film_smoothness_w0001_e100.yaml` |
| Wall | binary | `mikes_angels_wall_bin_recipe.yaml` |
| Ring | categorical | `ring_cat_recipe.yaml` |
| Ring | binary | `ring_bin_recipe.yaml` |
| Ring Adaptive | categorical | `ring_adaptive_cat_recipe_fpw001.yaml` |
| Ring Adaptive | binary | `ring_adaptive_bin_recipe.yaml` |
| Mike's Angels | categorical | `mikes_angels_cat_recipe.yaml` |
| Mike's Angels | binary | `mikes_angels_bin_recipe.yaml` |

All 8 share an identical recipe (FiLM conditioning, residual/x0 prediction, Min-SNR-γ
weighting, coarse-shape + false-positive + smoothness auxiliary losses) with **one
documented exception**: `ring_adaptive_cat_recipe_fpw001.yaml` lowers
`false_positive_weight` from the shared default `0.05` to `0.01` after the default
value caused a training-stability collapse on that specific dataset — see the config
file's own header comment and `references.md` for the full account. Every other config
in the directory (self-conditioning, attention, connectivity-boost, alternate
smoothness weights, etc.) is an ablation that was tested but is **not** part of the
recipe above — `references.md`'s Status column tracks exactly which components are
core vs. ablation-only.

## Generating / running inference

```bash
python3 generate.py <dataset> --mode {binary,categorical} \
    --checkpoint path/to/ckpt.pt --output-dir path/to/out \
    --grid-x 24 --grid-z 24 --grid-h 1 --grid-w 1
```

`<dataset>` is the stem of a config in `configs/` (used to load dataset paths and the
props file, independent of `--checkpoint`). `--grid-x`/`--grid-z` are absolute chunk
coordinates for the top-left sample; `--grid-h`/`--grid-w` set the sample grid size
(e.g. `2 2` for a 2×2 block). This writes a copy of the source Minecraft world with the
generated settlement placed into it, and renders it via `mcrender`
(https://github.com/avdstaaij/mcrender — Windows only, or via Wine on Linux) to
`<output-dir>/generate/render.png`.

Other flags worth knowing: `--sampler {ddim,ddpm}` (DDIM is the default, ~20 steps vs.
DDPM's full `T`); `--n-runs N` for a stochastic ensemble (block-union across `N`
independent DDIM runs); `--cat-threshold` and `--freq-weight-exp` control categorical
decode behavior (nearest-embedding distance cutoff and the frequency-weighting
exponent — see `references.md`).

### Reproducing the thesis's renders

The renders in the thesis use a specific, non-default coordinate convention that
`generate.py`'s own built-in `mcrender` call does not apply — its internal bounding box
is 1 block too large in each direction. To reproduce a render exactly as shown in the
thesis, re-render the already-placed world manually with the corrected box:

```bash
mcrender <output-dir>/generate/world <output-dir>/generate/render.png \
    --pos 384 60 384 --pos <384+vol_xz-1> 124 <384+vol_xz-1> --rotation 3 --force
```

where `vol_xz` is 64 for the Wall dataset and 96 for Ring / Ring Adaptive / Mike's
Angels (i.e. `--grid-h`/`--grid-w` × the dataset's per-sample block span, minus 1 for
the far corner). Ground-truth renders (no model involved) use the same far-corner
correction but a `+16`-block-shifted origin (`400 60 400` instead of `384 60 384`) —
see `references.md`'s development history if you need the full derivation.

## Reproducibility notes

- All 8 recipe runs use `seed: 42`.
- Mixed precision (bfloat16), gradient clipping at norm 1.0, and `num_workers: 4` /
  `batch_size >= 4` are set project-wide for GPU utilization; see individual configs
  for exact values.
- Per-run epoch counts differ (see `references.md` / the thesis's Methods chapter) —
  this is a known, documented confound for any binary-vs-categorical comparison, not an
  oversight.
- Training logs are tracked via Weights & Biases (`entity=LSpanring-JKU`,
  `project="Minecraft Diffusion"`); disable with `--no-wandb` if you don't have access
  to that project.
