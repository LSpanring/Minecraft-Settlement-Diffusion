"""
Train a conditional diffusion model for volume-to-volume Minecraft generation.

Usage: python train_diffusion.py --config config.yaml
       python train_diffusion.py --config config.yaml --resume path/to/ckpt.pt --epochs 200
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

from numpy_dataset import NumpyPairDataset

from model import ConditionalUNet3D
from diffusion import GaussianDiffusion


def compute_coarse_shape_loss(x0_pred: torch.Tensor, x0: torch.Tensor, downsample_factor: int = 4) -> torch.Tensor:
    """
    Auxiliary loss encouraging correct coarse/global shape (simplified from LAS-Diffusion's
    coarse-then-fine idea, Zheng et al. 2023, arxiv.org/abs/2305.04461 -- the full method uses a
    separate low-res model stage; this adds the same "get the global shape right" pressure as an
    extra loss term on the existing single model instead). Downsamples both the true and
    predicted x0 and penalizes disagreement at that coarse resolution, on top of the usual
    per-voxel loss -- so getting the overall structure shape right matters, not just per-voxel
    material choice.

    Called twice with different scales (see coarse_shape_weight / coarse_shape_weight_global):
    the original downsample_factor=4 checks local density in ~16x16x16-voxel neighborhoods, which
    doesn't push on whole-volume coordination -- a closed wall loop needs the far side of the
    circle to "agree" with the near side, a property no local neighborhood can see on its own. A
    much larger downsample_factor (e.g. 16, giving a 4x4x4 coarse grid close to the whole volume)
    adds that missing global-coordination pressure as a second, independent term.

    :param x0_pred: Predicted clean sample, shape (B, emb_dim, D, H, W).
    :param x0: True clean sample, same shape.
    :param downsample_factor: Pooling kernel size; must evenly divide D, H, W.
    :return: Scalar MSE loss at the downsampled resolution.
    """
    coarse_pred = F.avg_pool3d(x0_pred, kernel_size=downsample_factor)
    coarse_true = F.avg_pool3d(x0, kernel_size=downsample_factor)
    return F.mse_loss(coarse_pred, coarse_true)


def compute_false_positive_loss(x0_pred: torch.Tensor, x0: torch.Tensor, margin: float = 1e-6) -> torch.Tensor:
    """
    Auxiliary loss penalizing predicted structure-presence at voxels where the TRUE target has
    none. generation_weight (compute_smooth_weight_map) upweights getting voxels right *near*
    true structure (up to generation_weight, e.g. 50x) but false positives everywhere else --
    the ~98% of the volume that should stay empty -- only get the flat baseline weight (1x) on
    the noise-prediction MSE. Empirically that asymmetry isn't enough: a checkpoint with
    coarse_shape_loss (so aggregate density roughly matches) still paints a diffuse mass of
    material across nearly the entire ground surface rather than a concentrated shape --
    confirmed by zooming into the render, not just aggregate counts. Penalizes structure-presence
    (x0_pred magnitude, not noise-prediction MSE) directly, gated on the true x0 -- unlike a
    fixed geometric prior (e.g. a height threshold), this only fires on genuine mistakes, so it
    generalizes to any dataset without a dataset-specific assumption baked in.

    :param x0_pred: Predicted clean sample, shape (B, emb_dim, D, H, W).
    :param x0: True clean sample, same shape.
    :param margin: L2 norm below which a true voxel counts as "empty".
    :return: Scalar loss -- mean squared structure-presence at truly-empty voxels.
    """
    is_true_empty = x0.norm(dim=1) <= margin  # [B, D, H, W]
    structure_signal = x0_pred.pow(2).sum(dim=1)  # [B, D, H, W], differentiable "is structure" proxy
    return (structure_signal * is_true_empty.float()).mean()


def compute_smoothness_loss(x0_pred: torch.Tensor) -> torch.Tensor:
    """
    Total-variation-style loss penalizing voxel-to-voxel discontinuity in predicted structure-
    presence, to discourage scattered single-voxel noise in favor of fewer, larger, connected
    regions. Motivated by direct measurement: a checkpoint with grounding/false-positive/FiLM all
    fixed floating (11%) and over-generation (density near the true baseline), yet connected-
    component analysis of its OWN output still showed 293 components with a median size of 1.0
    voxel -- the model's per-voxel decisions are locally reasonable but globally fragmented, not
    forming a single coherent wall. connectivity_boost doesn't address this: it upweights loss on
    thin/small components of the TRUE target (an under-learning/recall problem), whereas this is
    about the model's OWN hallucinated fragments (a precision/coherence problem) -- a different
    failure mode needing a different fix.

    An isolated single active voxel has 6 face-neighbor boundaries in 3D, each contributing to
    this loss; the same amount of material arranged as one coherent blob has a far better
    surface-to-volume ratio and a much smaller total penalty. Minimizing this trades many small
    fragments for fewer, larger, connected ones -- directly optimizing for what connected-
    component analysis measures, without needing scipy.ndimage.label (not differentiable) in the
    loop at all.

    :param x0_pred: Predicted clean sample, shape (B, emb_dim, D, H, W).
    :return: Scalar loss -- mean absolute difference between axis-adjacent voxels' structure signal.
    """
    structure_signal = x0_pred.pow(2).sum(dim=1, keepdim=True)  # [B, 1, D, H, W]
    diff_d = (structure_signal[:, :, 1:] - structure_signal[:, :, :-1]).abs().mean()
    diff_h = (structure_signal[:, :, :, 1:] - structure_signal[:, :, :, :-1]).abs().mean()
    diff_w = (structure_signal[:, :, :, :, 1:] - structure_signal[:, :, :, :, :-1]).abs().mean()
    return diff_d + diff_h + diff_w


def compute_connectivity_weight(x0: torch.Tensor, boost: float = 5.0) -> torch.Tensor:
    """
    Topology-inspired auxiliary weight (simplified proxy for persistent-homology losses like
    Byrne et al. 2020 / TopoDiffusionNet, which need specialized libraries not available here).
    Upweights voxels belonging to thin/small connected components of the TRUE structure relative
    to its largest component (connected-component labeling, not differentiable -- computed once
    on the fixed target, not the live model output). Thin connecting segments have less
    surrounding "obviously structure" context to lean on and are exactly what determines whether
    a generated wall reads as one continuous barrier or a set of disconnected fragments.

    :param x0: True clean sample, shape (B, emb_dim, D, H, W).
    :param boost: Weight multiplier applied to voxels in non-largest components.
    :return: Weight map, shape (B, 1, D, H, W), 1.0 outside boosted regions.
    """
    from scipy import ndimage
    is_structure = (x0.norm(dim=1) > 1e-6).cpu().numpy()  # [B, D, H, W]
    weight = np.ones_like(is_structure, dtype=np.float32)
    for b in range(is_structure.shape[0]):
        labeled, n = ndimage.label(is_structure[b])
        if n <= 1:
            continue
        sizes = ndimage.sum(is_structure[b], labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        for comp_id in range(1, n + 1):
            if comp_id != largest:
                weight[b][labeled == comp_id] = boost
    return torch.from_numpy(weight).unsqueeze(1).to(x0.device)


def _binary_transform(x: torch.Tensor) -> torch.Tensor:
    return (x.float() * 2 - 1).unsqueeze_(0)


class _EmbeddingTransform:
    """Picklable transform that maps block IDs to embeddings — required for num_workers > 0 on Windows."""
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.moveaxis(self.embeddings[x], -1, 0).to(dtype=torch.float32)


def compute_smooth_weight_map(changed: torch.Tensor, generation_weight: float,
                               decay: float = 0.5, max_shells: int = 4) -> torch.Tensor:
    """
    Distance-graduated version of the binary generation_weight mask (DenseWeight-style,
    Steininger et al. 2021). The binary mask applies generation_weight right up to the
    structure's edge and 1.0 everywhere else -- a hard cliff that forced a tradeoff
    between too-low (scattered fragments) and too-high (blobs covering everything).
    Here, weight decays smoothly outward from the structure in voxel shells, so nearby
    voxels get graded partial credit instead of a sudden drop to baseline.

    :param changed: Binary structure mask, shape (B, 1, D, H, W).
    :param generation_weight: Peak weight applied directly on structure voxels.
    :param decay: Multiplicative falloff applied to (weight - 1) per shell out from the structure.
    :param max_shells: Number of dilation shells before falling back to baseline weight 1.0.
    :return: Smooth weight map, same shape as `changed`.
    """
    changed = changed.float()
    weight_map = torch.where(changed > 0, changed.new_full((), float(generation_weight)), changed.new_ones(()))

    frontier = changed
    current_weight = float(generation_weight)
    for _ in range(max_shells):
        dilated = F.max_pool3d(frontier, kernel_size=3, stride=1, padding=1)
        new_shell = (dilated > 0) & (frontier == 0)
        current_weight = 1.0 + (current_weight - 1.0) * decay
        weight_map = torch.where(new_shell, weight_map.new_full((), current_weight), weight_map)
        frontier = dilated

    return weight_map


def compute_min_snr_weight(diffusion, t: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Min-SNR-gamma loss weighting (Hang et al. 2023, https://arxiv.org/abs/2303.09556).
    Diffusion training implicitly treats each timestep as a separate task; low-noise timesteps
    (SNR(t) -> large as t->0) dominate the effective x0-reconstruction objective unless capped.
    Clamping SNR at gamma reins in that over-weighting while leaving high-noise timesteps
    (already below gamma) at their normal weight. Depends only on the noise schedule, not on
    dataset content -- unlike generation_weight, this generalizes to any dataset unchanged.

    :param diffusion: GaussianDiffusion instance (for alphas_cumprod).
    :param t: Timesteps, shape (B,).
    :param gamma: SNR clamp ceiling (paper default: 5.0).
    :return: Per-sample weight, shape (B, 1, 1, 1, 1) for broadcasting over a 5D loss tensor.
    """
    alpha_t = diffusion.alphas_cumprod[t]
    snr = alpha_t / (1.0 - alpha_t)
    weight = snr.clamp(max=gamma) / snr
    return weight.view(-1, 1, 1, 1, 1)


def hyperparam_folder(cfg: dict) -> str:
    lr = cfg.get("lr", 1e-4)
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return (
        f"emb{cfg.get('emb_dim', 32)}"
        f"_ch{cfg.get('base_ch', 32)}"
        f"_T{cfg.get('T', 100)}"
        f"_lr{lr_str}"
        f"_bs{cfg.get('batch_size', 1)}"
        f"_ep{cfg.get('epochs', 100)}"
    )

def train(cfg: dict, use_wandb: bool = True, wandb_name=None, resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # cuDNN autotunes the fastest conv algorithm for a given input size on first use, then
    # reuses it -- free speedup here since every batch in a run has the same fixed volume shape.
    torch.backends.cudnn.benchmark = True

    # Data
    mode = cfg.get("mode", "binary")
    if mode == "binary":
        cfg["emb_dim"] = 1  # binary: single channel per volume; overrides config
        dataset_path = cfg["dataset_path_binary"]
        dataset = NumpyPairDataset(dataset_path, targetDtype=None, useMmap=True, transform=_binary_transform)
    else:
        dataset_path = cfg["dataset_path_categorical"]
        embeddings = torch.from_numpy(np.load(cfg["embedding_path"]))
        # Normalize embeddings to [-1, 1] to match the cosine diffusion schedule's expected signal range.
        # Raw block2vec embeddings have values in ~[-3.6, 5.0] (mean norm ~3.7), which breaks the
        # noise schedule and causes the DDIM x0 clamp to truncate valid embeddings.
        emb_scale = float(embeddings.abs().max())
        embeddings = embeddings / emb_scale
        cfg["emb_scale"] = emb_scale  # persisted in checkpoint for generation
        dataset = NumpyPairDataset(
            dataset_path, targetDtype=np.int32, useMmap=True,
            transform1=_EmbeddingTransform(embeddings),
            transform2=_EmbeddingTransform(embeddings),
        )

    num_workers = cfg.get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )

    # Model and diffusion
    # All of these operate on x0's (B, emb_dim, D, H, W) shape generically -- emb_dim=1 for
    # binary, 32 for categorical -- so none of them are actually categorical-specific. Earlier
    # they were gated to categorical-only as a scoping artifact of developing them wall-first;
    # unrestricted here so binary datasets get the same recipe.
    self_conditioning = cfg.get("self_conditioning", False)
    coarse_shape_weight = cfg.get("coarse_shape_weight", 0.0)
    coarse_shape_weight_global = cfg.get("coarse_shape_weight_global", 0.0)
    coarse_shape_downsample_global = cfg.get("coarse_shape_downsample_global", 16)
    connectivity_boost = cfg.get("connectivity_boost", 0.0)
    false_positive_weight = cfg.get("false_positive_weight", 0.0)
    smoothness_weight = cfg.get("smoothness_weight", 0.0)
    model = ConditionalUNet3D(
        emb_dim=cfg.get("emb_dim", 32),
        base_ch=cfg.get("base_ch", 32),
        time_dim=cfg.get("time_dim", 128),
        attention=cfg.get("attention", False),
        self_conditioning=self_conditioning,
        film=cfg.get("film", False),
    ).to(device)

    diffusion = GaussianDiffusion(T=cfg.get("T", 100), device=str(device))

    # AdamW decouples weight decay from the adaptive gradient step (Adam applies it coupled,
    # which isn't the correct L2 penalty once gradients are rescaled). Opt-in since it changes
    # optimization dynamics -- existing configs keep plain Adam unless they ask for this.
    optimizer_cls = torch.optim.AdamW if cfg.get("use_adamw", False) else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # Mixed precision: on a compute-bound, tensor-core GPU this is usually the single biggest
    # wall-clock win available without changing the model. Default on; cfg can disable it.
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    # GradScaler is only needed for fp16 (bf16 has enough dynamic range to skip it). Disabled,
    # it's a no-op passthrough, so the backward/step code below stays identical either way.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # Cosine LR annealing: smoothly decays LR to lr_min over the run instead of holding it
    # constant, which typically improves late-training convergence. Opt-in via cfg since it
    # changes optimization dynamics -- existing configs keep a constant LR unless they ask.
    scheduler = None
    if cfg.get("lr_schedule") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("epochs", 100), eta_min=cfg.get("lr_min", 1e-6)
        )

    # Resume from checkpoint if provided
    start_epoch = 0
    resumed_wandb_id = None
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        resumed_wandb_id = ckpt.get("wandb_run_id")
        print(f"Resumed from {resume_path} (epoch {ckpt['epoch']} -> continuing from {start_epoch})")

    # Training loop
    out_dir = Path(cfg["output_dir"]) / hyperparam_folder(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        _log_keys = {"dataset_path_binary", "dataset_path_categorical",
                     "embedding_path", "output_dir", "props_path",
                     "wandb_project", "wandb_entity"}
        run_name = wandb_name or cfg.get("wandb_run_name") or hyperparam_folder(cfg)
        wandb.init(
            entity=cfg.get("wandb_entity", "LSpanring-JKU"),
            project=cfg.get("wandb_project", "Minecraft Diffusion"),
            name=run_name,
            config={k: v for k, v in cfg.items() if k not in _log_keys},
            dir=str(out_dir),
            id=resumed_wandb_id,
            resume="allow",
        )

    n_epochs  = cfg.get("epochs", 100)
    ckpt_freq = cfg.get("checkpoint_freq") or cfg.get("save_every", 10)
    epoch_bar = tqdm(range(start_epoch, n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0

        residual_mode = cfg.get("residual_mode", False)

        batch_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
        for batch in batch_bar:
            terrain, settlement = batch
            terrain = terrain.to(device)         # Condition c (embedding)
            settlement = settlement.to(device)   # Clean target x0

            # Residual mode: train on (settlement - terrain) so the sparse ring signal
            # dominates instead of the dense terrain copy.
            x0 = (settlement - terrain) if residual_mode else settlement

            t = torch.randint(0, diffusion.T, (terrain.shape[0],), device=device)
            x_t, noise = diffusion.q_sample(x0, t)

            model_condition = terrain

            x0_clamp_val = 2.0 if residual_mode else 1.0

            # Forward pass + loss under autocast: conv/matmul ops run in amp_dtype (tensor-core
            # speedup), ops that need fp32 precision (norm, scipy-backed connectivity, etc.) are
            # kept out of it below or auto-promoted by autocast's own op whitelist.
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Self-conditioning (Chen et al. 2023): half the time, first compute an x0 estimate
                # without self-conditioning (no grad -- it's only used as extra input, not backpropped
                # through), then train the real step with that estimate appended as conditioning. At
                # inference the model sees its own previous step's x0 estimate the same way.
                if self_conditioning:
                    self_cond = torch.zeros_like(x0)
                    if torch.rand(1).item() < 0.5:
                        with torch.no_grad():
                            prelim_condition = torch.cat([model_condition, self_cond], dim=1)
                            prelim_noise = model(x_t, t, prelim_condition)
                            alpha_t_b = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                            self_cond = ((x_t - (1 - alpha_t_b).sqrt() * prelim_noise) / alpha_t_b.sqrt()
                                         ).clamp(-x0_clamp_val, x0_clamp_val)
                    model_condition = torch.cat([model_condition, self_cond], dim=1)

                predicted_noise = model(x_t, t, model_condition)

                # Weighted MSE: multiplicative weighting mechanisms, each independent.
                #  - generation_weight: upweights voxels at/near structure (ring/wall positions),
                #    smoothly graduated by distance (DenseWeight-style) -- see
                #    compute_smooth_weight_map. Dataset-specific (depends on x0 content).
                #  - min_snr_gamma: Min-SNR-gamma per-timestep weighting (Hang et al. 2023) -- caps
                #    the implicit over-weighting of low-noise timesteps. Dataset-agnostic (depends
                #    only on the noise schedule) -- see compute_min_snr_weight.
                #  - connectivity_boost: topology-inspired, upweights voxels in small/thin true-
                #    structure components relative to the largest one -- see compute_connectivity_weight.
                weight_map = None
                generation_weight = cfg.get("generation_weight", cfg.get("ring_weight", 1.0))
                if generation_weight > 1.0:
                    with torch.no_grad():
                        changed = (x0.norm(dim=1, keepdim=True) > 1e-6)
                        weight_decay = cfg.get("generation_weight_decay", 0.5)
                        weight_shells = cfg.get("generation_weight_shells", 4)
                        weight_map = compute_smooth_weight_map(
                            changed, generation_weight, weight_decay, weight_shells
                        ).expand_as(noise)

                min_snr_gamma = cfg.get("min_snr_gamma")
                if min_snr_gamma is not None:
                    with torch.no_grad():
                        min_snr_weight = compute_min_snr_weight(diffusion, t, min_snr_gamma)
                        weight_map = min_snr_weight if weight_map is None else weight_map * min_snr_weight

                if connectivity_boost > 0:
                    with torch.no_grad():
                        conn_weight = compute_connectivity_weight(x0, connectivity_boost).expand_as(noise)
                        weight_map = conn_weight if weight_map is None else weight_map * conn_weight

                if weight_map is not None:
                    loss = (weight_map * (predicted_noise - noise).pow(2)).mean()
                else:
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                # coarse_shape_weight / coarse_shape_weight_global / false_positive_weight /
                # smoothness_weight: auxiliary losses on the predicted x0, added on top of the
                # main (weighted) noise-prediction loss. All need x0_pred, computed once and shared.
                if (coarse_shape_weight > 0 or coarse_shape_weight_global > 0
                        or false_positive_weight > 0 or smoothness_weight > 0):
                    alpha_t_b = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                    x0_pred = ((x_t - (1 - alpha_t_b).sqrt() * predicted_noise) / alpha_t_b.sqrt()
                               ).clamp(-x0_clamp_val, x0_clamp_val)
                    if coarse_shape_weight > 0:
                        loss = loss + coarse_shape_weight * compute_coarse_shape_loss(x0_pred, x0)
                    if coarse_shape_weight_global > 0:
                        loss = loss + coarse_shape_weight_global * compute_coarse_shape_loss(
                            x0_pred, x0, downsample_factor=coarse_shape_downsample_global
                        )
                    if false_positive_weight > 0:
                        loss = loss + false_positive_weight * compute_false_positive_loss(x0_pred, x0)
                    if smoothness_weight > 0:
                        loss = loss + smoothness_weight * compute_smoothness_loss(x0_pred)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1}/{n_epochs}     loss={avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        if use_wandb:
            log_dict = {"train/loss": avg_loss}
            if scheduler is not None:
                log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(log_dict, step=epoch + 1)

        if (epoch + 1) % ckpt_freq == 0 or (epoch + 1) == n_epochs:
            ckp_path = out_dir / f"ckpt_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "cfg": cfg,
                "wandb_run_id": wandb.run.id if use_wandb else None,
            }, ckp_path)
            print(f"Saved checkpoint to {ckp_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Override wandb run name")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve YAML paths relative to the config's parent directory (ml/src/diffusion/)
    script_dir = Path(args.config).resolve().parent.parent
    for key in ("dataset_path_binary", "dataset_path_categorical",
                "embedding_path", "output_dir"):
        if key in cfg:
            cfg[key] = str((script_dir / cfg[key]).resolve())

    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    if args.wandb_name is None:
        args.wandb_name = f"{Path(args.config).stem}_{hyperparam_folder(cfg)}"

    torch.manual_seed(cfg.get("seed", 42))
    train(cfg, use_wandb=not args.no_wandb, wandb_name=args.wandb_name,
          resume_path=args.resume)
