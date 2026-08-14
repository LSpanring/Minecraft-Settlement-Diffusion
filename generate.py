"""
Unified diffusion generation script.

Usage:
    python generate.py ring --mode binary --grid-x 0 --grid-z 0
    python generate.py adaptive_ring --mode binary
    python generate.py mikes_angels --mode binary --checkpoint path/to/ckpt.pt
    python generate.py ring --sampler ddpm --ddpm-steps 100

Sampler:
    DDIM (default): fast deterministic sampling, ~20 steps
    DDPM:           requires --sampler ddpm, uses full T steps

Output:
    output/diffusion/<dataset>/generate/world/     -- copy of minecraft world + placements
    output/diffusion/<dataset>/generate/render.png -- mcrender output
"""

import argparse
import json
import math
import shutil
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml
import nbt.nbt as nbt_lib
import anvil
import numpy as np
import torch

REPO_ROOT    = Path(__file__).parent.parent.parent.parent
CONFIG_DIR   = Path(__file__).parent / "configs"
SOURCE_WORLD = REPO_ROOT / "output" / "generator_test" / "terrain" / "16x16-6" / "world"

Y_OFFSET   = 60
DDIM_STEPS = 20
DDPM_STEPS = 100
GRID_H     = 2
GRID_W     = 2

MC_ADDITION = "minecraft:red_concrete"
MC_REMOVAL  = "minecraft:glass"

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from model import ConditionalUNet3D
from diffusion import GaussianDiffusion


# ── config loading ─────────────────────────────────────────────────────────────

def load_dataset_config(dataset: str) -> dict:
    cfg_path = CONFIG_DIR / f"{dataset}.yaml"
    if not cfg_path.exists():
        available = [p.stem for p in CONFIG_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"No config for '{dataset}'. Available: {available}"
        )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Paths in YAML are relative to ml/src/diffusion/ (the script directory)
    script_dir = cfg_path.parent.parent
    for key in ("dataset_path_binary", "dataset_path_categorical",
                "embedding_path", "output_dir", "props_path"):
        if key in cfg:
            cfg[key] = (script_dir / cfg[key]).resolve()
    return cfg


# ── coordinate mapping ─────────────────────────────────────────────────────────

def load_props(props_path: Path):
    with open(props_path) as f:
        p = json.load(f)
    chunk_offset  = p["chunkOffset"]
    build_size    = p["buildAreaChunkSize"][0]  # chunks per side
    sample_counts = p["sampleCounts"]           # [rows, cols]
    return chunk_offset, build_size, sample_counts

def sample_world_xz(sample_idx, chunk_offset, build_size, grid_z):
    x_grid = sample_idx // grid_z
    z_grid = sample_idx %  grid_z
    wx = (chunk_offset[0] + x_grid * build_size) * 16
    wz = (chunk_offset[1] + z_grid * build_size) * 16
    return wx, wz


# ── BlockStates NBT helpers ────────────────────────────────────────────────────

def _bpb(n):
    return max(4, math.ceil(math.log2(max(n, 2))))

def _unpack_bs(longs, palette_size):
    bits = _bpb(palette_size); bpl = 64 // bits; mask = (1 << bits) - 1; out = []
    for v in longs:
        u = v & 0xFFFFFFFFFFFFFFFF
        for j in range(bpl):
            out.append((u >> (j * bits)) & mask)
            if len(out) == 4096:
                return out
    return out[:4096]

def _pack_bs(indices, palette_size):
    bits = _bpb(palette_size); bpl = 64 // bits; mask = (1 << bits) - 1; result = []
    for i in range(0, 4096, bpl):
        v = 0
        for j in range(min(bpl, 4096 - i)):
            v |= (indices[i + j] & mask) << (j * bits)
        result.append(v)  # keep unsigned — TAG_Long_Array uses '>NQ' format
    return result

def _make_empty_section(sy: int):
    """Build a fresh all-air NBT section for a Y-index that has no on-disk section.

    Minecraft never saves sections that are pure air (standard space optimization), so
    any placement above the natural terrain height in a chunk has no section to write
    into. anvil.EmptySection/EmptyChunk already handle this for from-scratch chunks;
    this mirrors that for sections injected into an existing, otherwise-untouched chunk.
    """
    section = nbt_lib.TAG_Compound()
    section.tags.append(nbt_lib.TAG_Byte(name="Y", value=sy))
    air_entry = nbt_lib.TAG_Compound()
    air_entry.tags.append(nbt_lib.TAG_String(name="Name", value="minecraft:air"))
    palette_tag = nbt_lib.TAG_List(name="Palette", type=nbt_lib.TAG_Compound)
    palette_tag.tags.append(air_entry)
    section.tags.append(palette_tag)
    bstates_tag = nbt_lib.TAG_Long_Array(name="BlockStates")
    bstates_tag.value = _pack_bs([0] * 4096, 1)
    section.tags.append(bstates_tag)
    return section


def _apply_to_section(section, placements):
    tag_names = [t.name for t in section.tags]
    if "Palette" not in tag_names or "BlockStates" not in tag_names:
        return
    palette_tag = section["Palette"]
    bstates_tag = section["BlockStates"]
    old_size    = len(palette_tag)
    indices     = _unpack_bs(list(bstates_tag.value), old_size)
    pal_map = {entry["Name"].value: i for i, entry in enumerate(palette_tag)}
    for (_, _, _, name) in placements:
        if name not in pal_map:
            entry = nbt_lib.TAG_Compound()
            entry.tags.append(nbt_lib.TAG_String(name="Name", value=name))
            palette_tag.tags.append(entry)
            pal_map[name] = len(palette_tag) - 1
    for (lx, ly, lz, name) in placements:
        indices[ly * 256 + lz * 16 + lx] = pal_map[name]
    bstates_tag.value = _pack_bs(indices, len(palette_tag))


def build_name_to_palette_id(palette: list) -> dict:
    """Map base block name -> dataset palette index, ignoring block states.

    Mirrors apply_settlements/_apply_to_section, which already place blocks by base
    name only (palette[id][0]) and discard states. Prefers the state-less variant of
    a name when the palette has multiple entries for it.
    """
    name_to_id = {}
    for i, entry in enumerate(palette):
        name, states = entry[0], entry[1]
        if name not in name_to_id or not states:
            name_to_id[name] = i
    return name_to_id


def load_terrain_from_world(world_path: Path, world_positions, vol_xz: int, vol_y: int,
                            y_offset: int, name_to_id: dict) -> list:
    """Read terrain block IDs directly from a live Minecraft world.

    Conditioning terrain used to come from the static training dataset (data[i, 0]),
    which goes stale whenever --source-world points at a different or since-modified
    world: the model reasons about terrain that no longer matches what placements are
    diffed against and written into. Reading live from the actual source world -- the
    same approach ml/src/run.py's GAN pipeline already uses -- keeps conditioning and
    placement target in sync regardless of which world is passed.
    """
    region_cache = {}
    terrain_vols = []
    n_chunks = vol_xz // 16

    for wx_start, wz_start in world_positions:
        vol = np.zeros((vol_y, vol_xz, vol_xz), dtype=np.int32)
        cx0, cz0 = wx_start // 16, wz_start // 16

        for cdx in range(n_chunks):
            for cdz in range(n_chunks):
                cx_abs, cz_abs = cx0 + cdx, cz0 + cdz
                rx, rz = cx_abs // 32, cz_abs // 32
                if (rx, rz) not in region_cache:
                    region_file = world_path / "region" / f"r.{rx}.{rz}.mca"
                    region_cache[(rx, rz)] = anvil.Region.from_file(str(region_file)) if region_file.exists() else None
                region = region_cache[(rx, rz)]
                if region is None:
                    continue
                try:
                    chunk = region.get_chunk(cx_abs % 32, cz_abs % 32)
                except Exception:
                    continue

                for section in chunk.data["Sections"]:
                    sy = section["Y"].value
                    sec_y0 = sy * 16
                    if sec_y0 + 16 <= y_offset or sec_y0 >= y_offset + vol_y:
                        continue
                    tag_names = [t.name for t in section.tags]
                    if "Palette" not in tag_names or "BlockStates" not in tag_names:
                        continue
                    palette_tag = section["Palette"]
                    bstates_tag = section["BlockStates"]
                    local_names = [entry["Name"].value.split(":", 1)[-1] for entry in palette_tag]
                    local_to_target = np.array([name_to_id.get(n, 0) for n in local_names], dtype=np.int32)
                    indices = _unpack_bs(list(bstates_tag.value), len(palette_tag))
                    section_ids = local_to_target[indices].reshape(16, 16, 16)  # [ly, lz, lx]

                    for ly in range(16):
                        wy = sec_y0 + ly
                        yi = wy - y_offset
                        if 0 <= yi < vol_y:
                            xi0, zi0 = cdx * 16, cdz * 16
                            vol[yi, xi0:xi0 + 16, zi0:zi0 + 16] = section_ids[ly].T  # [lz,lx] -> [lx,lz]

        terrain_vols.append(vol)

    return terrain_vols


# ── model helpers ──────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("ckpt_epoch*.pt"),
                   key=lambda p: int(p.stem.replace("ckpt_epoch", "")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return ckpts[-1]

def load_model(ckpt_path: Path, device: str):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = ConditionalUNet3D(
        emb_dim          = cfg.get("emb_dim",    1),
        base_ch          = cfg.get("base_ch",   32),
        time_dim         = cfg.get("time_dim", 128),
        attention        = cfg.get("attention", False),
        self_conditioning = cfg.get("self_conditioning", False),
        film             = cfg.get("film", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    diff   = GaussianDiffusion(T=cfg.get("T", 100), device=device)
    diff.T = cfg.get("T", 100)
    return model, diff, cfg

def generate_samples(model, diffusion, terrain_vols, device, sampler="ddim", steps=DDIM_STEPS,
                     mode="binary", embeddings=None, residual_mode=False, residual_threshold=None,
                     boundary_crop=0, batch_size=4, eta=0.0, self_conditioning=False):
    x0_clamp = 2.0 if residual_mode else 1.0
    n = len(terrain_vols)
    settlements = []

    for batch_start in range(0, n, batch_size):
        batch = terrain_vols[batch_start:batch_start + batch_size]
        b = len(batch)
        print(f"  Generating samples {batch_start+1}-{batch_start+b}/{n} ({sampler}, {steps} steps)...")

        if mode == "binary":
            cond = torch.from_numpy(
                np.stack(batch).astype(np.float32) * 2 - 1
            ).unsqueeze(1).to(device)                                   # [B, 1, D, H, W]
        else:
            embs = np.stack([embeddings[tv].transpose(3, 0, 1, 2) for tv in batch])
            cond = torch.from_numpy(embs.astype(np.float32)).to(device) # [B, emb_dim, D, H, W]

        # model_condition feeds the network; cond stays narrow (emb_dim) for the shape arg
        # and the residual add-back below, which must match the model's emb_dim-channel output.
        model_condition = cond

        with torch.no_grad():
            if sampler == "ddim":
                out = diffusion.ddim_sample(model, model_condition, shape=cond.shape,
                                            ddim_steps=steps, x0_clamp=x0_clamp, eta=eta,
                                            self_conditioning=self_conditioning)
            else:
                diffusion.T = steps
                out = diffusion.p_sample_loop(model, model_condition, shape=cond.shape)

        if residual_mode:
            if boundary_crop > 0:
                out[:, :, :boundary_crop] = 0
                out[:, :, -boundary_crop:] = 0
            if residual_threshold is not None:
                residual_norm = out.norm(dim=1, keepdim=True)
                out = out * (residual_norm > residual_threshold).float()
            out = out + cond

        for j in range(b):
            settlements.append(out[j].cpu().numpy())  # [emb_dim, D, H, W] or [1, D, H, W]

    return settlements


def compute_freq_weights(dataset_path: str, n_blocks: int, freq_weight_exp: float) -> np.ndarray:
    """Compute per-block frequency weights for nearest-neighbour decode.

    Formula from van der Staaij et al. (2024): weight(b) = (total / count(b))^w
    Applied by multiplying squared distances before argmin, biasing toward common blocks.
    """
    data = np.load(dataset_path, mmap_mode="r")
    settlement_ids = data[:, 1].ravel().astype(np.int64)
    counts = np.bincount(settlement_ids, minlength=n_blocks).astype(np.float64)
    counts[:n_blocks] = np.where(counts[:n_blocks] > 0, counts[:n_blocks], 1)  # avoid div-by-zero
    total = counts.sum()
    weights = (total / counts) ** freq_weight_exp   # shape [n_blocks]
    return weights.astype(np.float32)


def decode_embeddings(settlement_emb, embeddings, threshold=None, freq_weights=None, batch_size=4096):
    """Nearest-neighbour decode: [emb_dim, D, H, W] → [D, H, W] block IDs.

    threshold: if set, voxels whose nearest embedding distance (L2) exceeds this
               value are decoded as air (ID 0). Tune with --cat-threshold.
    freq_weights: per-block weight array [n_blocks]; distances are multiplied by these
                  before argmin, biasing toward common blocks (van der Staaij et al. 2024).
    """
    emb_dim = embeddings.shape[1]
    flat = settlement_emb.reshape(emb_dim, -1).T          # [N, emb_dim]
    N = flat.shape[0]
    ids = np.zeros(N, dtype=np.int32)

    for start in range(0, N, batch_size):
        chunk = flat[start:start + batch_size]             # [B, emb_dim]
        dists = np.sum((chunk[:, None, :] - embeddings[None, :, :]) ** 2, axis=-1)  # [B, n_blocks]
        if freq_weights is not None:
            dists *= freq_weights[None, :]                 # multiply by (total/count)^w
        nearest = dists.argmin(axis=-1)
        if threshold is not None:
            min_dists = np.sum((chunk - embeddings[nearest]) ** 2, axis=-1)
            nearest[min_dists > threshold ** 2] = 0        # L2 dist > threshold → air
        ids[start:start + batch_size] = nearest

    return ids.reshape(settlement_emb.shape[1:])


# ── world editing ──────────────────────────────────────────────────────────────

def set_world_spawn(world_path: Path, x: int, y: int, z: int):
    level_dat = world_path / "level.dat"
    nbt_file = nbt_lib.NBTFile(str(level_dat))
    nbt_file["Data"]["SpawnX"].value = x
    nbt_file["Data"]["SpawnY"].value = y
    nbt_file["Data"]["SpawnZ"].value = z
    nbt_file.write_file(str(level_dat))



def apply_settlements(world_path, terrain_vols, settlements, sample_indices,
                      chunk_offset, build_size, grid_z, mode, threshold=0.0,
                      embeddings=None, palette=None, freq_weights=None,
                      world_positions=None, already_decoded=False):
    region_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, (sample_idx, terrain_vol, settlement_raw) in enumerate(
            zip(sample_indices, terrain_vols, settlements)):
        if world_positions is not None:
            wx_start, wz_start = world_positions[i]
        else:
            wx_start, wz_start = sample_world_xz(sample_idx, chunk_offset, build_size, grid_z)

        if mode == "binary":
            # settlement_raw carries a leading channel axis ([1, D, H, W], emb_dim=1) from
            # generate_samples()'s [B, emb_dim, D, H, W] output -- categorical's decode_embeddings
            # explicitly consumes and drops that axis, but binary has no decode step, so it must
            # be squeezed here or every downstream 3D index (diff_mask, settlement_bin[yi,xi,zi])
            # silently picks up an extra leading dimension.
            settlement_3d  = settlement_raw[0] if settlement_raw.ndim == 4 else settlement_raw
            settlement_bin = settlement_3d > threshold
            diff_mask      = settlement_bin != terrain_vol
            ys, xs, zs     = np.where(diff_mask)
            additions = removals = 0
            for yi, xi, zi in zip(ys.tolist(), xs.tolist(), zs.tolist()):
                wx = wx_start + int(xi)
                wy = Y_OFFSET  + int(yi)
                wz = wz_start  + int(zi)
                rx, rz         = wx // 512, wz // 512
                cx_abs, cz_abs = wx // 16,  wz // 16
                sy             = wy // 16
                lx, ly, lz     = wx % 16, wy % 16, wz % 16
                block = MC_ADDITION if bool(settlement_bin[yi, xi, zi]) else MC_REMOVAL
                region_data[(rx, rz)][(cx_abs, cz_abs)][sy].append((lx, ly, lz, block))
                if block == MC_ADDITION:
                    additions += 1
                else:
                    removals += 1
        else:
            # Categorical: decode embedding → block IDs, diff against terrain
            if already_decoded:
                sett_ids = settlement_raw  # [D, H, W] block IDs from ensemble union
            else:
                sett_ids = decode_embeddings(settlement_raw, embeddings, threshold=threshold,
                                             freq_weights=freq_weights)  # [D, H, W]
            AIR_ID   = 0
            additions = removals = 0
            diff_mask = sett_ids != terrain_vol
            ys, xs, zs = np.where(diff_mask)
            for yi, xi, zi in zip(ys.tolist(), xs.tolist(), zs.tolist()):
                sett_id    = int(sett_ids[yi, xi, zi])
                terrain_id = int(terrain_vol[yi, xi, zi])
                if sett_id == AIR_ID and terrain_id == AIR_ID:
                    continue
                wx = wx_start + int(xi)
                wy = Y_OFFSET  + int(yi)
                wz = wz_start  + int(zi)
                rx, rz         = wx // 512, wz // 512
                cx_abs, cz_abs = wx // 16,  wz // 16
                sy             = wy // 16
                lx, ly, lz     = wx % 16, wy % 16, wz % 16
                block_name = "minecraft:" + palette[sett_id][0]
                region_data[(rx, rz)][(cx_abs, cz_abs)][sy].append((lx, ly, lz, block_name))
                if terrain_id == AIR_ID:
                    additions += 1
                else:
                    removals += 1

        print(f"  Sample {i+1} (idx={sample_idx}) at world ({wx_start},{wz_start}): "
              f"{additions} additions, {removals} removals")

    for (rx, rz), chunks_data in region_data.items():
        region_file = world_path / "region" / f"r.{rx}.{rz}.mca"
        if not region_file.exists():
            print(f"  Warning: {region_file.name} not found -- skipping")
            continue
        print(f"  Editing {region_file.name} ...")
        src_region = anvil.Region.from_file(str(region_file))
        new_region = anvil.EmptyRegion(rx, rz)
        for cz_local in range(32):
            for cx_local in range(32):
                cx_abs = rx * 32 + cx_local
                cz_abs = rz * 32 + cz_local
                try:
                    chunk = src_region.get_chunk(cx_local, cz_local)
                except Exception:
                    continue
                if (cx_abs, cz_abs) in chunks_data:
                    chunk_placements = chunks_data[(cx_abs, cz_abs)]
                    sections_list = chunk.data["Sections"]
                    sections_by_y = {s["Y"].value: s for s in sections_list}
                    for sy, placements in chunk_placements.items():
                        section = sections_by_y.get(sy)
                        valid = section is not None and {"Palette", "BlockStates"} <= {t.name for t in section.tags}
                        if not valid:
                            # Pure-air section was never saved to disk -- create one so
                            # placements above the natural terrain height aren't dropped.
                            if section is not None:
                                sections_list.tags.remove(section)
                            section = _make_empty_section(sy)
                            sections_list.tags.append(section)
                            sections_by_y[sy] = section
                        _apply_to_section(section, placements)
                new_region.add_chunk(chunk)
        new_region.save(str(region_file))
        print(f"    Saved {region_file.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate diffusion model settlements and place them into a Minecraft world."
    )
    parser.add_argument("dataset",
                        help="Dataset config name — stem of configs/<name>.yaml "
                             "(e.g. ring, adaptive_ring, mikes_angels)")
    parser.add_argument("--mode",       default="binary", choices=["binary", "categorical"],
                        help="Dataset mode (default: binary)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint file (default: latest in output_dir)")
    parser.add_argument("--sampler",    default="ddim", choices=["ddim", "ddpm"],
                        help="Sampling method (default: ddim)")
    parser.add_argument("--ddim-steps", type=int, default=DDIM_STEPS,
                        help=f"DDIM sampling steps (default: {DDIM_STEPS})")
    parser.add_argument("--ddpm-steps", type=int, default=DDPM_STEPS,
                        help=f"DDPM sampling steps, used only with --sampler ddpm (default: {DDPM_STEPS})")
    parser.add_argument("--grid-x",    type=int, default=24,
                        help="Grid row of top-left sample (default: 24)")
    parser.add_argument("--grid-z",    type=int, default=24,
                        help="Grid column of top-left sample (default: 24)")
    parser.add_argument("--grid-h",    type=int, default=GRID_H,
                        help=f"Grid height in samples (default: {GRID_H})")
    parser.add_argument("--grid-w",    type=int, default=GRID_W,
                        help=f"Grid width in samples (default: {GRID_W})")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (checkpoint search and world/render output)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Binary placement threshold on model output in [-1,1] (default: 0.0)")
    parser.add_argument("--cat-threshold", type=float, default=None,
                        help="Categorical: max L2 distance to nearest embedding; farther voxels become air (default: none)")
    parser.add_argument("--freq-weight-exp", type=float, default=0.05,
                        help="Frequency weighting exponent w for NN decode (van der Staaij et al. 2024): "
                             "distances *= (total/count)^w, biasing toward common blocks. "
                             "0=disabled, paper best=0.05, range [0,0.075] (default: 0.05)")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Ensemble: number of stochastic DDIM runs; block union fills gaps (default: 1)")
    parser.add_argument("--eta", type=float, default=0.5,
                        help="DDIM stochasticity: 0=deterministic, 1=full DDPM noise (default: 0.5)")
    parser.add_argument("--residual-threshold", type=float, default=None,
                        help="Residual mode: zero voxels whose residual L2 norm is below this before decoding (default: none)")
    parser.add_argument("--boundary-crop", type=int, default=0,
                        help="Residual mode: zero out this many Y-slices at each volume boundary to suppress UNet edge artifacts (default: 0)")
    parser.add_argument("--batch-size",  type=int, default=4,
                        help="Samples processed per DDIM forward pass (default: 4)")
    parser.add_argument("--source-world", default=None,
                        help="Path to source Minecraft world to copy as base terrain (default: data/raw/large/minecraft/world)")
    parser.add_argument("--chunk-offset", type=int, nargs=2, default=None, metavar=("X", "Z"),
                        help="Override chunk offset from properties.json, e.g. --chunk-offset 0 0")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset config
    cfg = load_dataset_config(args.dataset)

    dataset_key = f"dataset_path_{args.mode}"
    if dataset_key not in cfg:
        raise KeyError(f"Config '{args.dataset}.yaml' has no field '{dataset_key}'")
    if "props_path" not in cfg:
        raise KeyError(f"Config '{args.dataset}.yaml' has no field 'props_path'")

    dataset_path = Path(cfg[dataset_key])
    ckpt_dir     = Path(args.output_dir).resolve() if args.output_dir else Path(cfg["output_dir"])
    output_dir   = ckpt_dir / "generate"

    chunk_offset, build_size, sample_counts = load_props(Path(cfg["props_path"]))
    if args.chunk_offset is not None:
        chunk_offset = args.chunk_offset
    grid_z = sample_counts[1]   # number of samples per row in the Z direction
    vol_xz = build_size * 16    # blocks per sample side

    # Checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(ckpt_dir)
    print(f"Checkpoint: {ckpt_path}")

    print("Loading model ...")
    model, diffusion, ckpt_cfg = load_model(ckpt_path, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --grid-x/z are absolute chunk coordinates; convert to sample-grid indices
    # by subtracting the dataset's chunk_offset and dividing by build_size.
    gx, gz, gh, gw = args.grid_x, args.grid_z, args.grid_h, args.grid_w
    gx_sample = (gx - chunk_offset[0]) // build_size
    gz_sample = (gz - chunk_offset[1]) // build_size
    sample_indices = [
        (gx_sample + dx) * grid_z + (gz_sample + dz)
        for dx in range(gh)
        for dz in range(gw)
    ]
    world_positions = [
        (gx * 16 + dx * vol_xz, gz * 16 + dz * vol_xz)
        for dx in range(gh)
        for dz in range(gw)
    ]

    source_world = Path(args.source_world).resolve() if args.source_world else SOURCE_WORLD

    data  = np.load(str(dataset_path), mmap_mode="r")
    vol_y = data.shape[2]
    print(f"  Dataset: {args.dataset} ({args.mode}), shape={data.shape}")
    print(f"  Grid chunk ({gx},{gz}) {gh}x{gw} -> sample indices {sample_indices}, world origin ({gx*16},{gz*16})")

    embeddings = None
    cat_palette = None
    freq_weights = None
    if args.mode == "categorical":
        embeddings = np.load(str(cfg["embedding_path"]))
        # Apply the same normalization used during training (scale to [-1, 1]).
        # emb_scale is stored in the checkpoint cfg; fall back to computing it from the
        # embeddings file so old checkpoints (trained without normalization) still work.
        emb_scale = ckpt_cfg.get("emb_scale", float(np.abs(embeddings).max()))
        embeddings = embeddings / emb_scale
        palette_path = Path(cfg["dataset_path_categorical"]).parent / "palette.json"
        with open(palette_path) as f:
            cat_palette = json.load(f)
        freq_weights = None
        if args.freq_weight_exp > 0:
            print(f"  Computing block frequency weights (w={args.freq_weight_exp}) ...")
            freq_weights = compute_freq_weights(
                cfg["dataset_path_categorical"], len(embeddings), args.freq_weight_exp
            )
        # Read conditioning terrain live from source_world rather than the frozen
        # training dataset -- keeps the model's input in sync with whatever world
        # placements actually get written into (see load_terrain_from_world docstring).
        print(f"Loading terrain samples live from {source_world} ...")
        name_to_id = build_name_to_palette_id(cat_palette)
        terrain_vols = np.array(load_terrain_from_world(
            source_world, world_positions, vol_xz, vol_y, Y_OFFSET, name_to_id
        ))
    else:
        print("Loading terrain samples ...")
        terrain_vols = np.array([data[i, 0] for i in sample_indices])

    residual_mode = ckpt_cfg.get("residual_mode", False)
    if residual_mode:
        print("  Residual mode: adding terrain back to model output before decode.")

    steps = args.ddim_steps if args.sampler == "ddim" else args.ddpm_steps
    residual_threshold = args.residual_threshold if args.mode == "categorical" else None
    active_threshold = args.cat_threshold if args.mode == "categorical" else args.threshold

    n_runs = args.n_runs if (args.mode == "categorical" and args.sampler == "ddim" and args.n_runs > 1) else 1
    eta = args.eta if n_runs > 1 else 0.0
    self_conditioning = ckpt_cfg.get("self_conditioning", False)

    if n_runs > 1:
        print(f"Stochastic ensemble: {n_runs} runs, eta={eta}")
        all_run_decoded = []
        for run_i in range(n_runs):
            print(f"\n  Run {run_i+1}/{n_runs} ...")
            setts = generate_samples(model, diffusion, terrain_vols, device, args.sampler, steps,
                                     mode=args.mode, embeddings=embeddings, residual_mode=residual_mode,
                                     residual_threshold=residual_threshold,
                                     boundary_crop=args.boundary_crop,
                                     batch_size=args.batch_size, eta=eta,
                                     self_conditioning=self_conditioning)
            run_decoded = [decode_embeddings(s, embeddings, threshold=active_threshold,
                                             freq_weights=freq_weights) for s in setts]
            all_run_decoded.append(run_decoded)
        print(f"\nComputing union of {n_runs} runs ...")
        settlements = []
        for j in range(len(sample_indices)):
            union = terrain_vols[j].copy()
            for run_decoded in all_run_decoded:
                mask = (run_decoded[j] != terrain_vols[j]) & (run_decoded[j] != 0)
                union[mask] = run_decoded[j][mask]
            settlements.append(union)
        already_decoded = True
    else:
        print(f"Generating settlements (sampler={args.sampler}, steps={steps}, eta={eta:.2f}) ...")
        settlements = generate_samples(model, diffusion, terrain_vols, device, args.sampler, steps,
                                       mode=args.mode, embeddings=embeddings, residual_mode=residual_mode,
                                       residual_threshold=residual_threshold,
                                       boundary_crop=args.boundary_crop,
                                       batch_size=args.batch_size, eta=eta,
                                       self_conditioning=self_conditioning)
        already_decoded = False

    world_path = output_dir / "world"
    print(f"\nCopying source world to {world_path} ...")
    if world_path.exists():
        shutil.rmtree(world_path)
    shutil.copytree(str(source_world), str(world_path),
                    ignore=shutil.ignore_patterns("session.lock"))
    print("  Done.")

    print("Placing settlements in world ...")
    apply_settlements(world_path, terrain_vols, settlements, sample_indices,
                      chunk_offset, build_size, grid_z, args.mode, active_threshold,
                      embeddings=embeddings, palette=cat_palette, freq_weights=freq_weights,
                      world_positions=world_positions, already_decoded=already_decoded)

    # Render bounding box for the full grid block
    render_x0 = gx * 16
    render_z0 = gz * 16
    render_x1 = render_x0 + gh * vol_xz
    render_z1 = render_z0 + gw * vol_xz
    render_y0  = Y_OFFSET
    render_y1  = Y_OFFSET + vol_y

    spawn_x = render_x0 - 2
    spawn_y = render_y0 + vol_y // 2
    spawn_z = render_z0 - 2
    print(f"Setting world spawn to ({spawn_x}, {spawn_y}, {spawn_z}) ...")
    set_world_spawn(world_path, spawn_x, spawn_y, spawn_z)

    output_dir.mkdir(parents=True, exist_ok=True)
    render_path = output_dir / "render.png"
    print(f"\nRendering with mcrender ...")
    cmd = [
        "mcrender", str(world_path), str(render_path),
        "--pos", str(render_x0), str(render_y0), str(render_z0),
        "--pos", str(render_x1), str(render_y1), str(render_z1),
        "--rotation", "3", "--force",
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  mcrender failed:", result.stderr[-300:] if result.stderr else "(none)")
    else:
        print(f"  Render saved to: {render_path}")


if __name__ == "__main__":
    main()
