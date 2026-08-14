import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Timestep embedding

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodes a scalar timestep t into a vector using sinusoids.
    Identical to positional encodings in transformers.
    """
    def __init__(self, dim: int):
        """
        :param dim: Output embedding dimension (must be even)
        """
        super().__init__()
        self.dim = dim
        half = dim // 2
        # Geometric sequence of frequencies, same as the original transformer positional encoding.
        # Depends only on `dim`, so it's precomputed once here rather than every forward call.
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        freq_k = exp(-log(10000) * k / (dim/2-1))
        emb = [sin(t * freq_0), ..., sin(t * freq_{d/2-1}), cos(t * freq_0), ..., cos(t * freq_{d/2-1})]

        :param t: Timestep tensor, shape: (batch_size,).
        :return: Sinusoidal time embedding, shape: (batch_size, dim)
        """
        emb = t[:, None].float() * self.freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)



# Building blocks

class ResBlock3D(nn.Module):
    """
    Residual block with GroupNorm and timestep injection.
    The timestep is projected to match the channel count and added to the feature map.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        """
        :param in_ch: Number of input channels.
        :param out_ch: Number of output channels.
        :param time_dim: Dimension of the timestep embedding vector.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, film: tuple = None) -> torch.Tensor:
        """
        :param x: Input feature map, shape: (batch_size, in_ch, H, W).
        :param time_emb: Timestep embedding, shape: (batch_size, time_dim).
        :param film: Optional (gamma, beta) from a FiLMHead3D, each shape (batch_size, out_ch, D, H, W),
            applied as gamma * h + beta right after timestep injection. None (default) skips this.
        :return: Output feature map, shape: (batch_size, out_ch, H, W).
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # reshape for broadcasting over spatial dims
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None, None]
        if film is not None:
            gamma, beta = film
            h = gamma * h + beta
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    """
    One encoder stage: ResBlock3D followed by a stride-2 conv that halves spatial dims.
    Returns both the downsampled output and pre-downsampling activations for the skip connection.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        """
        :param in_ch: Numer of input channels.
        :param out_ch: Number of output channels.
        :param time_dim: Dimension of the timestep embedding vector.
        """
        super().__init__()
        self.res = ResBlock3D(in_ch, out_ch, time_dim)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb, film: tuple = None):
        """
        :param x: Input feature map.
        :param t_emb: Timestep embedding vector.
        :param film: Optional (gamma, beta) passed through to the internal ResBlock3D.
        :return: (downsampled x, pre-downsample activations)
        """
        x = self.res(x, t_emb, film)
        return self.down(x), x


class UpBlock(nn.Module):
    """
    One decoder stage: Transposed conv doubles spatial dims, concatenates the encoder skip connection, then applies
    ResBlock3D to merge the two feature maps.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        """
        :param in_ch: Number of input channels from the previous decoder stage.
        :param skip_ch: Number of channels in the skip connection in the matching encoder stage.
        :param out_ch: Number of output channels.
        :param time_dim: Dimension of the timestep embedding vector.
        """
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.res = ResBlock3D(in_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        """
        :param x: Input feature map from the previous decoder stage.
        :param skip: Skip connection from the matching encoder stage.
        :param t_emb: Timestep embedding vector.
        :return: Merged feature map for the next decoder stage.
        """
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


# Full U-Net Denoiser

class SinusoidalPositionEmbedding3D(nn.Module):
    """
    Sinusoidal positional embedding for a 3D grid, computed dynamically from the actual
    (D, H, W) at forward time -- no fixed-size learned table, so it works unchanged on any
    volume shape rather than being tied to one dataset's dimensions.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.per_axis = channels // 3
        # freqs depend only on the (fixed) per-axis channel count, not on the volume shape
        # (D, H, W) seen at forward time, so they're precomputed once here. D and H always
        # share the same per-axis dim, so one buffer covers both.
        self._freqs_dh = self._make_freqs(self.per_axis)
        self._freqs_w = self._make_freqs(channels - 2 * self.per_axis)

    @staticmethod
    def _make_freqs(dim: int) -> torch.Tensor:
        if dim <= 0:
            return torch.zeros(0)
        # Ceiling division: for odd dim, floor division would produce 2*half < dim columns
        # before the final slice, silently returning fewer columns than the caller asked for
        # (only surfacing later as a shape-mismatch error at the .expand() call site).
        half = max((dim + 1) // 2, 1)
        return torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))

    def _axis_embedding(self, length: int, dim: int, freqs: torch.Tensor, device) -> torch.Tensor:
        if dim <= 0:
            return torch.zeros(length, 0, device=device)
        pos = torch.arange(length, device=device).float()
        angles = pos[:, None] * freqs.to(device)[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb[:, :dim]

    def forward(self, D: int, H: int, W: int, device) -> torch.Tensor:
        d_dim = self.per_axis
        h_dim = self.per_axis
        w_dim = self.channels - 2 * self.per_axis  # remainder (rounding) goes to W
        d_emb = self._axis_embedding(D, d_dim, self._freqs_dh, device)[:, None, None, :].expand(D, H, W, d_dim)
        h_emb = self._axis_embedding(H, h_dim, self._freqs_dh, device)[None, :, None, :].expand(D, H, W, h_dim)
        w_emb = self._axis_embedding(W, w_dim, self._freqs_w, device)[None, None, :, :].expand(D, H, W, w_dim)
        pos_emb = torch.cat([d_emb, h_emb, w_emb], dim=-1)  # [D, H, W, channels]
        return pos_emb.reshape(D * H * W, self.channels)


class SpatialAttention3D(nn.Module):
    """
    Multi-head self-attention over 3D feature maps (flattened to a sequence), with sinusoidal
    3D positional encoding. Without positional information, pure self-attention degenerates
    toward a rank-1, position-independent response (Dong et al. 2021, arxiv.org/abs/2103.03404)
    -- every query attends near-uniformly to all keys, producing the global-average-pooling
    block-filling collapse this module was previously removed for. Positional encoding lets
    attention distinguish voxels by location instead of only by feature content.

    The output projection is zero-initialized so this block starts as an exact identity
    (x + 0 = x), the same stabilization that made FiLMHead3D work: training begins identical to
    a model without attention, and the network only learns to use long-range attention if it
    actually helps. The first (unstabilized) attempt at adding this module regressed results --
    plausibly because random-init attention output was already perturbing every voxel from step
    one, on top of the rank-collapse risk positional encoding alone doesn't fully rule out.
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.pos_emb = SinusoidalPositionEmbedding3D(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x).reshape(B, C, -1).permute(0, 2, 1)  # [B, D*H*W, C]
        h = h + self.pos_emb(D, H, W, x.device)[None, :, :]
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, D, H, W)


class FiLMHead3D(nn.Module):
    """
    Predicts spatially-varying FiLM (Feature-wise Linear Modulation, Perez et al. 2017,
    arxiv.org/abs/1709.07871) scale/shift parameters for one U-Net resolution level, derived
    from the terrain condition. Without this, terrain conditioning is only seen once, channel-
    concatenated at the input -- every deeper layer has to propagate and re-derive "what does
    the terrain look like here" through ordinary convolutions. FiLM instead lets the terrain
    directly modulate features at every resolution level, the same way the timestep embedding
    already does (see ResBlock3D) but spatially-varying instead of a single global vector.

    Adaptively pools the (always full-resolution) condition tensor down to the exact spatial
    size needed at this level -- avoids any shape-mismatch risk from a separately-downsampled
    parallel tower, since adaptive pooling always hits the target size exactly regardless of
    rounding in the main network's stride-2 convs.

    The final conv is zero-initialized so this starts as an exact identity (gamma=1, beta=0):
    training begins identical to a model without FiLM, and the network only learns to use
    terrain-conditioned modulation if it actually helps, rather than starting from random
    scale/shift values that could destabilize the run from step one.
    """
    def __init__(self, cond_ch: int, out_ch: int, hidden_ch: int = None):
        super().__init__()
        hidden_ch = hidden_ch or max(out_ch, cond_ch)
        self.conv1 = nn.Conv3d(cond_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_ch, out_ch * 2, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, condition: torch.Tensor, target_shape) -> tuple:
        """
        :param condition: Terrain condition at its original full resolution, shape (B, cond_ch, D, H, W).
        :param target_shape: (D', H', W') to match the ResBlock's feature map at this level.
        :return: (gamma, beta), each shape (B, out_ch, D', H', W').
        """
        pooled = F.adaptive_avg_pool3d(condition, target_shape)
        h = F.silu(self.conv1(pooled))
        gamma_raw, beta = self.conv2(h).chunk(2, dim=1)
        return 1 + gamma_raw, beta


class ConditionalUNet3D(nn.Module):
    """
    3D U-Net that predicts the noise added to x_t at timestep t, conditioned on a terrain volume c. The terrain is
    concatenated channel-wise with x_t, so the network sees 2 * emb_dim input channels.
    """
    def __init__(self, emb_dim: int = 32, base_ch: int = 32, time_dim: int = 128, attention: bool = False,
                 self_conditioning: bool = False, film: bool = False):
        """
        :param emb_dim: Block2Vec embedding dimension (default: 32).
        :param base_ch: Base channel count for the U-Net doubles with each downscale.
        :param time_dim: Dimension of the sinusoidal timestep embedding.
        :param attention: If True, adds spatial self-attention after the bottleneck ResBlock.
        :param self_conditioning: If True, the condition tensor passed to forward() is expected to
            have an additional emb_dim channels appended (beyond emb_dim) holding the model's own
            previous x0 estimate (Chen et al. 2023, arxiv.org/abs/2208.04202), zeros if none yet.
            False (default) preserves the original channel count for backward compatibility.
        :param film: If True, adds a FiLMHead3D (see above) at each encoder resolution and the
            bottleneck, letting the terrain condition modulate features at every depth instead of
            only being seen once at the input. False (default) preserves the original architecture
            for backward compatibility with checkpoints trained before this option existed.
        """
        super().__init__()
        self_cond_ch = emb_dim if self_conditioning else 0
        cond_ch = emb_dim + self_cond_ch
        in_ch = emb_dim + cond_ch  # x_t + (condition + self-cond)
        self.film = film

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        # Encoder
        self.down1 = DownBlock(in_ch, base_ch, time_dim)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_dim)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_dim)

        # Bottleneck
        self.mid      = ResBlock3D(base_ch * 4, base_ch * 4, time_dim)
        self.mid_attn = SpatialAttention3D(base_ch * 4) if attention else nn.Identity()

        # Decoder
        self.up3 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, time_dim)
        self.up2 = UpBlock(base_ch * 2, base_ch * 2, base_ch, time_dim)
        self.up1 = UpBlock(base_ch, base_ch, emb_dim, time_dim)

        self.out = nn.Conv3d(emb_dim, emb_dim, kernel_size=1)

        if film:
            self.film1   = FiLMHead3D(cond_ch, base_ch)
            self.film2   = FiLMHead3D(cond_ch, base_ch * 2)
            self.film3   = FiLMHead3D(cond_ch, base_ch * 4)
            self.film_mid = FiLMHead3D(cond_ch, base_ch * 4)

    def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x_t: Noisy settlement volume, shape: (batch_size, emb_dim, D, H, W).
        :param t: Timestep tensor, shape: (batch_size,).
        :param condition: Terrain volume, shape: (batch_size, emb_dim [+ emb_dim if self-conditioning], D, H, W).
        :return: Predicted noise at timestep t, shape: (batch_size, emb_dim, D, H, W).
        """
        t_emb = self.time_mlp(t)
        # Concatenate terrain along the channel axis so every layer sees the condition
        x = torch.cat([x_t, condition], dim=1)

        film1 = self.film1(condition, x.shape[2:]) if self.film else None
        x, s1 = self.down1(x, t_emb, film1)
        film2 = self.film2(condition, x.shape[2:]) if self.film else None
        x, s2 = self.down2(x, t_emb, film2)
        film3 = self.film3(condition, x.shape[2:]) if self.film else None
        x, s3 = self.down3(x, t_emb, film3)

        film_mid = self.film_mid(condition, x.shape[2:]) if self.film else None
        x = self.mid(x, t_emb, film_mid)
        x = self.mid_attn(x)

        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        return self.out(x)

