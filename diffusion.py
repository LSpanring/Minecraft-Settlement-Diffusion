import torch
import numpy as np

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from https://arxiv.org/abs/2102.09672

    ᾱ_t = cos²(((t/T + s) / (1 + s)) · π/2)
    β_t = 1 - ᾱ_t / ᾱ_{t-1}

    :param T: Number of diffusion steps
    :param s: Small constant for numerical stability
    :return: Beta values for each diffusion step
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    # Cosine curve keeps noise levels low early in the schedule
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)

class GaussianDiffusion:
    """
    Wraps the forward and backward passes of the Gaussian diffusion model.
    """

    def __init__(self, T: int = 100, schedules: str = "cosine", device: str = "cpu"):
        """
        :param T: Number of time steps
        :param schedules: Name of the noise schedule to use (currently only "cosine" is supported)
        :param device: Torch device ("cuda" or "cpu")
        """
        self.T = T
        self.device = device

        # Precompute schedule quantities used in forward and reverse passes
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]])

        self.betas = betas  # β_t: per-step noise variance
        self.alphas_cumprod = alphas_cumprod    # α_t: cumulative signal retention
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()    # √ᾱ_t: signal weight in q_sample
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()  # √(1-ᾱ_t): noise weight in q_sample
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # Variance for the reverse step

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: add noise to original sample x_0 at timestep t.

        x_t = √ᾱ_t · x₀ + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)

        :param x_0: Clean sample, shape: (batch_size, embed_dim, D, H, W).
        :param t: 1D Tensor of timesteps, shape: (batch_size,).
        :return: Tuple of (noisy sample x_t, added noise), shape: (batch_size, embed_dim, D, H, W).
        """

        noise = torch.randn_like(x_0)
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t, x_0.ndim)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_0.ndim)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def p_sample_loop(
            self,
            model: torch.nn.Module,
            condition: torch.Tensor,
            shape: tuple,
    ) -> torch.Tensor:
        """
        Reverse process: generate a settlement from noise, based on terrain.

        :param model: Trained denoising model
        :param condition: Terrain volume, shape: (batch_size, embed_dim, D, H, W)
        :param shape: Output shape, (batch_size, embed_dim, D, H, W)
        :return: Generated settlement, shape: (batch_size, embed_dim, D, H, W)
        """
        device = condition.device
        x = torch.randn(shape, device=device)

        for t_int in reversed(range(self.T)):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            x = self._p_sample_step(model, x, t, condition)

        return x

    @torch.no_grad()
    def ddim_sample(self, model, condition, shape, ddim_steps=20, eta=0.0, x0_clamp=1.0,
                     self_conditioning=False):
        """
        Faster sampling using DDIM from https://arxiv.org/abs/2010.02502. No retraining needed.

        x̂₀ = (x_t - √(1-ᾱ_t) · ε_θ(x_t, t)) / √ᾱ_t
        x_{t-1} = √ᾱ_{t-1} · x̂₀ + √(1-ᾱ_{t-1}) · ε_θ(x_t, t)   (eta=0)

        :param model: Trained denoising model
        :param condition: Terrain volume, shape: (batch_size, embed_dim, D, H, W)
        :param shape: Output shape, (batch_size, embed_dim, D, H, W)
        :param ddim_steps: Number of DDIM sampling steps
        :param eta: Stochasticity factor; 0.0 is fully deterministic, 1.0 recovers DDPM noise
        :param x0_clamp: Absolute value clamp applied to predicted x0 each step. Use 1.0 for
                         full-embedding targets (normalized to [-1,1]) and 2.0 for residual targets
                         (settlement - terrain, which spans [-2, 2]).
        :param self_conditioning: If True, the model was trained to also see its own previous x0
                         estimate as extra conditioning (Chen et al. 2023, arxiv.org/abs/2208.04202).
                         The model must have been constructed with self_conditioning=True, since
                         this widens the expected condition channel count by shape[1].
        :return: Generated settlement volume, shape: (batch_size, embed_dim, D, H, W)
        """
        device = condition.device
        step_size = self.T // ddim_steps
        timesteps = list(reversed(range(0, self.T, step_size)))

        x = torch.randn(shape, device=device)
        self_cond = torch.zeros(shape, device=device) if self_conditioning else None
        for i, t_int in enumerate(timesteps):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            t_prev_int = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            alpha_t = self.alphas_cumprod[t_int]
            alpha_prev = self.alphas_cumprod[t_prev_int] if t_prev_int >= 0 else torch.ones(1, device=device)

            model_condition = torch.cat([condition, self_cond], dim=1) if self_conditioning else condition
            pred_noise = model(x, t, model_condition)
            x0_pred = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-x0_clamp, x0_clamp)
            if self_conditioning:
                self_cond = x0_pred
            # DDIM step with eta stochasticity (Song et al. 2020, Eq. 12)
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            direction = ((1 - alpha_prev - sigma ** 2).clamp(min=0)).sqrt() * pred_noise
            noise = sigma * torch.randn_like(x) if sigma > 0 else 0
            x = alpha_prev.sqrt() * x0_pred + direction + noise

        return x


    @staticmethod
    def _gather(values: torch.Tensor, t: torch.Tensor, ndim: int = 0) -> torch.Tensor:
        """
        Gathers schedule values for each t, then reshapes them for broadcasting.

        :param values: 1D schedule tensor, shape: (T,).
        :param t: Timesteps, shape: (batch_size,).
        :param ndim: Number of dimensions of the target tensor
        :return: Schedule values for each t, shape: (batch_size, 1, 1, ..., 1)
        """
        out = values.gather(-1, t)
        return out.reshape(t.shape[0], *([1] * (ndim - 1)))

    def _p_sample_step(
            self,
            model: torch.nn.Module,
            x_t: torch.Tensor,
            t: torch.Tensor,
            condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDPM denoising step from x_t to x_{t-1}.

        μ_θ = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t, c))
        x_{t-1} = μ_θ + √β̃_t · z,   z ~ N(0, I)   (z = 0 at t=0)

        :param model: Trained denoising model
        :param x_t: Noisy sample at timestep t, shape: (batch_size, embed_dim, D, H, W)
        :param t: Current timestep, shape: (batch_size,)
        :param condition: Terrain volume, shape: (batch_size, embed_dim, D, H, W)
        :return: Denoised volume x_{t-1}, shape: (batch_size, embed_dim, D, H, W)
        """
        predicted_noise = model(x_t, t, condition)
        betas_t = self._gather(self.betas, t, x_t.ndim)
        sqrt_one_minus_t = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        sqrt_recip_alpha = (1.0 / (1.0 - betas_t).sqrt())

        # DDPM posterior mean: reconstruct x_{t-1} from predicted noise
        mean = sqrt_recip_alpha * (x_t - betas_t * predicted_noise / sqrt_one_minus_t)

        # No noise is added at the final step. Returning the mean gives a clear sample.
        if t[0] == 0:
            return mean

        variance = self._gather(self.posterior_variance, t, x_t.ndim)
        return mean + variance.sqrt() * torch.randn_like(x_t)


