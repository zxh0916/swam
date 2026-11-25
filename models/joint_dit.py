"""Wan-style DiT for joint denoising of future 3D latents and robot actions."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = t.float()[:, None] * freqs[None]
    return torch.cat([angles.cos(), angles.sin()], -1)


def apply_rope(x, positions, freqs):
    outs = []
    for axis, chunk in enumerate(x.chunk(4, -1)):
        angles = freqs[axis][positions[..., axis]].unsqueeze(1)
        cos, sin = angles.cos().to(chunk.dtype), angles.sin().to(chunk.dtype)
        pairs = chunk.reshape(*chunk.shape[:-1], -1, 2)
        even, odd = pairs.unbind(-1)
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)
        outs.append(rotated)
    return torch.cat(outs, -1)


def rope_freqs(head_dim, grid_shape):
    pairs = head_dim // 8
    base = torch.arange(pairs, dtype=torch.float32) / pairs
    return tuple(10000 ** (-base)[None] * torch.arange(n, dtype=torch.float32)[:, None] for n in grid_shape)


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SelfAttention(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.proj = (nn.Linear(dim, dim) for _ in range(4))
        self.q_norm, self.k_norm = RMSNorm(dim), RMSNorm(dim)

    def forward(self, x, positions, freqs):
        b, n, c = x.shape
        q = self.q_norm(self.q(x)).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        k = self.k_norm(self.k(x)).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        v = self.v(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        q, k = apply_rope(q, positions, freqs), apply_rope(k, positions, freqs)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return self.proj(y)


class CrossAttention(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.proj = (nn.Linear(dim, dim) for _ in range(4))
        self.q_norm, self.k_norm = RMSNorm(dim), RMSNorm(dim)

    def forward(self, x, context, context_mask):
        b, n, c = x.shape
        m = context.shape[1]
        q = self.q_norm(self.q(x)).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        k = self.k_norm(self.k(context)).reshape(b, m, self.heads, c // self.heads).transpose(1, 2)
        v = self.v(context).reshape(b, m, self.heads, c // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        y = self.proj(y)
        return y * context_mask.to(y.dtype).reshape(b, 1, 1)


class DiTBlock(nn.Module):

    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, elementwise_affine=False), nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.cross_attn = CrossAttention(dim, heads)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 1, 6, dim) / math.sqrt(dim))

    @staticmethod
    def modulate(x, shift, scale):
        return (x.float() * (1 + scale) + shift).to(x.dtype)

    def forward(self, x, e, positions, freqs, context=None, context_mask=None):
        e = tuple(v.squeeze(2) for v in (self.modulation + e.float()).chunk(6, 2))
        y = self.attn(self.modulate(self.norm1(x), e[0], e[1]), positions, freqs)
        x = x + y * e[2].to(y.dtype)
        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context, context_mask)
        y = self.ffn(self.modulate(self.norm2(x), e[3], e[4]))
        return x + y * e[5].to(y.dtype)


class ModulatedHead(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 1, 2, dim) / math.sqrt(dim))

    def forward(self, x, e):
        shift, scale = (v.squeeze(2) for v in (self.modulation + e.float()).chunk(2, 2))
        x = DiTBlock.modulate(self.norm(x), shift, scale)
        return self.proj(x)


class JointTrajectoryDiT(nn.Module):
    """Denoise a fixed-length latent/action clip conditional on clean history.

    Latents are (B,T,C,D,H,W), actions are (B,T,A), and each modality has its
    own flow time. A zero flow time marks a clean conditioning modality. An
    optional goal latent (B,C,D,H,W) is read by cross-attention; a zero goal
    mask disables that conditioning per sample.
    """

    def __init__(
        self,
        latent_channels,
        action_dim,
        grid,
        n_history=8,
        n_future=4,
        patch_size=(8, 8, 8),
        temporal_patch=1,
        dim=512,
        depth=12,
        heads=8,
        ffn_dim=None,
        freq_dim=256,
    ):
        super().__init__()
        grid, patch_size = tuple(grid), tuple(patch_size)
        ffn_dim = ffn_dim or 4 * dim
        if len(grid) != 3 or len(patch_size) != 3:
            raise ValueError("grid and patch_size are spatial (D,H,W) tuples")
        if any(g % p for g, p in zip(grid, patch_size)):
            raise ValueError("Every latent grid dimension must be divisible by patch_size")
        if n_history % temporal_patch or n_future % temporal_patch:
            raise ValueError("History and future lengths must be divisible by temporal_patch")
        if dim % heads or (dim // heads) % 8:
            raise ValueError("dim/heads must be divisible by 8 for four-axis RoPE")
        if freq_dim % 2:
            raise ValueError("freq_dim must be even")
        self.config = dict(
            latent_channels=latent_channels,
            action_dim=action_dim,
            grid=grid,
            n_history=n_history,
            n_future=n_future,
            patch_size=patch_size,
            temporal_patch=temporal_patch,
            dim=dim,
            depth=depth,
            heads=heads,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
        )
        self.C, self.A, self.grid = latent_channels, action_dim, grid
        self.n_history, self.n_future = n_history, n_future
        self.patch_size, self.temporal_patch = patch_size, temporal_patch
        self.hist_patches, self.future_patches = n_history // temporal_patch, n_future // temporal_patch
        self.time_patches = self.hist_patches + self.future_patches
        self.patch_grid = tuple(g // p for g, p in zip(grid, patch_size))
        self.spatial_tokens = math.prod(self.patch_grid)
        self.dim = dim
        self.gradient_checkpointing = False

        self.patch_embedding = nn.Conv3d(latent_channels * temporal_patch, dim, patch_size, patch_size)
        self.goal_embedding = nn.Conv3d(latent_channels, dim, patch_size, patch_size)
        self.goal_position = nn.Parameter(torch.randn(1, self.spatial_tokens, dim) * 0.02)
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim * temporal_patch, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.modality_embedding = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.clean_embedding = nn.Embedding(2, dim)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.head_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, ffn_dim) for _ in range(depth)])
        patch_voxels = math.prod(patch_size)
        self.latent_head = ModulatedHead(dim, latent_channels * temporal_patch * patch_voxels)
        self.action_head = ModulatedHead(dim, action_dim * temporal_patch)

        tt, zz, yy, xx = torch.meshgrid(
            torch.arange(self.time_patches),
            torch.arange(self.patch_grid[0]),
            torch.arange(self.patch_grid[1]),
            torch.arange(self.patch_grid[2]),
            indexing="ij",
        )
        z_pos = torch.stack((tt, zz, yy, xx), -1).reshape(1, -1, 4)
        a_pos = torch.zeros(self.time_patches, 4, dtype=torch.long)
        a_pos[:, 0] = torch.arange(self.time_patches)
        self.register_buffer("positions", torch.cat([z_pos, a_pos[None]], 1), persistent=False)
        for axis, n in enumerate((self.time_patches, *self.patch_grid)):
            self.register_buffer(f"rope_{axis}", rope_freqs(dim // heads, (n,) * 4)[axis], persistent=False)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.patch_embedding.bias)
        nn.init.xavier_uniform_(self.goal_embedding.weight.flatten(1))
        nn.init.zeros_(self.goal_embedding.bias)
        nn.init.zeros_(self.latent_head.proj.weight)
        nn.init.zeros_(self.latent_head.proj.bias)
        nn.init.zeros_(self.action_head.proj.weight)
        nn.init.zeros_(self.action_head.proj.bias)

    def embed_inputs(self, z, actions):
        b, t, c, d, h, w = z.shape
        pt = self.temporal_patch
        tp = t // pt
        z = z.reshape(b, tp, pt, c, d, h, w).permute(0, 1, 3, 2, 4, 5, 6)
        z = self.patch_embedding(z.reshape(b * tp, c * pt, d, h, w))
        z = z.reshape(b, tp, self.dim, *self.patch_grid).permute(0, 2, 1, 3, 4, 5)
        z_tokens = z.flatten(2).transpose(1, 2)
        a_tokens = self.action_embedding(actions.reshape(b, tp, pt * self.A))
        tokens = torch.cat([z_tokens, a_tokens], 1)
        modality = torch.cat(
            [self.modality_embedding[0:1].expand(z_tokens.shape[1], -1), self.modality_embedding[1:2].expand(tp, -1)]
        )
        return tokens + modality[None].to(tokens.dtype)

    def encode_goal(self, goal):
        if goal is None:
            return None
        tokens = self.goal_embedding(goal).flatten(2).transpose(1, 2)
        return tokens + self.goal_position.to(tokens.dtype)

    def unpatchify_latents(self, x):
        b = len(x)
        c, pt = self.C, self.temporal_patch
        pd, ph, pw = self.patch_size
        dp, hp, wp = self.patch_grid
        x = x.reshape(b, self.future_patches, dp, hp, wp, c, pt, pd, ph, pw)
        x = x.permute(0, 1, 6, 5, 2, 7, 3, 8, 4, 9)
        return x.reshape(b, self.n_future, c, *self.grid)

    def forward(self, z_hist, a_hist, z_future, a_future, tau_z, tau_a, goal=None, goal_mask=None):
        b = len(z_hist)
        z = torch.cat([z_hist, z_future], 1)
        actions = torch.cat([a_hist, a_future], 1)
        x = self.embed_inputs(z, actions)
        goal_tokens = self.encode_goal(goal)
        if goal_tokens is not None and goal_mask is None:
            goal_mask = torch.ones(b, device=z.device)
        zeros_z = tau_z.new_zeros(b, self.hist_patches)
        zeros_a = tau_a.new_zeros(b, self.hist_patches)
        tau_z_tokens = torch.cat([zeros_z, tau_z[:, None].expand(b, self.future_patches)], 1)
        tau_a_tokens = torch.cat([zeros_a, tau_a[:, None].expand(b, self.future_patches)], 1)
        tau_z_tokens = tau_z_tokens.repeat_interleave(self.spatial_tokens, 1)
        tau_tokens = torch.cat([tau_z_tokens, tau_a_tokens], 1)
        flat_tau = tau_tokens.flatten()
        noise_emb = self.time_embedding(timestep_embedding(flat_tau * 1000, self.time_embedding[0].in_features))
        noise_emb = noise_emb.reshape(b, -1, self.dim)
        x = x + self.clean_embedding((tau_tokens == 0).long()).to(x.dtype)
        e = self.head_projection(noise_emb).unflatten(2, (2, self.dim))
        modulation = self.time_projection(noise_emb).unflatten(2, (6, self.dim))
        freqs = tuple(getattr(self, f"rope_{axis}") for axis in range(4))
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, modulation, self.positions, freqs, goal_tokens, goal_mask, use_reentrant=False)
            else:
                x = block(x, modulation, self.positions, freqs, goal_tokens, goal_mask)

        z0 = self.hist_patches * self.spatial_tokens
        z1 = z0 + self.future_patches * self.spatial_tokens
        a0 = self.time_patches * self.spatial_tokens + self.hist_patches
        a1 = a0 + self.future_patches
        vz = self.latent_head(x[:, z0:z1], e[:, z0:z1])
        va = self.action_head(x[:, a0:a1], e[:, a0:a1])
        vz = self.unpatchify_latents(vz)
        va = va.reshape(b, self.n_future, self.A)
        return vz.float(), va.float()


class NoiseExpertDiT(nn.Module):
    """Route each sample to a high-noise or low-noise JointTrajectoryDiT."""

    def __init__(self, expert_config, switch_tau=0.5):
        super().__init__()
        self.high = JointTrajectoryDiT(**expert_config)
        self.low = JointTrajectoryDiT(**expert_config)
        self.switch_tau = float(switch_tau)
        self.config = dict(expert_config=dict(expert_config), switch_tau=self.switch_tau)
        self.C, self.A, self.grid = self.high.C, self.high.A, self.high.grid
        self.n_history, self.n_future = self.high.n_history, self.high.n_future

    @property
    def gradient_checkpointing(self):
        return self.high.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.high.gradient_checkpointing = value
        self.low.gradient_checkpointing = value

    def forward(self, z_hist, a_hist, z_future, a_future, tau_z, tau_a, goal=None, goal_mask=None):
        high = torch.maximum(tau_z, tau_a) >= self.switch_tau
        out_z = torch.empty_like(z_future, dtype=torch.float32)
        out_a = torch.empty_like(a_future, dtype=torch.float32)
        for mask, expert in ((high, self.high), (~high, self.low)):
            indices = mask.nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue
            vz, va = expert(
                z_hist[indices],
                a_hist[indices],
                z_future[indices],
                a_future[indices],
                tau_z[indices],
                tau_a[indices],
                goal=None if goal is None else goal[indices],
                goal_mask=None if goal_mask is None else goal_mask[indices],
            )
            out_z = out_z.index_copy(0, indices, vz)
            out_a = out_a.index_copy(0, indices, va)
        return out_z, out_a


@torch.no_grad()
def sample_trajectory(model, z_hist, a_hist, mode="joint", actions=None, latents=None, goal=None, steps=8):
    """Euler-sample a clip. ``world`` fixes actions; ``inverse`` fixes latents."""
    if mode not in ("joint", "world", "inverse"):
        raise ValueError("mode must be joint, world or inverse")
    if mode == "world" and actions is None:
        raise ValueError("world sampling requires normalized future actions")
    if mode == "inverse" and latents is None:
        raise ValueError("inverse sampling requires clean future latents")
    model.eval()
    device = z_hist.device
    b = len(z_hist)
    z = torch.randn(b, model.n_future, model.C, *model.grid, device=device, dtype=z_hist.dtype)
    a = torch.randn(b, model.n_future, model.A, device=device, dtype=a_hist.dtype)
    if mode == "world":
        a = actions.to(device=device, dtype=a.dtype).clone()
    if mode == "inverse":
        z = latents.to(device=device, dtype=z.dtype).clone()
    times = torch.linspace(1, 0, steps + 1, device=device)
    for i in range(steps):
        t = times[i]
        tau_z = t.new_zeros(b) if mode == "inverse" else t.expand(b)
        tau_a = t.new_zeros(b) if mode == "world" else t.expand(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            vz, va = model(z_hist, a_hist, z, a, tau_z, tau_a, goal=goal)
        dt = times[i + 1] - t
        if mode != "inverse":
            z = z + dt.to(z.dtype) * vz.to(z.dtype)
        if mode != "world":
            a = a + dt.to(a.dtype) * va.to(a.dtype)
    return z, a


def load_joint_dit(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    model = NoiseExpertDiT(**config) if "expert_config" in config else JointTrajectoryDiT(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
