"""Train joint latent/action flow matching on precomputed latent episodes.

    python scripts/train_joint_dit.py --latent-dir latents/train --val-latent-dir latents/val \
      --out-dir runs/task/joint_dit --n-history 8 --n-future 4

Modes: ``joint`` denoises both futures, ``world`` keeps actions clean, and
``inverse`` keeps future latents clean. The final expert future latent is the
goal condition; world mode masks it out. Use --dual-experts for Wan-style
high/low-noise experts. Multi-GPU training uses torchrun, for example:

    torchrun --standalone --nproc_per_node=4 scripts/train_joint_dit.py ...
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.latent_seq import LatentSeqDataset
from models.joint_dit import JointTrajectoryDiT, NoiseExpertDiT

JOINT, WORLD, INVERSE = 0, 1, 2
MODE_IDS = {"joint": JOINT, "world": WORLD, "inverse": INVERSE}


def masked_mean(err, mask):
    return (err * mask).sum() / mask.expand_as(err).sum().clamp_min(1.0)


def flow_step(
    model,
    batch,
    n_history,
    mode_ids,
    tau,
    object_alpha=20.0,
    tactile_alpha=1.0,
    action_weight=1.0,
    history_noise=0.0,
    goal_dropout=0.0,
):
    z = batch["z_seq"]
    actions = batch["actions"]
    z_hist, a_hist = z[:, :n_history], actions[:, :n_history]
    z_future, a_future = z[:, n_history:], actions[:, n_history:]
    if history_noise:
        z_hist = z_hist + torch.randn_like(z_hist) * history_noise
    b = len(z)
    tau_z = torch.where(mode_ids == INVERSE, torch.zeros_like(tau), tau)
    tau_a = torch.where(mode_ids == WORLD, torch.zeros_like(tau), tau)
    tz = tau_z.reshape(b, 1, 1, 1, 1, 1)
    ta = tau_a.reshape(b, 1, 1)
    noise_z, noise_a = torch.randn_like(z_future), torch.randn_like(a_future)
    z_in = z_future + tz * (noise_z - z_future)
    a_in = a_future + ta * (noise_a - a_future)
    target_z, target_a = noise_z - z_future, noise_a - a_future
    goal = z_future[:, -1]
    goal_mask = (mode_ids != WORLD).float()
    if goal_dropout:
        goal_mask = goal_mask * (torch.rand(b, device=z.device) >= goal_dropout).float()
    pred_z, pred_a = model(z_hist, a_hist, z_in, a_in, tau_z, tau_a, goal=goal, goal_mask=goal_mask)
    pred_z, pred_a = pred_z.float(), pred_a.float()

    active_z = (mode_ids != INVERSE).float().reshape(b, 1, 1, 1, 1, 1)
    active_a = (mode_ids != WORLD).float().reshape(b, 1, 1)
    obj = batch["obj_occ"][:, n_history:].to(pred_z.dtype)
    weights = 1.0 + (object_alpha - 1.0) * obj
    if "tac_occ" in batch and tactile_alpha != 1.0:
        tac = batch["tac_occ"][:, n_history:].to(pred_z.dtype)
        weights = weights * (1.0 + (tactile_alpha - 1.0) * tac)
    weights = weights.unsqueeze(2) * active_z
    latent_loss = masked_mean((pred_z - target_z).square(), weights)
    action_loss = masked_mean((pred_a - target_a).square(), active_a)

    z0 = z_in - tz * pred_z
    a0 = a_in - ta * pred_a
    z_mse = masked_mean((z0 - z_future).square(), active_z)
    a_mse = masked_mean((a0 - a_future).square(), active_a)
    return dict(
        loss=latent_loss + action_weight * action_loss,
        latent=latent_loss,
        action=action_loss,
        z_mse=z_mse,
        a_mse=a_mse,
    )


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def random_modes(modes, batch_size, device):
    ids = torch.tensor(modes, device=device)
    return ids[torch.randint(len(modes), (batch_size,), device=device)]


def validation_modes(modes, batch_size, device):
    return torch.tensor([modes[i % len(modes)] for i in range(batch_size)], device=device)


def add_metrics(total, metrics, weight=1.0):
    for key, value in metrics.items():
        total[key] += float(value) * weight


def validate(model, loader, args, modes, device):
    model.eval()
    total = {key: 0.0 for key in ("loss", "latent", "action", "z_mse", "a_mse")}
    count = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.val_iters:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            b = len(batch["z_seq"])
            mode_ids = validation_modes(modes, b, device)
            tau = (torch.arange(b, device=device, dtype=torch.float32) + 0.5) / b
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                metrics = flow_step(
                    model,
                    batch,
                    args.n_history,
                    mode_ids,
                    tau,
                    args.object_alpha,
                    args.tactile_alpha,
                    args.action_weight,
                )
            add_metrics(total, metrics)
            count += 1
    if args.distributed:
        keys = list(total)
        values = torch.tensor([total[key] for key in keys], device=device)
        counts = torch.tensor(count, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total, count = dict(zip(keys, values.tolist())), int(counts.item())
    return {key: value / max(count, 1) for key, value in total.items()}


def save_checkpoint(path, model, opt, sched, epoch, best, action_mean, action_std, args):
    torch.save(
        dict(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            scheduler_state_dict=sched.state_dict(),
            epoch=epoch,
            best_val=best,
            action_mean=action_mean,
            action_std=action_std,
            model_config=model.config,
            args=vars(args),
        ),
        path,
    )


def setup_distributed(args):
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))
    args.distributed = args.world_size > 1
    if not args.distributed:
        return torch.device(args.device)
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend)
    return device


def load_initial_weights(model, ckpt):
    state = ckpt["model_state_dict"]
    source_dual = "expert_config" in ckpt["model_config"]
    if isinstance(model, NoiseExpertDiT) and not source_dual:
        model.high.load_state_dict(state)
        model.low.load_state_dict(state)
    elif not isinstance(model, NoiseExpertDiT) and source_dual:
        model.load_state_dict({key.removeprefix("high."): value for key, value in state.items() if key.startswith("high.")})
    else:
        model.load_state_dict(state)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--val-latent-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-history", type=int, default=8)
    p.add_argument("--n-future", type=int, default=4)
    p.add_argument("--modes", nargs="+", choices=list(MODE_IDS), default=list(MODE_IDS))
    p.add_argument("--dual-experts", action="store_true", help="use separate high/low-noise DiT experts")
    p.add_argument("--expert-switch", type=float, default=0.5, help="flow time at which the experts switch")
    p.add_argument("--goal-dropout", type=float, default=0.0, help="probability of dropping goal conditioning")
    p.add_argument("--patch-size", type=int, nargs=3, default=(8, 8, 8), metavar=("PD", "PH", "PW"))
    p.add_argument("--temporal-patch", type=int, default=1)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=None)
    p.add_argument("--freq-dim", type=int, default=256)
    p.add_argument("--object-alpha", type=float, default=20.0)
    p.add_argument("--tactile-alpha", type=float, default=1.0)
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--history-noise", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--val-iters", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--init", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = setup_distributed(args)
    out = Path(args.out_dir)
    if args.rank == 0:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if args.distributed:
        dist.barrier()
    seq_len = args.n_history + args.n_future
    norm = out / "action_norm_params.npz"
    if args.rank == 0 and not norm.exists():
        mean, std = LatentSeqDataset(args.latent_dir, seq_len).compute_action_norm_params()
        np.savez(norm, action_mean=mean, action_std=std)
    if args.distributed:
        dist.barrier()
    norm = np.load(norm)
    amean, astd = norm["action_mean"], norm["action_std"]
    train_ds = LatentSeqDataset(args.latent_dir, seq_len, action_mean=amean, action_std=astd)
    val_ds = LatentSeqDataset(args.val_latent_dir, seq_len, action_mean=amean, action_std=astd)
    train_sampler = (
        DistributedSampler(train_ds, args.world_size, args.rank, shuffle=True, seed=args.seed, drop_last=True)
        if args.distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, args.world_size, args.rank, shuffle=False, drop_last=True) if args.distributed else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    sample = train_ds[0]["z_seq"]
    model_config = dict(
        latent_channels=sample.shape[1],
        action_dim=train_ds.action_dim,
        grid=tuple(sample.shape[-3:]),
        n_history=args.n_history,
        n_future=args.n_future,
        patch_size=args.patch_size,
        temporal_patch=args.temporal_patch,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        freq_dim=args.freq_dim,
    )
    raw = NoiseExpertDiT(model_config, args.expert_switch) if args.dual_experts else JointTrajectoryDiT(**model_config)
    raw = raw.to(device)
    raw.gradient_checkpointing = True
    model = torch.compile(raw) if args.compile else raw
    if args.distributed:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.iters, eta_min=args.min_lr)
    start, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        start, best = ckpt["epoch"] + 1, ckpt["best_val"]
    elif args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        load_initial_weights(raw, ckpt)
    modes = [MODE_IDS[name] for name in args.modes]
    if args.rank == 0:
        print(
            f"[model] {sum(p.numel() for p in raw.parameters()) / 1e6:.2f} M parameters, "
            f"{len(train_ds)} train and {len(val_ds)} val windows, modes {args.modes}, "
            f"experts {2 if args.dual_experts else 1}, world size {args.world_size}",
            flush=True,
        )
    for epoch in range(start, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        total = {key: 0.0 for key in ("loss", "latent", "action", "z_mse", "a_mse")}
        iterator = iter(train_loader)
        for _ in range(args.iters):
            for _ in range(args.grad_accum):
                batch, iterator = next_batch(train_loader, iterator)
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                b = len(batch["z_seq"])
                mode_ids = random_modes(modes, b, device)
                tau = torch.rand(b, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    metrics = flow_step(
                        model,
                        batch,
                        args.n_history,
                        mode_ids,
                        tau,
                        args.object_alpha,
                        args.tactile_alpha,
                        args.action_weight,
                        args.history_noise,
                        args.goal_dropout,
                    )
                (metrics["loss"] / args.grad_accum).backward()
                add_metrics(total, metrics, 1.0 / args.grad_accum)
            torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        train_metrics = {key: value / args.iters for key, value in total.items()}
        val_metrics = validate(model, val_loader, args, modes, device)
        best = min(best, val_metrics["loss"])
        if args.rank == 0:
            print(
                f"[ep {epoch}] train {train_metrics['loss']:.5f} "
                f"(z {train_metrics['latent']:.5f} a {train_metrics['action']:.5f}) "
                f"val {val_metrics['loss']:.5f} z0 {val_metrics['z_mse']:.5f} a0 {val_metrics['a_mse']:.5f} "
                f"lr {sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
            save_checkpoint(out / "checkpoints" / "latest.pth", raw, opt, sched, epoch, best, amean, astd, args)
            if val_metrics["loss"] == best:
                save_checkpoint(out / "checkpoints" / "best.pth", raw, opt, sched, epoch, best, amean, astd, args)
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(out / "checkpoints" / f"model_{epoch}.pth", raw, opt, sched, epoch, best, amean, astd, args)
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
