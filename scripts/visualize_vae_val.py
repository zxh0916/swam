#!/usr/bin/env python3
"""Render a VAE reconstruction against RGB views in a packed episode."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.aloha_packed import AlohaPackedPCDataset
from models.background_bank import BackgroundGaussians
from models.big_head import BigHeadRoundShim
from models.build import load_vae, normalize_latent
from models.class_seeds import class_masks, select_class_seeds
from models.gsplat_render_loss import concat_gaussians, render_gaussians
from models.world_model import orbit_viewmat


def encode(model, xyz, rgb, pos_mean, pos_std):
    z = model.encode(xyz, rgb, pos_mean=pos_mean, pos_std=pos_std)[0]
    return normalize_latent(z, model.latent_split)


def load_head(ck, device):
    spec = ck.get("new_head", {})
    head = BigHeadRoundShim(feat_dim=64, K=int(spec.get("children", 16)), shape=spec.get("shape", "aniso"), cap=spec.get("cap", 3.0)).to(device).eval()
    state = ck.get("seed_head_state_dict")
    if state is None and spec.get("state_dict") is not None:
        state = {f"head.{k}": v for k, v in spec["state_dict"].items()}
    if state is None:
        raise KeyError("checkpoint has neither seed_head_state_dict nor new_head.state_dict")
    head.load_state_dict(state)
    return head


def load_background(ck, device):
    state = ck.get("bg_bank_state_dict")
    if not state:
        return None
    bank = BackgroundGaussians(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)).to(device)
    bank.load_resized({k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()})
    return bank.gaussians()


def decode_gaussians(model, head, features, z, xyz_world, max_seeds):
    with torch.no_grad():
        occ_mask = model.encoder._compute_bottleneck_occ_mask(xyz_world.reshape(-1, 3), batch_size=1)
        out = model.decode(z, occ_mask_0=occ_mask, use_gt_mask=True)
        level = len(model.grid_shapes) - 1
        masks = class_masks(out[f"scene_occ_{level}"], out[f"tac_occ_{level}"], out[f"signal_{level}"])
        seeds, seed_cls = select_class_seeds(masks, max_seeds=max_seeds)
        if seeds.shape[0] == 0:
            raise RuntimeError("decoder produced no occupied seed voxels")
        nx, ny, nz = model.grid_shapes[-1]
        bmin = model.bounds_min_world
        vsize = (model.bounds_max_world - bmin) / torch.tensor([nx, ny, nz], device=z.device, dtype=bmin.dtype)
        _, raw = head(features["feat_2"], seeds, bmin, vsize, seed_cls=seed_cls)
        return {k: raw[k].reshape(-1, *shape) for k, shape in {"xyz": (3,), "scale": (3,), "opacity": (), "rgb": (3,), "quat": (4,)}.items()}


def read_rgb(g, frame):
    image = cv2.imdecode(np.asarray(g["jpg"][frame], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode render jpg")
    return image[..., ::-1].copy()


def ood_viewmat(viewmat, center, yaw_deg):
    c2w = np.linalg.inv(viewmat).astype(np.float64)
    eye, forward = c2w[:3, 3], c2w[:3, 2]
    target = eye + forward * np.dot(center - eye, forward)
    offset = eye - target
    radius = np.linalg.norm(offset)
    az = np.rad2deg(np.arctan2(offset[1], offset[0])) + yaw_deg
    el = np.rad2deg(np.arcsin(offset[2] / radius))
    return orbit_viewmat(target, az, el, radius)


def render_view(gaussians, viewmat, K, w, h):
    with torch.no_grad():
        rendered, alpha = render_gaussians(gaussians["xyz"], gaussians["scale"], gaussians["opacity"], (gaussians["rgb"].clamp(-1, 1) + 1) * 0.5, viewmat[None], K[None], w, h, quats=gaussians.get("quat"))
    return (rendered[0] + (1.0 - alpha[0])).clamp(0, 1).mul(255).byte().cpu().numpy()


def labeled_row(images, labels):
    h, w = images[0].shape[:2]
    row = np.full((h + 24, w * len(images), 3), 255, dtype=np.uint8)
    for i, (label, image) in enumerate(zip(labels, images)):
        row[24:, i * w : (i + 1) * w] = image
        cv2.putText(row, label, (i * w + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--packed-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=100000)
    p.add_argument("--ood-yaw", type=float, nargs="+", default=[], metavar="DEG", help="render extra views orbited around the head camera target")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-background", action="store_true")
    a = p.parse_args()
    device = torch.device(a.device)
    ck = torch.load(a.vae_ckpt, map_location=device, weights_only=False)
    model = load_vae(ck, device)
    head = load_head(ck, device)
    background = None if a.no_background else load_background(ck, device)
    features = {}
    model.decoder.up_finest.register_forward_hook(lambda _, __, out: features.__setitem__("feat_2", out))
    ds = AlohaPackedPCDataset(
        a.packed_dir,
        n_points=None,
        use_sor_filter=False,
        pos_mean=ck["pos_mean"],
        pos_std=ck["pos_std"],
    )
    chosen = [i for i, meta in enumerate(ds.ep_meta) if a.episode is None or meta["ep_id"] == a.episode]
    if not chosen:
        raise SystemExit(f"episode {a.episode} was not found under {a.packed_dir}")
    ei, frame = chosen[0], int(a.frame)
    indices = [i for i, (e, t) in enumerate(ds.index) if e == ei and t == frame]
    if not indices:
        raise SystemExit(f"frame {frame} was not found in episode {ds.ep_meta[ei]['ep_id']}")
    sample = ds[indices[0]]
    xyz, rgb = sample["xyz"].to(device)[None], sample["rgb"].to(device)[None]
    pos_mean, pos_std = ck["pos_mean"], ck["pos_std"]
    xyz_world = xyz * torch.as_tensor(pos_std, device=device) + torch.as_tensor(pos_mean, device=device)
    with torch.no_grad():
        z = encode(model, xyz, rgb, pos_mean, pos_std)
        fg = decode_gaussians(model, head, features, z, xyz_world, a.max_seeds)
        gaussians = concat_gaussians(fg, background) if background is not None else fg
    f = ds._h(ei)
    cam_ids = [int(c) for c in f.attrs.get("render_cam_ids", f.attrs["cam_ids"])]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_id, rows, written = ds.ep_meta[ei]["ep_id"], [], 0
    for cid in cam_ids:
        g = f.get(f"render/camera_{cid}")
        if g is None:
            continue
        vm = np.asarray(g["viewmat"][frame], dtype=np.float32)
        if not np.isfinite(vm).all():
            continue
        try:
            gt = read_rgb(g, frame)
        except ValueError:
            continue
        h, w = gt.shape[:2]
        K = torch.as_tensor(g["K"][frame], device=device, dtype=torch.float32)
        viewmat = torch.as_tensor(vm, device=device, dtype=torch.float32)
        pred = render_view(gaussians, viewmat, K, w, h)
        diff = np.abs(gt.astype(np.int16) - pred.astype(np.int16)).astype(np.uint8)
        rows.append(([gt, pred, diff], [f"GT cam {cid}", f"Render cam {cid}", f"Diff cam {cid} | MAE {diff.mean():.2f}"]))
        written += 1
    if a.ood_yaw:
        base_cid = 0 if 0 in cam_ids else cam_ids[0]
        g = f[f"render/camera_{base_cid}"]
        gt = read_rgb(g, frame)
        h, w = gt.shape[:2]
        K = torch.as_tensor(g["K"][frame], device=device, dtype=torch.float32)
        vm = np.asarray(g["viewmat"][frame], dtype=np.float32)
        center = ((model.bounds_min_world + model.bounds_max_world) / 2).cpu().numpy()
        ood_images = []
        for yaw in a.ood_yaw:
            viewmat = torch.as_tensor(ood_viewmat(vm, center, yaw), device=device)
            ood_images.append(render_view(gaussians, viewmat, K, w, h))
        rows.append(([gt, *ood_images], [f"GT cam {base_cid} reference", *[f"OOD yaw {yaw:+g} deg" for yaw in a.ood_yaw]]))
    rendered_rows = [labeled_row(images, labels) for images, labels in rows]
    width = max(row.shape[1] for row in rendered_rows)
    for i, row in enumerate(rendered_rows):
        if row.shape[1] < width:
            padded = np.full((row.shape[0], width, 3), 255, dtype=np.uint8)
            padded[:, : row.shape[1]] = row
            rendered_rows[i] = padded
    comparison = np.concatenate(rendered_rows, axis=0)
    output_path = out_dir / f"episode_{episode_id}_frame_{frame:04d}_comparison.png"
    cv2.imwrite(str(output_path), comparison[..., ::-1])
    print(f"wrote {written} camera comparisons and {len(a.ood_yaw)} OOD views to {output_path}")


if __name__ == "__main__":
    main()
