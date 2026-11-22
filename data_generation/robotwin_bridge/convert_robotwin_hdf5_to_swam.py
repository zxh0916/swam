#!/usr/bin/env python3
"""Convert native RoboTwin HDF5 episodes to SWAM packed_v2 episodes.

The adapter keeps the native files untouched.  It converts millimetre depth to
metres, changes SAPIEN OpenGL camera poses to SWAM's +Z camera convention, and
adds the render group required by ``swam.datasets.frames.Frames``.  It is
intended for the SWAM VAE ``--recipe minimal`` path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
GL_TO_SWAM = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def vlen_bytes(parent: h5py.Group, name: str, rows: list[np.ndarray]) -> None:
    ds = parent.create_dataset(name, shape=(len(rows),), dtype=h5py.vlen_dtype(np.uint8))
    for i, row in enumerate(rows):
        ds[i] = np.asarray(row, dtype=np.uint8).reshape(-1)


def jpeg_from_rgb(rgb: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("failed to encode RGB frame")
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


def decode_native(bit: np.ndarray) -> np.ndarray:
    if isinstance(bit, (bytes, bytearray)):
        encoded = np.frombuffer(bit, dtype=np.uint8)
    else:
        encoded = np.asarray(bit, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode native RoboTwin RGB")
    # Native collection uses the XPolicyLab RGB marker.  The screening files
    # are standard RGB JPEGs, so one BGR->RGB conversion is correct here.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_episode(src: Path, dst: Path, episode_id: int, origin: np.ndarray, bbox: np.ndarray) -> None:
    with h5py.File(src, "r") as inp, h5py.File(dst, "w") as out:
        head = inp["vision/cam_head/colors"]
        T = len(head)
        sample = decode_native(head[0])
        H, W = sample.shape[:2]
        out.attrs.update(
            format_version="packed_v2",
            dataset="robotwin",
            episode=int(episode_id),
            src_episode=str(src),
            num_frames=T,
            H=H,
            W=W,
            cam_ids=np.asarray([0, 1, 2], np.int32),
            render_cam_ids=np.asarray([0, 1, 2], np.int32),
            n_arms=2,
            has_obj_mask=False,
            bbox=bbox.astype(np.float32),
            depth_unit="metre",
            source_depth_unit="millimetre",
        )
        img = out.create_group("img")
        cam = out.create_group("cam")
        render = out.create_group("render")
        for cid, cname in enumerate(CAMERAS):
            src_cam = inp[f"vision/{cname}"]
            rgb = [decode_native(src_cam["colors"][t]) for t in range(T)]
            depth = np.asarray(src_cam["depths"], dtype=np.float32) / 1000.0
            valid = np.isfinite(depth) & (depth > 0)
            K = np.asarray(src_cam["intrinsic_matrix"], dtype=np.float32)
            E_gl = np.asarray(src_cam["extrinsics_matrix"], dtype=np.float32)
            E = np.matmul(E_gl, GL_TO_SWAM)
            E[:, :3, 3] -= origin[None]
            V = np.linalg.inv(E).astype(np.float32)

            ig = img.create_group(f"camera_{cid}")
            ig.create_dataset("depth", data=depth.astype(np.float16), compression="gzip")
            ig.create_dataset("valid_mask", data=np.packbits(valid.reshape(T, -1), axis=1), compression="gzip")
            vlen_bytes(ig, "rgb_in_ir", [jpeg_from_rgb(frame) for frame in rgb])

            cg = cam.create_group(f"camera_{cid}")
            cg.create_dataset("K_ir", data=K)
            cg.create_dataset("E_cam2world", data=E)

            rg = render.create_group(f"camera_{cid}")
            rg.attrs.update(H=H, W=W, crop=np.asarray([0, 0, W, H], np.int32))
            vlen_bytes(rg, "jpg", [jpeg_from_rgb(frame) for frame in rgb])
            rg.create_dataset("K", data=K)
            rg.create_dataset("viewmat", data=V)
            rg.create_dataset("valid_mask", data=np.packbits(valid.reshape(T, -1), axis=1), compression="gzip")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path, help="native RoboTwin episode HDF5 files")
    p.add_argument("--output", type=Path, required=True, help="SWAM dataset root")
    p.add_argument("--val-count", type=int, default=1)
    p.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.74], metavar=("X", "Y", "Z"))
    p.add_argument("--bbox", nargs=6, type=float, default=[-0.65, 0.65, -0.45, 0.45, -0.15, 0.85], metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    a = p.parse_args()
    files = sorted(Path(x).resolve() for x in a.inputs)
    if not files:
        raise SystemExit("no input episodes")
    out = a.output.resolve()
    train = out / "train"
    val = out / "val"
    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(a.origin, np.float32)
    bbox = np.asarray(a.bbox, np.float32)
    val_count = min(max(a.val_count, 0), len(files))
    split = [(val if i >= len(files) - val_count else train) for i in range(len(files))]
    for i, (src, part) in enumerate(zip(files, split)):
        dst = part / f"episode_{i}.hdf5"
        convert_episode(src, dst, i, origin, bbox)
        print(dst)


if __name__ == "__main__":
    main()
