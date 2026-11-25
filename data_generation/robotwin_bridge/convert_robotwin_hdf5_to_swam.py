#!/usr/bin/env python3
"""Convert native RoboTwin HDF5 episodes to SWAM full-recipe packed_v3 episodes.

The adapter keeps the native files untouched.  It converts millimetre depth to
metres, changes SAPIEN OpenGL camera poses to SWAM's +Z camera convention, and
adds the render group required by ``swam.datasets.frames.Frames``.  Full-recipe
object, robot, and gripper labels are generated in the same command.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from PIL import ImageColor


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


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w = np.clip(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:]
    angle = 2.0 * np.arccos(w)
    scale = angle / np.maximum(np.linalg.norm(xyz, axis=-1), 1e-8)
    return xyz * scale[..., None]


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
            # The bank init reads full-scene depth; native RoboTwin depth is
            # the whole SAPIEN scene with 0 = hole.
            depth_semantics="full_scene_lrc",
        )
        img = out.create_group("img")
        cam = out.create_group("cam")
        render = out.create_group("render")
        action_group = out.create_group("action")
        left_pose = np.asarray(inp["action/left_ee_poses"], dtype=np.float32)
        right_pose = np.asarray(inp["action/right_ee_poses"], dtype=np.float32)
        left_grip = np.asarray(inp["action/left_ee_joint_states"], dtype=np.float32).reshape(-1, 1)
        right_grip = np.asarray(inp["action/right_ee_joint_states"], dtype=np.float32).reshape(-1, 1)
        action_group.create_dataset(
            "cartesian_action",
            data=np.concatenate(
                [left_pose[:, :3], quat_to_rotvec(left_pose[:, 3:]), left_grip,
                 right_pose[:, :3], quat_to_rotvec(right_pose[:, 3:]), right_grip], axis=1
            ).astype(np.float32),
        )
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

            # SWAM's valid mask controls which depth pixels become training
            # points. Keep only points inside the configured world workspace.
            yy, xx = np.indices((H, W), dtype=np.float32)
            z = depth
            x = (xx[None] - K[:, None, None, 0, 2]) * z / K[:, None, None, 0, 0]
            y = (yy[None] - K[:, None, None, 1, 2]) * z / K[:, None, None, 1, 1]
            cam_points = np.stack([x, y, z], axis=-1)
            world_points = np.einsum("thwc,tdc->thwd", cam_points, E[:, :3, :3]) + E[:, None, None, :3, 3]
            valid &= np.all(
                (world_points >= bbox[[0, 2, 4]]) &
                (world_points <= bbox[[1, 3, 5]]),
                axis=-1,
            )

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


def convert_main(inputs: list[Path], output: Path, val_count: int, episode_offset: int,
                 origin: np.ndarray, bbox: np.ndarray) -> None:
    files = sorted(Path(x).resolve() for x in inputs)
    if not files:
        raise SystemExit("no input episodes")
    out = output.resolve()
    train = out / "train"
    val = out / "val"
    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)
    val_count = min(max(val_count, 0), len(files))
    split = [(val if i >= len(files) - val_count else train) for i in range(len(files))]
    for i, (src, part) in enumerate(zip(files, split)):
        episode_id = episode_offset + i
        dst = part / f"episode_{episode_id}.hdf5"
        convert_episode(src, dst, episode_id, origin, bbox)
        print(dst)


GRID_SHAPES = ((64, 48, 64), (128, 96, 128))  # x, y, z


def entity_of(actor):
    return getattr(actor, "actor", getattr(actor, "entity", actor))


def scene_id(actor) -> int:
    entity = entity_of(actor)
    return int(entity.get_per_scene_id())


def load_task(root: Path, task_name: str, seed: int):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from scripts.collect_data import class_decorator, get_embodiment_config

    config = yaml.safe_load((root / "env_cfg/task_config/demo_clean.yml").read_text())
    embodiments = yaml.safe_load((root / "env_cfg/task_config/_embodiment_config.yml").read_text())
    robot_file = (root / embodiments["aloha-agilex"]["file_path"]).resolve()
    embodiment = get_embodiment_config(str(robot_file))
    config.update(
        task_name=task_name,
        task_config="demo_clean",
        save_path=str(root / "data"),
        seed=int(seed),
        need_plan=False,
        save_data=False,
        render_freq=0,
        left_robot_file=str(robot_file),
        right_robot_file=str(robot_file),
        left_embodiment_config=embodiment,
        right_embodiment_config=embodiment,
        dual_arm_embodied=True,
        embodiment_name="aloha-agilex",
    )
    task = class_decorator(task_name)
    task.setup_demo(now_ep_num=0, **config)
    return task


def targets(task_name: str, task):
    if task_name == "dump_bin_bigbin":
        return [(task.deskbin, "deskbin")] + [(x, "garbage") for x in task.sphere_lst]
    if task_name == "put_bottles_dustbin":
        return [(x, f"bottle_{i}") for i, x in enumerate(task.bottles)]
    if task_name == "put_object_cabinet":
        return [(task.object, "object")]
    if task_name == "place_can_basket":
        return [(task.can, "can"), (task.basket, "basket")]
    raise ValueError(task_name)


def link_ids(task):
    robot_ids, gripper_ids = set(), set()
    for entity in (task.robot.left_entity, task.robot.right_entity):
        for link in entity.get_links():
            name = link.get_name()
            if not hasattr(link, "get_entity"):
                continue
            sid = int(link.get_entity().get_per_scene_id())
            robot_ids.add(sid)
            if "link7" in name or "link8" in name:
                gripper_ids.add(sid)
    return robot_ids, gripper_ids


def pack_rows(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.reshape(mask.shape[0], -1).astype(np.uint8), axis=1)


def voxel_rows(points_per_frame, bounds, shape):
    lo = np.asarray(bounds[0], np.float32)
    hi = np.asarray(bounds[1], np.float32)
    shape = np.asarray(shape, np.int32)
    rows = []
    for points in points_per_frame:
        grid = np.zeros(tuple(shape[::-1]), dtype=bool)
        if len(points):
            uvw = np.floor((points - lo) / (hi - lo) * shape).astype(np.int32)
            keep = np.all((uvw >= 0) & (uvw < shape), axis=1)
            uvw = uvw[keep]
            if len(uvw):
                grid[uvw[:, 2], uvw[:, 1], uvw[:, 0]] = True
        rows.append(np.packbits(grid.reshape(-1).astype(np.uint8)))
    return np.stack(rows)


def process_episode(task_name: str, root: Path, packed: Path, source: Path, seed: int, bounds):
    task = load_task(root, task_name, seed)
    try:
        object_targets = targets(task_name, task)
        robot_ids, gripper_ids = link_ids(task)
        palette = np.asarray([ImageColor.getrgb(c) for c in sorted(set(ImageColor.colormap.values()))], dtype=np.uint8)
        object_ids = [(scene_id(actor), name) for actor, name in object_targets]

        with h5py.File(source, "r") as raw, h5py.File(packed, "r+") as out:
            T = int(out.attrs["num_frames"])
            H, W = int(out.attrs["H"]), int(out.attrs["W"])
            object_masks = []
            robot_points = [[] for _ in range(T)]
            gripper_points = [[] for _ in range(T)]
            render_gripper = {}
            render_object = {}
            for cid, (raw_name, cam_name) in enumerate(zip(CAMERAS, CAMERA_KEYS)):
                seg = np.asarray(raw[f"vision/{raw_name}/actor_segmentation"][:T], dtype=np.uint8)
                obj = np.zeros((T, H, W), dtype=np.uint8)
                rob = np.zeros((T, H, W), dtype=bool)
                grip = np.zeros((T, H, W), dtype=bool)
                for label, (sid, _) in enumerate(object_ids, start=1):
                    if sid < len(palette):
                        obj[np.all(seg == palette[sid], axis=-1)] = label
                for sid in robot_ids:
                    if sid < len(palette):
                        rob |= np.all(seg == palette[sid], axis=-1)
                for sid in gripper_ids:
                    if sid < len(palette):
                        grip |= np.all(seg == palette[sid], axis=-1)
                object_masks.append(obj)
                render_object[cid] = pack_rows(obj > 0)
                render_gripper[cid] = pack_rows(grip)

                depth = np.asarray(out[f"img/camera_{cid}/depth"][:T], dtype=np.float32)
                K = np.asarray(out[f"cam/camera_{cid}/K_ir"][:T], dtype=np.float32)
                E = np.asarray(out[f"cam/camera_{cid}/E_cam2world"][:T], dtype=np.float32)
                yy, xx = np.indices((H, W), dtype=np.float32)
                for t in range(T):
                    m = rob[t] & np.isfinite(depth[t]) & (depth[t] > 0)
                    if not m.any():
                        continue
                    v, u = np.nonzero(m)
                    z = depth[t, v, u]
                    x = (u.astype(np.float32) - K[t, 0, 2]) * z / K[t, 0, 0]
                    y = (v.astype(np.float32) - K[t, 1, 2]) * z / K[t, 1, 1]
                    pts = np.stack([x, y, z], axis=1) @ E[t, :3, :3].T + E[t, :3, 3]
                    robot_points[t].append(pts)
                    mg = grip[t, v, u]
                    if mg.any():
                        gripper_points[t].append(pts[mg])

                rg = out[f"render/camera_{cid}"]
                if "obj_mask" in rg:
                    del rg["obj_mask"]
                if "gripper_mask" in rg:
                    del rg["gripper_mask"]
                rg.create_dataset("obj_mask", data=render_object[cid], compression="gzip", chunks=(1, render_object[cid].shape[1]))
                rg.create_dataset("gripper_mask", data=render_gripper[cid], compression="gzip", chunks=(1, render_gripper[cid].shape[1]))
                ig = out[f"img/camera_{cid}"]
                if "obj_mask" in ig:
                    del ig["obj_mask"]
                ig.create_dataset("obj_mask", data=object_masks[cid], compression="gzip", chunks=(1, H, W))

            robot_points = [np.concatenate(x, axis=0) if x else np.zeros((0, 3), np.float32) for x in robot_points]
            gripper_points = [np.concatenate(x, axis=0) if x else np.zeros((0, 3), np.float32) for x in gripper_points]
            if "labels" in out:
                del out["labels"]
            labels = out.create_group("labels")
            labels.attrs["bounds"] = np.asarray(bounds, np.float32)
            robot_group = labels.create_group("robot")
            gripper_group = labels.create_group("gripper")
            for level, shape in enumerate(GRID_SHAPES):
                zyx = tuple(reversed(shape))
                r = voxel_rows(robot_points, bounds, shape)
                g = voxel_rows(gripper_points, bounds, shape)
                dr = robot_group.create_dataset(str(level), data=r, compression="gzip", chunks=(1, r.shape[1]))
                dg = gripper_group.create_dataset(str(level), data=g, compression="gzip", chunks=(1, g.shape[1]))
                dr.attrs["shape"] = zyx
                dg.attrs["shape"] = zyx
            out.attrs["has_obj_mask"] = True
            out.attrs["obj_mask_kind"] = "RoboTwin actor segmentation"
            out.attrs["format_version"] = "packed_v3"
            out.attrs["robot_label_source"] = "visible SAPIEN actor segmentation depth lift"
            out.attrs["gripper_label_source"] = "SAPIEN gripper link segmentation depth lift"
        print(packed)
    finally:
        task.close_env(clear_cache=False)


def main() -> None:
    """Convert native RoboTwin episodes and add all full-recipe labels."""
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="RoboTwin task name")
    p.add_argument("--robotwin-root", type=Path, default=Path(__file__).parents[1] / "robotwin")
    p.add_argument("--dataset-root", "--output", dest="dataset_root", type=Path, required=True, help="output SWAM dataset root")
    p.add_argument("--native-root", type=Path, help="native episode directory containing data/ and seed.txt")
    p.add_argument("inputs", nargs="*", type=Path, help="native episode HDF5 files, or use --native-root")
    p.add_argument("--val-count", type=int, default=1)
    p.add_argument("--episode-offset", type=int, default=0)
    p.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.74], metavar=("X", "Y", "Z"))
    p.add_argument("--bbox", nargs=6, type=float, default=[-0.5, 0.5, -0.35, 0.35, -0.1, 0.4], metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    a = p.parse_args()

    native_root = a.native_root.resolve() if a.native_root else None
    files = sorted(x.resolve() for x in a.inputs)
    if not files:
        if native_root is None:
            p.error("provide native HDF5 inputs or --native-root")
        files = sorted((native_root / "data").glob("episode_*.hdf5"))
    if native_root is None:
        native_root = files[0].parent.parent
    seeds = [int(x) for x in (native_root / "seed.txt").read_text().split()]
    if len(seeds) <= a.episode_offset + len(files) - 1:
        raise ValueError("seed.txt does not contain all requested episodes")

    origin = np.asarray(a.origin, np.float32)
    bbox = np.asarray(a.bbox, np.float32)
    convert_main(files, a.dataset_root, a.val_count, a.episode_offset, origin, bbox)
    bounds = [[bbox[0], bbox[2], bbox[4]], [bbox[1], bbox[3], bbox[5]]]
    val_count = min(max(a.val_count, 0), len(files))
    for i, source in enumerate(files):
        episode = a.episode_offset + i
        split = "val" if i >= len(files) - val_count else "train"
        packed = a.dataset_root.resolve() / split / f"episode_{episode}.hdf5"
        process_episode(a.task, a.robotwin_root.resolve(), packed, source, seeds[episode], bounds)


if __name__ == "__main__":
    main()
