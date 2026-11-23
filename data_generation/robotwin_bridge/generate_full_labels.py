#!/usr/bin/env python3
"""Generate RoboTwin actor, gripper, and visible robot labels for SWAM full recipe.

The simulator is replayed with the episode seed to recover SAPIEN per-scene IDs.
Actor segmentation is then converted into SWAM object and gripper masks. Robot
voxel labels are built from the visible robot surfaces in the recorded depth
views, which matches the points available to the VAE input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import ImageColor


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--robotwin-root", type=Path, default=Path(__file__).parents[1] / "robotwin")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--native-root", type=Path, required=True, help="native RoboTwin episode directory")
    p.add_argument("--bounds", nargs=6, type=float, default=[-0.5, 0.5, -0.35, 0.35, -0.1, 0.4])
    a = p.parse_args()
    root = a.robotwin_root.resolve()
    dataset = a.dataset_root.resolve()
    native = a.native_root.resolve()
    seeds = [int(x) for x in (native / "seed.txt").read_text().split()]
    flat = np.asarray(a.bounds, np.float32)
    bounds = np.asarray(
        [[flat[0], flat[2], flat[4]], [flat[1], flat[3], flat[5]]],
        np.float32,
    ).tolist()
    for packed in sorted((dataset / "train").glob("episode_*.hdf5")) + sorted((dataset / "val").glob("episode_*.hdf5")):
        episode = int(packed.stem.split("_")[-1])
        source = native / "data" / f"episode_{episode:07d}.hdf5"
        if not source.exists():
            raise FileNotFoundError(source)
        process_episode(a.task, root, packed, source, seeds[episode], bounds)


if __name__ == "__main__":
    main()
