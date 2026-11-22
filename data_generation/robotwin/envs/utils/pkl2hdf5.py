import h5py, pickle
import json
import numpy as np
import os
import sys
import cv2
from pathlib import Path
from .images_to_video import images_to_video

# Collection writes stored bits only through encode_image_bit (via
# images_encoding). The local copy lives next to the dataset; prefer
# XPolicyLab.utils.process_data when that package is on the path.
_ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
if str(_ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOTWIN_ROOT))

from data.decode_image_bit import images_encoding


CAMERA_MAP = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
    "front_camera": "cam_third_view",
}


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def _ensure_2d(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    return values


def _to_4x4(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2 and values.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3, :4] = values
        return result
    if values.ndim == 3 and values.shape[1:] == (3, 4):
        result = np.repeat(np.eye(4, dtype=np.float32)[None], len(values), axis=0)
        result[:, :3, :4] = values
        return result
    return values


def create_xpolicylab_hdf5(data, hdf5_path, instructions, frequency):
    joints = data["joint_action"]
    frame_num = len(joints["left_arm"])
    if frame_num < 2:
        raise ValueError("At least two frames are required to create state/action pairs")

    instruction_values = [str(value) for value in (instructions or []) if str(value)]
    if not instruction_values:
        instruction_values = [""]

    with h5py.File(hdf5_path, "w") as f:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        f.attrs["source_format"] = "RoboTwin"
        f.attrs["source_path"] = "native_collection"
        f.create_dataset("data_format_version", data="v1.0", dtype=string_dtype)
        f.create_dataset(
            "instructions",
            data=json.dumps(instruction_values, ensure_ascii=False),
            dtype=string_dtype,
        )
        f.create_group("additional_info").create_dataset(
            "frequency", data=np.asarray(frequency, dtype=np.int32)
        )

        state = f.create_group("state")
        action = f.create_group("action")
        joint_fields = [
            ("left_arm", "left_arm_joint_states"),
            ("left_gripper", "left_ee_joint_states"),
            ("right_arm", "right_arm_joint_states"),
            ("right_gripper", "right_ee_joint_states"),
        ]
        for source_name, target_name in joint_fields:
            if source_name not in joints:
                continue
            values = _ensure_2d(joints[source_name])
            state.create_dataset(target_name, data=values[:-1])
            action.create_dataset(target_name, data=values[1:])

        endpose = data.get("endpose", {})
        for source_name, target_name in [
            ("left_endpose", "left_ee_poses"),
            ("right_endpose", "right_ee_poses"),
        ]:
            if source_name not in endpose:
                continue
            values = _ensure_2d(endpose[source_name])
            state.create_dataset(target_name, data=values[:-1])
            action.create_dataset(target_name, data=values[1:])

        vision = f.create_group("vision")
        observations = data["observation"]
        for source_name, target_name in CAMERA_MAP.items():
            if source_name not in observations or "rgb" not in observations[source_name]:
                continue
            source_camera = observations[source_name]
            target_camera = vision.create_group(target_name)
            colors = np.asarray(source_camera["rgb"])[:-1]
            encoded_colors, max_len = images_encoding(colors)
            target_camera.create_dataset(
                "colors", data=encoded_colors, dtype=f"S{max_len}"
            )
            target_camera.create_dataset(
                "shape", data=np.asarray(colors[0].shape, dtype=np.int32)
            )

            if "depth" in source_camera:
                target_camera.create_dataset(
                    "depths", data=np.asarray(source_camera["depth"])[:-1]
                )

            for segmentation_name in ("actor_segmentation", "mesh_segmentation"):
                if segmentation_name in source_camera:
                    target_camera.create_dataset(
                        segmentation_name,
                        data=np.asarray(source_camera[segmentation_name])[:-1],
                        compression="gzip",
                    )

            intrinsic = source_camera.get(
                "intrinsic_cv", source_camera.get("intrinsic_matrix")
            )
            if intrinsic is not None:
                target_camera.create_dataset(
                    "intrinsic_matrix", data=np.asarray(intrinsic)[:-1]
                )

            extrinsic = source_camera.get("cam2world_gl")
            if extrinsic is None:
                extrinsic = source_camera.get("extrinsic_cv")
            if extrinsic is None:
                extrinsic = source_camera.get("extrinsics_matrix")
            if extrinsic is not None:
                target_camera.create_dataset(
                    "extrinsics_matrix", data=_to_4x4(extrinsic)[:-1]
                )

    return frame_num - 1


def pkl_files_to_hdf5_and_video(
    pkl_files,
    hdf5_path,
    video_path,
    *,
    instructions=None,
    frequency=15,
    rgb_debug=False,
):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path)
    if rgb_debug:
        debug_path = str(Path(video_path).with_name(Path(video_path).stem + "_rgb_debug.mp4"))
        write_rgb_debug_video(data_list, debug_path, frequency)
    return create_xpolicylab_hdf5(data_list, hdf5_path, instructions, frequency)


def write_rgb_debug_video(data, output_path, frequency):
    """Write normal RGB beside actor-level mesh segmentation for diagnosis."""
    camera = data["observation"]["head_camera"]
    rgb = np.asarray(camera["rgb"])
    segmentation = camera.get("actor_segmentation")
    if segmentation is None:
        raise KeyError("rgb_debug requires actor_segmentation in the collected observations")
    segmentation = np.asarray(segmentation)
    if len(rgb) != len(segmentation):
        raise ValueError("RGB and actor segmentation frame counts differ")
    h, w = rgb.shape[1:3]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(frequency),
        (w * 2, h),
    )
    try:
        for color, mesh in zip(rgb, segmentation):
            left = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_RGB2BGR)
            right = cv2.cvtColor(mesh.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.rectangle(left, (0, 0), (w, 24), (20, 20, 20), -1)
            cv2.rectangle(right, (0, 0), (w, 24), (20, 20, 20), -1)
            cv2.putText(left, "RGB", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(right, "ACTOR MESH", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(np.concatenate([left, right], axis=1))
    finally:
        writer.release()


def process_folder_to_hdf5_video(
    folder_path,
    hdf5_path,
    video_path,
    *,
    instructions=None,
    frequency=15,
    rgb_debug=False,
):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    return pkl_files_to_hdf5_and_video(
        pkl_files,
        hdf5_path,
        video_path,
        instructions=instructions,
        frequency=frequency,
        rgb_debug=rgb_debug,
    )
