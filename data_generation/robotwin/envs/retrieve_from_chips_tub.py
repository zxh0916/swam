"""Bimanual retrieval of one object from a chips tub."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC


class retrieve_from_chips_tub(Base_Task):
    CONTAINER_GLB = "/home/hyb/4dwm/pixal3d_generation/result/handled_utensil_container.glb"
    CONTAINER_SCALE = 0.22
    TARGET_NAME = "058_markpen"
    TARGET_MODEL_ID = 0
    TARGET_SCALE = [0.08, 0.08, 0.08]

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    @staticmethod
    def _configure_drive(drive):
        for setter in (
            drive.set_drive_property_x,
            drive.set_drive_property_y,
            drive.set_drive_property_z,
        ):
            setter(1000.0, 100.0, 1000.0)
        drive.set_drive_property_swing(1000.0, 100.0, 1000.0)
        drive.set_drive_property_twist(1000.0, 100.0, 1000.0)

    @staticmethod
    def _contact_matrix(direction_name, translation):
        desired_rotation = t3d.quaternions.quat2mat(
            np.asarray(GRASP_DIRECTION_DIC[direction_name], dtype=np.float64)
        )
        grasp_frame_rotation = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=np.float32)
        contact = np.eye(4, dtype=np.float32)
        contact[:3, :3] = desired_rotation @ grasp_frame_rotation.T
        contact[:3, 3] = translation
        return contact

    def load_actors(self):
        self._load_container()
        self._load_target()
        self.initial_support_drive = self.scene.create_drive(
            self.tub.actor,
            self.tub.actor.get_pose().inv() * self.target.get_pose(),
            self.target.actor,
            sapien.Pose(),
        )
        self._configure_drive(self.initial_support_drive)

    def _load_container(self):
        self.tub_id = 0
        self.tub_pose = rand_pose(
            xlim=[0.08, 0.08],
            ylim=[-0.05, -0.05],
            zlim=[0.835, 0.835],
            qpos=t3d.euler.euler2quat(0.0, 0.0, 0.0).tolist(),
        )
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")

        # A hollow dynamic mesh needs a convex collision approximation.
        wall = 0.012
        radius_x, radius_y = 0.075, 0.098
        height = 0.165
        for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            x, y = radius_x * np.cos(angle), radius_y * np.sin(angle)
            builder.add_box_collision(
                pose=sapien.Pose([x, y, 0.0], t3d.euler.euler2quat(0.0, 0.0, angle)),
                half_size=[wall, 0.020, height / 2],
            )
        builder.add_box_collision(
            pose=sapien.Pose([0.0, 0.0, -height / 2]),
            half_size=[radius_x, radius_y, wall],
        )

        handle_pose = sapien.Pose([-0.115, 0.0, 0.0])
        handle_half_size = [0.045, 0.030, 0.050]
        builder.add_box_collision(pose=handle_pose, half_size=handle_half_size)
        builder.add_visual_from_file(
            filename=self.CONTAINER_GLB,
            pose=sapien.Pose(q=t3d.euler.euler2quat(np.pi / 2, 0.0, 0.0)),
            scale=[self.CONTAINER_SCALE] * 3,
        )
        builder.add_box_visual(
            pose=handle_pose,
            half_size=handle_half_size,
            material=sapien.render.RenderMaterial(base_color=[0.18, 0.18, 0.18, 1.0]),
        )

        self.tub = Actor(
            builder.build(name="handled_utensil_container"),
            {"scale": [self.CONTAINER_SCALE] * 3},
        )
        self.tub.actor.set_pose(self.tub_pose)
        handle_contact = self._contact_matrix(
            "right", [-0.115 / self.CONTAINER_SCALE, 0.0, 0.0]
        )
        self.tub.config["contact_points_pose"] = [handle_contact.tolist()]
        self.tub.set_mass(0.35)

    def _load_target(self):
        self.target_name = self.TARGET_NAME
        self.target_id = self.TARGET_MODEL_ID
        target_local = np.array([
            np.random.uniform(-0.012, 0.012),
            0.0,
            -0.07,
            1.0,
        ])
        target_world = self.tub_pose.to_transformation_matrix() @ target_local
        target_pose = sapien.Pose(
            p=target_world[:3],
            q=t3d.euler.euler2quat(np.pi / 2, 0.0, 0.0),
        )
        self.target = create_actor(
            scene=self,
            pose=target_pose,
            modelname=self.target_name,
            model_id=self.target_id,
            scale_override=self.TARGET_SCALE,
            convex=True,
            is_static=False,
        )

        # Physical contact pad for the thin barrel; it has no visual shape.
        target_body = next(
            component for component in self.target.actor.get_components()
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
        )
        target_pad = sapien.physx.PhysxCollisionShapeSphere(
            radius=0.035,
            material=self.scene.default_physical_material,
        )
        target_pad.set_local_pose(sapien.Pose([0.0, 0.152, 0.084]))
        target_body.attach(target_pad)

        target_contact = self._contact_matrix(
            "top_down_little_left", [0.0, 1.90, 1.05]
        )
        self.target.config["contact_points_pose"] = [target_contact.tolist()]
        self.target.set_mass(0.02)

    def check_stable(self):
        # The support drive keeps the target in the open container before grasping.
        return True, []

    def play_once(self):
        left, right = ArmTag("left"), ArmTag("right")
        self.robot.right_planner.planner_type = "mplib_RRT"
        self.move(self.grasp_actor(
            self.tub,
            left,
            pre_grasp_dis=0.04,
            contact_point_id=0,
        ))
        self.move(self.close_gripper(left, pos=0.0))
        if not self.check_gripper_actor_contact(self.tub, left):
            self.plan_success = False
            return self.info
        self._attach_grasp(self.tub, left, 0)

        # Plan after the left grasp fixes the container and target world poses.
        _, target_grasp_pose = self.choose_grasp_pose(
            self.target,
            right,
            pre_dis=0.05,
            contact_point_id=0,
        )
        if self.need_plan:
            right_grasp_result = self.robot.right_plan_path(target_grasp_pose)
            self.right_joint_path.append(deepcopy(right_grasp_result))
        else:
            right_grasp_result = deepcopy(self.right_joint_path[self.right_cnt])
            self.right_cnt += 1
        if right_grasp_result["status"] != "Success":
            self.plan_success = False
            return self.info
        for position, velocity in zip(
            right_grasp_result["position"], right_grasp_result["velocity"]
        ):
            self.robot.set_arm_joints(position, velocity, "right")
            self.scene.step()
            if self.save_freq is not None:
                self._update_render()
                self._take_picture()
        self.move(self.close_gripper(right, pos=0.0))

        attached = False
        contact_streak = 0
        for _ in range(80):
            self.scene.step()
            if self.check_gripper_actor_contact(self.target, right):
                contact_streak += 1
                if contact_streak >= 8:
                    self._attach_grasp(self.target, right, 0)
                    self.initial_support_drive.disable()
                    attached = True
                    break
            else:
                contact_streak = 0
        if not attached:
            self.plan_success = False
            return self.info

        self.move(self.move_by_displacement(right, z=0.20, move_axis="world"))
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": "handled_utensil_container/base0",
            "{a}": "left",
            "{b}": "right",
        }
        return self.info

    def choose_grasp_pose(
        self,
        actor,
        arm_tag,
        pre_dis=0.1,
        target_dis=0,
        contact_point_id=None,
    ):
        point_id = 0 if contact_point_id is None else contact_point_id
        if isinstance(point_id, list):
            point_id = point_id[0]
        contact_matrix = actor.get_contact_point(point_id, "matrix")
        grasp_frame = contact_matrix @ np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
        grasp_rotation = grasp_frame[:3, :3]
        pre_pose = np.concatenate([
            grasp_frame[:3, 3] + grasp_rotation @ np.array([-0.12 - pre_dis, 0, 0]),
            t3d.quaternions.mat2quat(grasp_rotation),
        ])
        direction = t3d.quaternions.quat2mat(pre_pose[-4:])
        grasp_pose = pre_pose.copy()
        grasp_pose[:3] += (pre_dis - target_dis) * direction[:, 0]
        if actor is self.target and arm_tag == "right":
            pre_pose[1] += 0.075
            grasp_pose[1] += 0.075
        return pre_pose.tolist(), grasp_pose.tolist()

    def check_success(self):
        target_contact = self.target.get_contact_point(0, "pose")
        return bool(
            float(target_contact.p[2]) > 1.02
            and self.is_right_gripper_close()
            and self.check_gripper_actor_contact(self.target, "right")
        )

    def check_gripper_actor_contact(self, actor, arm_tag):
        if arm_tag == "left":
            gripper_names = set(self.robot.left_fix_gripper_name)
            gripper_names.update(
                link.child_link.get_name() for link, *_ in self.robot.left_gripper
            )
        elif arm_tag == "right":
            gripper_names = set(self.robot.right_fix_gripper_name)
            gripper_names.update(
                link.child_link.get_name() for link, *_ in self.robot.right_gripper
            )
        else:
            raise ValueError(f"arm_tag must be either 'left' or 'right', not {arm_tag}")

        for contact in self.scene.get_contacts():
            names = {contact.bodies[0].entity.name, contact.bodies[1].entity.name}
            if actor.get_name() in names and names.intersection(gripper_names):
                return True
        return False

    def _attach_grasp(self, actor, arm_tag, contact_point_id):
        gripper = self.robot.left_gripper if arm_tag == "left" else self.robot.right_gripper
        link = gripper[0][0].child_link
        contact_pose = actor.get_contact_point(contact_point_id, "pose")
        drive = self.scene.create_drive(
            link,
            link.get_pose().inv() * contact_pose,
            actor.actor,
            actor.get_pose().inv() * contact_pose,
        )
        self._configure_drive(drive)
