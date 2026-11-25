import sapien

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC


class orient_pencup_to_camera(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_camera(self, **kwargs):
        sapien.render.set_camera_shader_dir("default")
        super().load_camera(**kwargs)
        camera_id = self.cameras.static_camera_name.index("head_camera")
        head_camera = self.cameras.static_camera_list[camera_id]
        pose = head_camera.entity.get_pose()
        position = pose.p.copy()
        position[2] = 1.55
        head_camera.entity.set_pose(sapien.Pose(position, pose.q))

    def load_actors(self):
        self.pencup_id = 0
        self.pencup = rand_create_actor(
            self,
            xlim=[-0.25, -0.18],
            ylim=[-0.08, 0.02],
            modelname="059_pencup_empty",
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
            convex=True,
            model_id=self.pencup_id,
        )
        self.add_prohibit_area(self.pencup, padding=0.08)

        center = np.array(self.pencup.config["center"])
        self.mouth_local = np.array([
            center[0],
            center[1] + self.pencup.config["extents"][1] / 2,
            center[2],
        ]) * np.array(self.pencup.config["scale"])

        self.markpen_id = 0
        cup_pose = self.pencup.get_pose()
        markpen_pose = sapien.Pose(
            [cup_pose.p[0] + 0.015, cup_pose.p[1], cup_pose.p[2] + 0.006],
            cup_pose.q,
        )
        self.markpen = create_actor(
            scene=self,
            pose=markpen_pose,
            modelname="058_markpen",
            model_id=self.markpen_id,
            scale_override=[0.07, 0.07, 0.07],
            convex=True,
            is_static=False,
        )
        marker_grasps = np.array(self.markpen.config["contact_points_pose"])
        marker_grasps[:, 1, 3] = 1.85
        self.markpen.config["contact_points_pose"] = marker_grasps.tolist()
        self.markpen.config["contact_points_group"] = [list(range(len(marker_grasps)))]
        self.markpen.config["contact_points_mask"] = [True]
        self.delay(2)

    def _mouth_state(self):
        cup_matrix = self.pencup.get_pose().to_transformation_matrix()
        mouth_position = cup_matrix[:3, :3] @ self.mouth_local + cup_matrix[:3, 3]
        mouth_axis = cup_matrix[:3, :3] @ np.array([0.0, 1.0, 0.0])
        return mouth_position, mouth_axis

    def _marker_center(self):
        marker_matrix = self.markpen.get_pose().to_transformation_matrix()
        marker_center = np.array(self.markpen.config["center"]) * np.array(self.markpen.config["scale"])
        return marker_matrix[:3, :3] @ marker_center + marker_matrix[:3, 3]

    def _marker_in_cup(self):
        world_center = self._marker_center()
        cup_inverse = np.linalg.inv(self.pencup.get_pose().to_transformation_matrix())
        local_center = (cup_inverse @ np.append(world_center, 1.0))[:3]
        radial_distance = np.linalg.norm(local_center[[0, 2]])
        return bool(radial_distance < 0.04 and 0.0 < local_center[1] < 0.12)

    def _face_head_camera(self, arm_tag):
        camera_id = self.cameras.static_camera_name.index("head_camera")
        camera_position = self.cameras.static_camera_list[camera_id].entity.get_pose().p
        mouth_position, mouth_axis = self._mouth_state()
        target_axis = camera_position - mouth_position
        rotation = get_align_matrix(mouth_axis, target_axis)

        gripper_pose = np.array(self.robot.get_left_ee_pose())
        gripper_rotation = t3d.quaternions.quat2mat(gripper_pose[-4:])
        target_position = mouth_position + rotation @ (gripper_pose[:3] - mouth_position)
        target_quat = t3d.quaternions.mat2quat(rotation @ gripper_rotation)
        target_pose = target_position.tolist() + target_quat.tolist()
        return self.move_to_pose(arm_tag, target_pose)

    def _grasp_marker(self, arm_tag):
        contact_position = self.markpen.get_contact_point(0, "pose").p
        rotation = t3d.quaternions.quat2mat(GRASP_DIRECTION_DIC["front_left"])
        quat = t3d.quaternions.mat2quat(rotation)
        pre_position = contact_position + rotation @ np.array([-0.20, 0.0, 0.0])
        grasp_position = contact_position + rotation @ np.array([-0.14, 0.0, 0.0])
        return arm_tag, [
            Action(arm_tag, "move", target_pose=pre_position.tolist() + quat.tolist()),
            Action(
                arm_tag,
                "move",
                target_pose=grasp_position.tolist() + quat.tolist(),
                constraint_pose=[1, 1, 1, 0, 0, 0],
            ),
            Action(arm_tag, "close", target_gripper_pos=0.0),
        ]

    def _right_marker_contact(self):
        finger_names = {link.child_link.get_name() for link, *_ in self.robot.right_gripper}
        contacted_fingers = set()
        for contact in self.scene.get_contacts():
            names = {contact.bodies[0].entity.name, contact.bodies[1].entity.name}
            if self.markpen.get_name() in names:
                contacted_fingers.update(names.intersection(finger_names))
        return finger_names.issubset(contacted_fingers)

    def _attach_marker(self):
        link = self.robot.right_gripper[0][0].child_link
        contact_pose = self.markpen.get_contact_point(0, "pose")
        self.marker_grasp_drive = self.scene.create_drive(
            link,
            link.get_pose().inv() * contact_pose,
            self.markpen.actor,
            self.markpen.get_pose().inv() * contact_pose,
        )
        for setter in (
            self.marker_grasp_drive.set_drive_property_x,
            self.marker_grasp_drive.set_drive_property_y,
            self.marker_grasp_drive.set_drive_property_z,
        ):
            setter(1000.0, 100.0, 1000.0)
        self.marker_grasp_drive.set_drive_property_swing(1000.0, 100.0, 1000.0)
        self.marker_grasp_drive.set_drive_property_twist(1000.0, 100.0, 1000.0)

    def _marker_vertices_world(self):
        vertices = []
        marker_pose = self.markpen.get_pose()
        for component in self.markpen.actor.components:
            if not isinstance(component, sapien.physx.PhysxRigidBodyComponent):
                continue
            for shape in component.collision_shapes:
                if not isinstance(shape, sapien.physx.PhysxCollisionShapeConvexMesh):
                    continue
                local_vertices = np.asarray(shape.vertices) * np.asarray(shape.scale)
                transform = (marker_pose * shape.local_pose).to_transformation_matrix()
                vertices.append(local_vertices @ transform[:3, :3].T + transform[:3, 3])
        return np.concatenate(vertices)

    def _lift_marker_out(self, arm_tag):
        mouth_position, mouth_axis = self._mouth_state()
        mouth_axis = mouth_axis / np.linalg.norm(mouth_axis)
        vertices = self._marker_vertices_world()
        lowest_axial = np.min((vertices - mouth_position) @ mouth_axis)
        lift_distance = max(0.0, -lowest_axial) + 0.05
        target_pose = np.array(self.robot.get_right_ee_pose())
        target_pose[:3] += mouth_axis * lift_distance
        return self.move_to_pose(arm_tag, target_pose)

    def _marker_table_actions(self, arm_tag):
        gripper_pose = np.array(self.robot.get_right_ee_pose())
        lowest_z_offset = self._marker_vertices_world()[:, 2].min() - gripper_pose[2]
        place_z = 0.74 + self.table_z_bias + 0.008 - lowest_z_offset
        self.marker_place_xy = np.array([0.26, -0.12])

        above_pose = gripper_pose.copy()
        above_pose[:3] = [*self.marker_place_xy, max(gripper_pose[2], place_z + 0.10)]
        place_pose = gripper_pose.copy()
        place_pose[:3] = [*self.marker_place_xy, place_z]
        return arm_tag, [
            Action(arm_tag, "move", target_pose=above_pose),
            Action(arm_tag, "move", target_pose=place_pose),
        ]

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(self.pencup, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.30))
        self.move(self._face_head_camera(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.24))

        right_arm_tag = ArmTag("right")
        self.move(self._grasp_marker(right_arm_tag))
        if not self.plan_success:
            return self.info

        contact_streak = 0
        for _ in range(100):
            self.scene.step()
            if self.save_freq is not None:
                self._update_render()
                self._take_picture()
            if self.is_right_gripper_close() and self._right_marker_contact():
                contact_streak += 1
                if contact_streak >= 20:
                    break
            else:
                contact_streak = 0
        if contact_streak < 20:
            self.plan_success = False
            return self.info

        self._attach_marker()
        self.move(self._lift_marker_out(right_arm_tag))
        if not self.plan_success:
            return self.info
        self.move(self._marker_table_actions(right_arm_tag))
        if not self.plan_success:
            return self.info

        self.marker_grasp_drive.disable()
        self.move(self.open_gripper(right_arm_tag))
        for _ in range(80):
            self.scene.step()
            if self.save_freq is not None:
                self._update_render()
                self._take_picture()
        self.move(self.move_by_displacement(arm_tag=right_arm_tag, z=0.12))

        self.info["info"] = {
            "{A}": f"059_pencup_empty/base{self.pencup_id}",
            "{B}": f"058_markpen/base{self.markpen_id}",
            "{a}": str(arm_tag),
            "{b}": str(right_arm_tag),
        }
        return self.info

    def check_success(self):
        mouth_position, _ = self._mouth_state()
        contact = self.get_gripper_actor_contact_position(self.pencup.get_name())
        marker_bottom = self._marker_vertices_world()[:, 2].min()
        marker_center = self._marker_center()
        return bool(
            mouth_position[2] > 0.90 + self.table_z_bias
            and self.is_left_gripper_close()
            and len(contact) > 0
            and self.is_right_gripper_open()
            and not self.marker_grasp_drive.is_enabled
            and not self._marker_in_cup()
            and abs(marker_bottom - (0.74 + self.table_z_bias)) < 0.02
            and abs(marker_center[0]) < 0.55
            and abs(marker_center[1]) < 0.30
        )
