import os
# os.environ['MUJOCO_GL'] = 'glx'
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np
import collections
import os
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import yaml
import torch

from constants import DT, XML_DIR, START_UAM_POSE
from constants import UAM_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import UAM_GRIPPER_POSITION_NORMALIZE_FN
from constants import SIM_TASK_CONFIGS
from constants import UMI_GRIPPER_OPEN, UMI_GRIPPER_CLOSE

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import sys
sys.path.append("..")
sys.path.append("../am_trajectory_controller/src")
sys.path.append("../am_trajectory_controller/src/planner")

from controllers.motion_controller_4dof import UAM4DoFMotionController
from am_trajectory_controller.src.planner.ee_mpc_acado_4dof import ArmMPCPlanner
from am_trajectory_controller.src.planner.ee_ik_DH_4dof import DHIKPlanner


def make_ee_sim_env(task_name, camera_names=['ee'], disturbance_enabled=False, acados_build_dir=None):
    """Environment for simulated ee-centric aerial manipulation with end-effector control."""

    if task_name == 'uam_peg':
        xml_path = os.path.join(XML_DIR, f'hexa_scorpion_4dofarm_peginhole_difficult_1.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PegInHoleEETask_4DoF(physics, task_name, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'uam_pick':
        xml_path = os.path.join(XML_DIR, f'hexa_scorpion_4dofarm_pnp.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceEETask_4DoF(physics, task_name, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'uam_valve':
        xml_path = os.path.join(XML_DIR, f'hexa_scorpion_4dofarm_rotatevalve.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = RotateValveEETask_4DoF(physics, task_name, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'uam_cabinet':
        xml_path = os.path.join(XML_DIR, f'hexa_scorpion_4dofarm_cabinet.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = CabinetEETask_4DoF(physics, task_name, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled, acados_build_dir=acados_build_dir)
    elif task_name == 'ur10e_cabinet':
        xml_path = os.path.join(XML_DIR, 'ur10e_umi_cabinet.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = CabinetUR10eEETask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'ur10e_pick':
        xml_path = os.path.join(XML_DIR, 'ur10e_umi_pnp.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceUR10eEETask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'ur10e_valve':
        xml_path = os.path.join(XML_DIR, 'ur10e_umi_rotatevalve.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = RotateValveUR10eEETask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'ur10e_peg':
        xml_path = os.path.join(XML_DIR, 'ur10e_umi_peginhole_difficult_1.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PegInHoleUR10eEETask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'umi_cabinet':
        xml_path = os.path.join(XML_DIR, 'umi_cabinet.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = CabinetUMIOracleTask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'umi_peg':
        xml_path = os.path.join(XML_DIR, 'umi_peginhole_difficult_1.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PegInHoleUMIOracleTask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'umi_pick':
        xml_path = os.path.join(XML_DIR, 'umi_pnp.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceUMIOracleTask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    elif task_name == 'umi_valve':
        xml_path = os.path.join(XML_DIR, 'umi_rotatevalve.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = RotateValveUMIOracleTask(physics, random=False, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
    else:
        raise NotImplementedError
    env = control.Environment(physics, task, time_limit=200, control_timestep=DT, n_sub_steps=None, flat_observation=False)
    return env

# ============================================================================
# Robot Classes 
# ============================================================================

class UMIOracleBaseTask(base.Task):
    """Base class for all UMI Oracle tasks (mocap-controlled standalone UMI robot).
    
    Common properties:
    - Action space: [x, y, z, qw, qx, qy, qz, grip] (8D, grip ∈ [0,1], 0=open, 1=closed)
    - Mocap-based EE control
    - UMI-style gripper mapping
    - Standard observation format
    
    Subclasses should implement:
    - Task-specific initialization in initialize_episode()
    - Task-specific reward logic in get_reward()
    """
    
    GRIPPER_MIN = UMI_GRIPPER_CLOSE
    GRIPPER_MAX = UMI_GRIPPER_OPEN
    
    DEFAULT_REF_EE_POS = np.array([0.0, 0.0, 1.2])
    
    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        super().__init__(random=random)
        self.camera_names = list(camera_names)
        
        self.lfinger_actuator_id = physics.model.name2id("lfinger_servo", "actuator")
        self.rfinger_actuator_id = physics.model.name2id("rfinger_servo", "actuator")
    
    def _set_gripper_state(self, physics, state):
        """Set gripper to initial state (OPEN or CLOSED)."""
        value = self.GRIPPER_MAX if state == 'OPEN' else self.GRIPPER_MIN
        physics.data.ctrl[self.lfinger_actuator_id] = value
        physics.data.ctrl[self.rfinger_actuator_id] = value
    
    def before_step(self, action, physics, trajectory=None):
        """Map 8-D action to mocap + gripper targets."""
        pos = action[:3]
        quat = np.array([action[3], action[4], action[5], action[6]])
        physics.data.mocap_pos[0] = pos
        physics.data.mocap_quat[0] = quat
        
        grip = np.clip(action[7], 0.0, 1.0)
        gripper_target = self.GRIPPER_MIN + (1.0 - grip) * (self.GRIPPER_MAX - self.GRIPPER_MIN)
        physics.data.ctrl[self.lfinger_actuator_id] = gripper_target
        physics.data.ctrl[self.rfinger_actuator_id] = gripper_target
    
    def _initialize_episode_start(self, physics):
        """Common initialization logic for all UMI tasks - call at start of initialize_episode."""
        physics.data.mocap_pos[0] = self.DEFAULT_REF_EE_POS.copy()
        physics.data.mocap_quat[0] = np.array([1, 0, 0, 0])
        physics.data.qpos[:3] = self.DEFAULT_REF_EE_POS.copy()
        physics.forward()
    
    def get_observation(self, physics):
        """Standard observation format: mocap pose + normalized gripper."""
        obs = collections.OrderedDict()
        
        gripper_pos = physics.data.ctrl[self.lfinger_actuator_id]
        gripper_normalized = (self.GRIPPER_MAX - gripper_pos) / (self.GRIPPER_MAX - self.GRIPPER_MIN)
        gripper_normalized = np.clip(gripper_normalized, 0.0, 1.0)
        
        obs["qpos"] = np.concatenate([
            physics.data.mocap_pos[0],
            physics.data.mocap_quat[0],
            [gripper_normalized]
        ])
        obs["images"] = {name: physics.render(480, 640, camera_id=name)
                         for name in self.camera_names}
        return obs
    
    @staticmethod
    def get_env_state(physics):
        """Return full environment state (all qpos)."""
        return physics.data.qpos.copy()

class UR10eBaseTask(base.Task):
    """
    Base task for UR10e with UMI gripper:
    - 6-DoF EE pose tracking via damped least-squares Jacobian IK → joint position targets
    - UMI-style gripper mapping on ctrl indices [6, 7]
    - Shared rollout-based trajectory cost gradient for 8-step horizon
    """

    GRIPPER_MIN = UMI_GRIPPER_CLOSE
    GRIPPER_MAX = UMI_GRIPPER_OPEN

    def __init__(self, physics, random=None, camera_names=("ee",), gradient_method='jacobian'):
        super().__init__(random=random)
        self.camera_names = list(camera_names)
        self.gradient_method = gradient_method  # 'jacobian' or 'autodiff'

        self.ee_bid = physics.model.name2id("ee", "body")

        self.ur_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.ur_jids = [physics.model.name2id(n, "joint") for n in self.ur_joint_names]
        self.ur_qpos_addrs = [physics.model.jnt_qposadr[jid] for jid in self.ur_jids]

        self.lambda_dls = 0.05
        self.max_joint_step = 0.05

        self.lfinger_jid = physics.model.name2id("lfinger_slide", "joint")
        self.rfinger_jid = physics.model.name2id("rfinger_slide", "joint")
        self.lfinger_qpos_addr = physics.model.jnt_qposadr[self.lfinger_jid]
        self.rfinger_qpos_addr = physics.model.jnt_qposadr[self.rfinger_jid]
        
        self.lfinger_actuator_id = physics.model.name2id("lfinger_servo", "actuator")
        self.rfinger_actuator_id = physics.model.name2id("rfinger_servo", "actuator")
        
        self._ur_base_bid = physics.model.name2id("base", "body")

        self.max_joint_velocity = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float64)
        self.rollout_horizon_s = 0.64
    
    def _set_gripper_state(self, physics, state):
        """Set gripper to initial state (OPEN or CLOSED)."""
        value = self.GRIPPER_MAX if state == 'OPEN' else self.GRIPPER_MIN
        physics.data.ctrl[self.lfinger_actuator_id] = value
        physics.data.ctrl[self.rfinger_actuator_id] = value
        physics.data.qpos[self.lfinger_qpos_addr] = value
        physics.data.qpos[self.rfinger_qpos_addr] = value
    
    def _initialize_ur10e_robot(self, physics, base_pos, ee_pos, ee_quat=None):
        """Common UR10e initialization pattern: set base, move EE, hold targets."""
        if ee_quat is None:
            ee_quat = np.array([1, 0, 0, 0])
        physics.model.body_pos[self._ur_base_bid] = base_pos
        physics.forward()
        self.move_ee_to_pose(physics, ee_pos, ee_quat)
        self.hold_current_joint_targets(physics)

    def _get_ur_qpos_view(self, physics):
        qpos = physics.data.qpos
        indices = []
        for addr in self.ur_qpos_addrs:
            indices.append(addr)
        return qpos[indices]

    def _set_ur_ctrl_targets(self, physics, q_target):
        physics.data.ctrl[0:6] = q_target

    def _orientation_error_vee(self, q_current_wxyz, q_des_wxyz):
        qc_xyzw = np.array([q_current_wxyz[1], q_current_wxyz[2], q_current_wxyz[3], q_current_wxyz[0]])
        qd_xyzw = np.array([q_des_wxyz[1],    q_des_wxyz[2],    q_des_wxyz[3],    q_des_wxyz[0]])
        Rc = R.from_quat(qc_xyzw)
        Rd = R.from_quat(qd_xyzw)
        Rerr = Rd * Rc.inv()
        return Rerr.as_rotvec()

    def before_step(self, action, physics, trajectory=None):
        pos_d  = action[:3]
        quat_wxyz_d = np.array([action[3], action[4], action[5], action[6]])

        ee_pos = physics.data.xpos[self.ee_bid][:3].copy()
        ee_quat_wxyz = physics.data.xquat[self.ee_bid].copy()

        dx = pos_d - ee_pos
        dtheta = self._orientation_error_vee(ee_quat_wxyz, quat_wxyz_d)
        delta = np.concatenate([dx, dtheta])

        nv = physics.model.nv
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, Jp, Jr, self.ee_bid)
        J = np.vstack([Jp, Jr])

        cols = []
        for jid in self.ur_jids:
            dof_adr = physics.model.jnt_dofadr[jid]
            cols.append(dof_adr)
        J_ur = J[:, cols]

        JJt = J_ur @ J_ur.T
        A = JJt + (self.lambda_dls ** 2) * np.eye(6)
        dq = J_ur.T @ np.linalg.solve(A, delta)

        max_step_control = self.max_joint_velocity * DT
        dq = np.clip(dq, -max_step_control, max_step_control)
        q_curr = self._get_ur_qpos_view(physics).copy()
        q_tgt = q_curr + dq
        self._set_ur_ctrl_targets(physics, q_tgt)

        grip = np.clip(action[7], 0.0, 1.0)
        finger_target = self.GRIPPER_MIN + (1.0 - grip) * (self.GRIPPER_MAX - self.GRIPPER_MIN)
        physics.data.ctrl[self.lfinger_actuator_id] = finger_target
        physics.data.ctrl[self.rfinger_actuator_id] = finger_target

    def move_ee_to_pose(self, physics, target_pos_wxyz, target_quat_wxyz=np.array([1, 0, 0, 0])):
        pos_d = np.asarray(target_pos_wxyz, dtype=np.float64)
        quat_wxyz_d = np.asarray(target_quat_wxyz, dtype=np.float64)
        nv = physics.model.nv
        for _ in range(600):
            ee_pos = physics.data.xpos[self.ee_bid][:3].copy()
            ee_quat_wxyz = physics.data.xquat[self.ee_bid].copy()
            dx = pos_d - ee_pos
            dtheta = self._orientation_error_vee(ee_quat_wxyz, quat_wxyz_d)
            dx = np.clip(dx, -0.015, 0.015)
            dtheta = np.clip(dtheta, -0.02, 0.02)
            delta = np.concatenate([dx, dtheta])

            Jp = np.zeros((3, nv))
            Jr = np.zeros((3, nv))
            mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, Jp, Jr, self.ee_bid)
            J = np.vstack([Jp, Jr])
            cols = [physics.model.jnt_dofadr[jid] for jid in self.ur_jids]
            J_ur = J[:, cols]
            JJt = J_ur @ J_ur.T
            A = JJt + (self.lambda_dls ** 2) * np.eye(6)
            dq = J_ur.T @ np.linalg.solve(A, delta)
            dq = np.clip(dq, -0.08, 0.08)
            for i, addr in enumerate(self.ur_qpos_addrs):
                physics.data.qpos[addr] += dq[i]
            physics.forward()

            if np.linalg.norm(dx) < 1e-3 and np.linalg.norm(dtheta) < 1e-3:
                break

    def hold_current_joint_targets(self, physics):
        q_hold = np.array([physics.data.qpos[addr] for addr in self.ur_qpos_addrs])
        physics.data.ctrl[0:6] = q_hold

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        # qpos: 8-D to match other embodiments: [ee_pos(3), ee_quat(4), gripper_normalized(1)]
        # Use the same normalization as other embodiments (from finger joint qpos)
        gripper_pos_left = physics.data.qpos[self.lfinger_qpos_addr]
        gripper_normalized = 1.0 - UAM_GRIPPER_POSITION_NORMALIZE_FN(gripper_pos_left)
        gripper_normalized = float(np.clip(gripper_normalized, 0.0, 1.0))
        obs["qpos"] = np.concatenate([
            physics.data.xpos[self.ee_bid][:3],
            physics.data.xquat[self.ee_bid].copy(),
            np.array([gripper_normalized], dtype=np.float64)
        ])
        obs["qvel"] = physics.data.cvel[self.ee_bid].copy()
        obs["images"] = {name: physics.render(480, 640, camera_id=name)
                          for name in self.camera_names}
        return obs

    def compute_trajectory_cost_with_tracking_gradient(self, physics, trajectory, method=None):
        """
        Evaluate tracking cost for trajectory and return gradient.
        
        Args:
            physics: MuJoCo physics instance
            trajectory: H×8 array [x,y,z, w,x,y,z, g]
            method: 'jacobian' (fast, approximate) or 'autodiff' (slow, exact)
        
        Returns:
            total_cost, tracking_cost, gradient (H,7), mpc_x_opt (H,7)
        """
        method = method or self.gradient_method
        if method == 'jacobian':
            return self._compute_gradient_jacobian(physics, trajectory)
        elif method == 'autodiff':
            return self._compute_gradient_autodiff(physics, trajectory)
        else:
            raise ValueError(f"Unknown gradient method: {method}. Use 'jacobian' or 'autodiff'.")
    
    def _compute_gradient_jacobian(self, physics, trajectory):
        """
        Evaluate tracking cost using frozen-Jacobian DLS method (fast, approximate).
        Returns: total_cost, tracking_cost, gradient (H,7), mpc_x_opt (H,7)
        """
        if trajectory is None or len(trajectory) == 0:
            return float("inf"), float("inf"), None, None

        H = trajectory.shape[0]
        pos_ref = trajectory[:, :3]
        quat_ref = trajectory[:, 3:7]

        # Current state and Jacobian
        q_curr = np.array([physics.data.qpos[addr] for addr in self.ur_qpos_addrs])
        pos0 = physics.data.xpos[self.ee_bid][:3].copy()
        quat0 = physics.data.xquat[self.ee_bid].copy()

        nv = physics.model.nv
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, Jp, Jr, self.ee_bid)
        J = np.vstack([Jp, Jr])
        cols = [physics.model.jnt_dofadr[jid] for jid in self.ur_jids]
        J_ur = J[:, cols]  # 6x6

        lam = self.lambda_dls
        JJt = J_ur @ J_ur.T
        A = JJt + (lam ** 2) * np.eye(6)

        # Rollout with frozen Jacobian
        pos_pred = np.zeros((H, 3), dtype=np.float64)
        quat_pred = np.zeros((H, 4), dtype=np.float64)
        pos_k = pos0.copy()
        quat_k = quat0.copy()
        q_k = q_curr.copy()

        for k in range(H):
            pos_pred[k] = pos_k
            quat_pred[k] = quat_k
            dx = pos_ref[k] - pos_k
            dtheta = self._orientation_error_vee(quat_k, quat_ref[k])
            delta = np.concatenate([dx, dtheta])
            dq = J_ur.T @ np.linalg.solve(A, delta)
            # dt-aware per-joint step clip using rollout dt (T/H)
            dt_rollout = self.rollout_horizon_s / H
            max_step_rollout = self.max_joint_velocity * dt_rollout
            dq = np.clip(dq, -max_step_rollout, max_step_rollout)
            q_k = q_k + dq
            delta_task = J_ur @ dq
            pos_k = pos_k + delta_task[:3]
            dtheta_step = delta_task[3:]
            qxyzw = R.from_quat([quat_k[1], quat_k[2], quat_k[3], quat_k[0]])
            qxyzw = (R.from_rotvec(dtheta_step) * qxyzw).as_quat()
            quat_k = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])

        # Costs (position L2 + quaternion chordal with sign alignment)
        pos_err = pos_ref - pos_pred
        dots = np.sum(quat_ref * quat_pred, axis=1, keepdims=True)
        signs = np.sign(dots)
        quat_err = quat_ref - signs * quat_pred

        w_pos = 1.0
        w_rot = 1.0
        J_pos = np.sum(np.sum(pos_err ** 2, axis=1))
        J_rot = np.sum(np.sum(quat_err ** 2, axis=1))
        total_cost = tracking_cost = w_pos * J_pos + w_rot * J_rot

        grad_pos = 2.0 * w_pos * pos_err
        grad_quat = 2.0 * w_rot * quat_err
        gradient = np.concatenate([grad_pos, grad_quat], axis=1).astype(np.float32)

        mpc_x_opt = np.concatenate([pos_pred, quat_pred], axis=1).astype(np.float32)

        return float(total_cost), float(tracking_cost), gradient, mpc_x_opt


    def _compute_gradient_autodiff(self, physics, trajectory):
        """
        Differentiable tracking cost through frozen-Jacobian IK rollout using PyTorch autodiff (slow, exact).

        Args:
            physics: dm_control mujoco physics handle (used only to fetch initial state and Jacobian).
            trajectory: (H, 8) float array [x,y,z, w,x,y,z, g]. Gripper dim is ignored.

        Returns:
            total_cost (float), tracking_cost (float), gradient (H,7), mpc_x_opt (H,7)
            where gradient is d(total_cost)/d(trajectory[:,:7]).
        """
        if trajectory is None or len(trajectory) == 0:
            return float("inf"), float("inf"), None, None

        # Shapes and references
        H = trajectory.shape[0]
        pos_ref_np = trajectory[:, :3].astype(np.float64)
        quat_ref_np = trajectory[:, 3:7].astype(np.float64)

        # Current state and frozen Jacobian (constants for this call)
        q_curr_np = np.array([physics.data.qpos[addr] for addr in self.ur_qpos_addrs], dtype=np.float64)
        pos0_np = physics.data.xpos[self.ee_bid][:3].copy().astype(np.float64)
        quat0_np = physics.data.xquat[self.ee_bid].copy().astype(np.float64)

        nv = physics.model.nv
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, Jp, Jr, self.ee_bid)
        J_np = np.vstack([Jp, Jr])
        cols = [physics.model.jnt_dofadr[jid] for jid in self.ur_jids]
        J_ur_np = J_np[:, cols]  # (6,6)

        lam = self.lambda_dls
        JJt_np = J_ur_np @ J_ur_np.T
        A_np = JJt_np + (lam ** 2) * np.eye(6)

        # Torch tensors
        dtype = torch.float64
        device = torch.device("cpu")

        traj_vars = torch.tensor(
            np.concatenate([pos_ref_np, quat_ref_np], axis=1),
            dtype=dtype, device=device, requires_grad=True
        )  # (H,7)
        pos_k = torch.tensor(pos0_np, dtype=dtype, device=device)
        quat_k = torch.tensor(quat0_np, dtype=dtype, device=device)
        q_k = torch.tensor(q_curr_np, dtype=dtype, device=device)

        J_ur = torch.tensor(J_ur_np, dtype=dtype, device=device)
        A = torch.tensor(A_np, dtype=dtype, device=device)

        dt_rollout = self.rollout_horizon_s / float(H)
        max_step_rollout = torch.tensor(self.max_joint_velocity * dt_rollout, dtype=dtype, device=device)

        def quat_mul(q1_wxyz: torch.Tensor, q2_wxyz: torch.Tensor) -> torch.Tensor:
            w1, x1, y1, z1 = q1_wxyz
            w2, x2, y2, z2 = q2_wxyz
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ])

        def quat_conj(q_wxyz: torch.Tensor) -> torch.Tensor:
            return torch.stack([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]])

        def quat_inv(q_wxyz: torch.Tensor) -> torch.Tensor:
            # Unit quaternion inverse is conjugate
            return quat_conj(q_wxyz)

        def rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
            eps = 1e-12
            theta = torch.linalg.norm(rv)
            half = 0.5 * theta
            cos_half = torch.cos(half)
            # scale = sin(half)/theta with stable small-angle fallback ~ 0.5 - theta^2/48
            scale = torch.where(
                theta > eps,
                torch.sin(half) / theta,
                0.5 - (theta * theta) / 48.0,
            )
            v = scale * rv
            return torch.stack([cos_half, v[0], v[1], v[2]])

        def orientation_error_vee_torch(q_current_wxyz: torch.Tensor, q_des_wxyz: torch.Tensor) -> torch.Tensor:
            # q_err = q_des * inv(q_current)
            q_err = quat_mul(q_des_wxyz, quat_inv(q_current_wxyz))
            w = q_err[0]
            v = q_err[1:]
            v_norm = torch.linalg.norm(v)
            eps = 1e-12
            # angle = 2*atan2(||v||, w)
            angle = 2.0 * torch.atan2(v_norm, w)
            # rotvec = angle * v/||v|| ; small-angle fallback: ~ 2*v
            safe = v_norm > eps
            rotvec = torch.where(
                safe.unsqueeze(0),
                (angle * (v / (v_norm + 1e-18))).unsqueeze(0),
                (2.0 * v).unsqueeze(0),
            ).squeeze(0)
            return rotvec

        pos_pred_list = []
        quat_pred_list = []

        for k in range(H):
            pos_pred_list.append(pos_k)
            quat_pred_list.append(quat_k)

            ref_k = traj_vars[k]
            pos_ref_k = ref_k[:3]
            quat_ref_k = ref_k[3:7]

            dx = pos_ref_k - pos_k
            dtheta = orientation_error_vee_torch(quat_k, quat_ref_k)
            delta = torch.cat([dx, dtheta], dim=0)

            dq = J_ur.T @ torch.linalg.solve(A, delta)
            dq = torch.clamp(dq, -max_step_rollout, max_step_rollout)
            q_k = q_k + dq

            delta_task = J_ur @ dq
            pos_k = pos_k + delta_task[:3]
            dtheta_step = delta_task[3:]
            quat_k = quat_mul(rotvec_to_quat(dtheta_step), quat_k)

        pos_pred = torch.stack(pos_pred_list, dim=0)
        quat_pred = torch.stack(quat_pred_list, dim=0)

        # Costs (position L2 + quaternion chordal with sign alignment)
        w_pos = 1.0
        w_rot = 1.0

        pos_err = traj_vars[:, :3] - pos_pred
        dots = torch.sum(traj_vars[:, 3:7] * quat_pred, dim=1, keepdim=True)
        signs = torch.sign(dots)
        quat_err = traj_vars[:, 3:7] - signs * quat_pred

        J_pos = (pos_err.pow(2).sum(dim=1)).sum()
        J_rot = (quat_err.pow(2).sum(dim=1)).sum()
        total_cost = w_pos * J_pos + w_rot * J_rot

        total_cost.backward()

        gradient = traj_vars.grad.detach().cpu().numpy().astype(np.float32)
        mpc_x_opt = torch.cat([pos_pred, quat_pred], dim=1).detach().cpu().numpy().astype(np.float32)

        total_cost_val = float(total_cost.detach().cpu().item())
        tracking_cost_val = total_cost_val

        return total_cost_val, tracking_cost_val, gradient, mpc_x_opt

class UAMBaseTask(base.Task):
    DEFAULT_REF_EE_POS = np.array([0.0, 0.0, 1.2])
    
    GRIPPER_MIN = UMI_GRIPPER_CLOSE
    GRIPPER_MAX = UMI_GRIPPER_OPEN
    
    def __init__(self, physics, task_name, random=None, camera_names=['ee'], disturbance_enabled=False, acados_build_dir=None):
        super().__init__(random=random)
        self.task_name = task_name
        # Initialize MuJoCo model and data
        self.ee_bid = physics.model.name2id("ee", "body")
        
        # Allow overriding the MPC config via env var (set by CLI such as --vis_long)
        mpc_config_override = os.environ.get('MPC_CONFIG_OVERRIDE')
        if mpc_config_override:
            mpc_config_path = mpc_config_override
        else:
            # Use os.path.join with absolute path to ensure correct path resolution
            mpc_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'am_trajectory_controller/config/traj_sender_mpc_ee.yaml'
            )
        with open(mpc_config_path, 'r') as f:
            self.mpc_config = yaml.safe_load(f)
        mpc_params = self.mpc_config['mpc']
        mass = mpc_params['mass']
        T = mpc_params['T']
        N = mpc_params['N']
        N_mini = mpc_params['N_mini']
        Q = mpc_params['Q']
        R = mpc_params['R']
        R_arm_delta = mpc_params['R_delta']
        acc2thrust_gain = mpc_params['acc2thrust_gain']
        pos_min = mpc_params['pos_min']
        vel_min = mpc_params['vel_min']
        vel_max = mpc_params['vel_max']
        acc_min = mpc_params['acc_min']
        acc_max = mpc_params['acc_max']
        joint_min = mpc_params['joint_min']
        joint_max = mpc_params['joint_max']
        default_arm_angle = mpc_params['default_arm_angle']
        output_filter_gain = mpc_params['output_filter_gain']
        moment_of_inertia = mpc_params['moment_of_inertia']

        task_config = SIM_TASK_CONFIGS.get(task_name, {})
        base_x_ub = task_config.get('base_x_ub', 0.1)
        pos_max = [base_x_ub, 100.0, 100.0]
        
        # Initialize IK planner (for forward kinematics only, so base_x_ub doesn't affect anything)
        self.ik_planner = DHIKPlanner(base_x_ub=base_x_ub)

        if acados_build_dir:
            main_build_dir = os.path.join(acados_build_dir, "main")
            mini_build_dir = os.path.join(acados_build_dir, "mini")
        else:
            main_build_dir = "./acados_mpc_build"
            mini_build_dir = "./acados_mpc_build_mini"

        self.mpc_planner = ArmMPCPlanner(mass, T, N, N_mini, Q, R, R_arm_delta, acc2thrust_gain,
                                       pos_min, pos_max, vel_min, vel_max, acc_min, acc_max, joint_min, joint_max, default_arm_angle, output_filter_gain, moment_of_inertia, main_build_dir, "fully_actuated_uav_main")
        # Secondary planner with a shortened horizon (N_mini) for gradient queries
        self.mpc_planner_mini = ArmMPCPlanner(mass, T, N_mini, N_mini, Q, R, R_arm_delta, acc2thrust_gain,
                                       pos_min, pos_max, vel_min, vel_max, acc_min, acc_max, joint_min, joint_max, default_arm_angle, output_filter_gain, moment_of_inertia, mini_build_dir, "fully_actuated_uav_mini")

        # Initialize controllers
        self.motion_controller = UAM4DoFMotionController(physics.data.time)

        self.current_base_pos = np.zeros(3)
        self.current_base_ori = np.array([0, 0, 0, 1])
        self.current_manipulator_joint_angles = np.zeros(4)
        self.u_prev = np.zeros(10)

        self.camera_names = camera_names
        self.disturbance_enabled = disturbance_enabled

        self.controller_info_targets = []
        print(f"Universal aerial manipulation task initialized with MPC control")
        
        # Gripper actuator indices (lookup by name for robustness)
        self.lfinger_actuator_id = physics.model.name2id("manipulation_lfinger_motor", "actuator")
        self.rfinger_actuator_id = physics.model.name2id("manipulation_rfinger_motor", "actuator")
        
        # Gripper joint qpos addresses
        self.lfinger_jid = physics.model.name2id("arm_lfinger_joint", "joint")
        self.rfinger_jid = physics.model.name2id("arm_rfinger_joint", "joint")
        self.lfinger_qpos_addr = physics.model.jnt_qposadr[self.lfinger_jid]
        self.rfinger_qpos_addr = physics.model.jnt_qposadr[self.rfinger_jid]
        
        # ------------------------------------------------------------
        # Disturbance (gust) model state – initialise once
        # ------------------------------------------------------------
        if self.disturbance_enabled:
            # timer & update interval
            self._gust_interval = 40      # steps 
            self._gust_timer    = 50

            # current & target values
            self._wind          = np.zeros(3)
            self._wind_target   = np.zeros(3)
            self._torque        = np.zeros(3)
            self._torque_target = np.zeros(3)

            # ranges and dynamics (read from task config, with defaults)
            self._wind_range   = task_config.get('wind_range', (-8.0, 8.0))     # N
            self._torque_range = task_config.get('torque_range', (-0.0, 0.0))   # N·m
            self._gust_alpha   = 0.02            # smoothing factor
    
    def _set_gripper_state(self, physics, state):
        """Set gripper to initial state (OPEN or CLOSED)."""
        value = self.GRIPPER_MAX if state == 'OPEN' else self.GRIPPER_MIN
        physics.data.ctrl[self.lfinger_actuator_id] = value
        physics.data.ctrl[self.rfinger_actuator_id] = value
        physics.data.qpos[self.lfinger_qpos_addr] = value
        physics.data.qpos[self.rfinger_qpos_addr] = value

    def _get_mpc_warmstart_path(self):
        """Return path for storing MPC warm-start JSON for the mini solver."""
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'am_trajectory_controller', 'config')
        filename = f"mpc_warmstart_main_{self.task_name}.json"
        return os.path.join(base_dir, filename)

    def _load_mpc_warmstart(self):
        """Load warm-start guesses (x,u,lam,pi) into mini solver if JSON exists."""
        import json
        path = self._get_mpc_warmstart_path()
        if not os.path.isfile(path):
            return False
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            solver = self.mpc_planner.mpc.ocp_solver
            N = solver.N
            # States and inequality multipliers: 0..N
            if 'x' in data:
                for k in range(N + 1):
                    solver.set(k, 'x', np.array(data['x'][k], dtype=float))
            if 'lam' in data:
                for k in range(N + 1):
                    solver.set(k, 'lam', np.array(data['lam'][k], dtype=float))
            # Inputs and equality multipliers: 0..N-1
            if 'u' in data:
                for k in range(N):
                    solver.set(k, 'u', np.array(data['u'][k], dtype=float))
            if 'pi' in data:
                for k in range(N):
                    solver.set(k, 'pi', np.array(data['pi'][k], dtype=float))
            print(f"✅ Loaded MPC warm-start from {path}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load MPC warm-start: {e}")
            return False

    def _save_mpc_warmstart_if_missing(self):
        """Save current mini solver warm-start to JSON if file does not already exist."""
        import json
        path = self._get_mpc_warmstart_path()
        if os.path.isfile(path):
            return
        try:
            solver = self.mpc_planner.mpc.ocp_solver
            x = [solver.get(k, 'x').tolist() for k in range(solver.N + 1)]
            u = [solver.get(k, 'u').tolist() for k in range(solver.N)]
            lam = [solver.get(k, 'lam').tolist() for k in range(solver.N + 1)]
            pi = [solver.get(k, 'pi').tolist() for k in range(solver.N)]
            data = { 'x': x, 'u': u, 'lam': lam, 'pi': pi }
            with open(path, 'w') as f:
                json.dump(data, f)
            print(f"💾 Saved MPC warm-start to {path}")
        except Exception as e:
            print(f"⚠️  Failed to save MPC warm-start: {e}")

    def before_step(self, action, physics, trajectory=None):
        # action: [target_ee_pos, target_ee_quat[wxyz], gripper_ctrl]
        target_ee_state = action[:7].copy()
        target_ee_pos = target_ee_state[:3].copy()
        target_ee_quat_wxyz = target_ee_state[3:7]
        
        # --- Get Current State ---
        raw_current_state = physics.data.qpos.copy()
        current_base_pos = raw_current_state[:3]
        current_base_ori = np.array([raw_current_state[4], raw_current_state[5], raw_current_state[6], raw_current_state[3]])
        current_manipulator_joint_angles = raw_current_state[7:11]
        base_euler = R.from_quat(current_base_ori).as_euler('zyx', degrees=False)[::-1]
        
        # --- Coordinate Frame Transformations for EE Target ---
        target_ee_quat_xyzw = np.array([target_ee_quat_wxyz[1], target_ee_quat_wxyz[3], -target_ee_quat_wxyz[2], target_ee_quat_wxyz[0]])
        target_rot = R.from_quat(target_ee_quat_xyzw)
        coord_frame_transform = R.from_rotvec([np.pi/2, 0, 0])
        transformed_rot = coord_frame_transform * target_rot
        target_ee_ori_for_ik = transformed_rot.as_quat()
        target_ee_quat_rotated_wxyz_for_mpc = transformed_rot.as_quat()[[3, 0, 1, 2]]

        # --- Run MPC Planner (always) to get optimal commands ---
        if trajectory is not None:
            if trajectory.shape[0] >= 2:
                last_point = trajectory[-1]
                
                if trajectory.shape[0] < 32:
                    num_repeats = 32 - trajectory.shape[0]
                    repeated_points = np.tile(last_point, (num_repeats, 1))
                    trajectory = np.vstack([trajectory, repeated_points])
            else: 
                trajectory = np.tile(target_ee_state, (32, 1))
            
            total_cost, mpc_outputs = self.compute_trajectory_cost(physics, trajectory)
            force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt = mpc_outputs
        else:
            N = self.mpc_config['mpc']['N']
            repeated_target = np.concatenate([target_ee_pos, target_ee_quat_wxyz])
            single_step_trajectory = np.tile(repeated_target, (N, 1))
            total_cost, mpc_outputs = self.compute_trajectory_cost(physics, single_step_trajectory)
            force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt = mpc_outputs

        # --- Apply MPC Control ---
        self.u_prev = np.concatenate([force_cmd, torque_cmd, arm_angle_cmd])
        
        self.motion_controller.manipulator_controller_link1_pitch.setpoint(arm_angle_cmd[0])
        self.motion_controller.manipulator_controller_link2_pitch.setpoint(arm_angle_cmd[1])
        self.motion_controller.manipulator_controller_link3_pitch.setpoint(arm_angle_cmd[2])
        self.motion_controller.manipulator_controller_link4_roll.setpoint(arm_angle_cmd[3])
        
        current_time = physics.data.time
        arm_torques = np.array([
            self.motion_controller.manipulator_controller_link1_pitch.get_control(current_manipulator_joint_angles[0], current_time),
            self.motion_controller.manipulator_controller_link2_pitch.get_control(current_manipulator_joint_angles[1], current_time),
            self.motion_controller.manipulator_controller_link3_pitch.get_control(current_manipulator_joint_angles[2], current_time),
            self.motion_controller.manipulator_controller_link4_roll.get_control(current_manipulator_joint_angles[3], current_time)
        ])
        
        uam_control = np.concatenate([force_cmd, torque_cmd, arm_torques])

        # --- Final Control Application ---
        final_control = np.concatenate([uam_control, [0.0, 0.0]])
        
        if self.disturbance_enabled:
            if self._gust_timer == 0:
                self._wind_target   = np.random.uniform(*self._wind_range,   size=3)
                self._torque_target = np.random.uniform(*self._torque_range, size=3)

            self._gust_timer = (self._gust_timer + 1) % self._gust_interval

            a = self._gust_alpha
            self._wind   += a * (self._wind_target   - self._wind)
            self._torque += a * (self._torque_target - self._torque)

            final_control[0:3] += self._wind
            final_control[3:6] += self._torque
        
        np.copyto(physics.data.ctrl, final_control)
        
        # Apply UMI-style gripper mapping (0=open, 1=closed)
        grip = np.clip(action[7], 0.0, 1.0)
        gripper_target = self.GRIPPER_MIN + (1.0 - grip) * (self.GRIPPER_MAX - self.GRIPPER_MIN)
        physics.data.ctrl[self.lfinger_actuator_id] = gripper_target
        physics.data.ctrl[self.rfinger_actuator_id] = gripper_target

    def initialize_robots(self, physics, target_ee_pos):
        # reset joint position
        physics.named.data.qpos[:11] = START_UAM_POSE
        self.motion_controller.__init__()
        self.u_prev = np.zeros(10)

        # Note: gripper qpos will be set by _set_gripper_state() after this

        physics.forward()

        target_ee_quat_wxyz = np.array([1, 0, 0, 0])
        reference_pose = np.concatenate([target_ee_pos, target_ee_quat_wxyz])
        
        N = self.mpc_config['mpc']['N']
        reference_trajectory = np.tile(reference_pose, (N, 1))
        
        self._load_mpc_warmstart()
        
        print(f"🎯 Running MPC optimization to find optimal initial configuration...")
        print(f"   Target EE position: {target_ee_pos}")
        print(f"   Target EE quaternion (wxyz): {target_ee_quat_wxyz}")
        
        current_ee_pos_before = physics.data.xpos[self.ee_bid][:3]
        current_ee_quat_before = physics.data.xquat[self.ee_bid]
        print(f"   Current EE position (before): {current_ee_pos_before}")
        print(f"   Current EE quaternion (before): {current_ee_quat_before}")
        
        try:
            max_iterations = 1000
            convergence_threshold = 0.001
            
            for iteration in range(max_iterations):
                total_cost, mpc_outputs = self.compute_trajectory_cost(physics, reference_trajectory)
                force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt = mpc_outputs
                
                optimal_base_pos = p_opt
                optimal_base_euler = base_euler_opt
                optimal_arm_angles = arm_angle_opt
                
                physics.data.qpos[0:3] = optimal_base_pos
                
                from scipy.spatial.transform import Rotation as R
                base_quat_xyzw = R.from_euler('xyz', optimal_base_euler).as_quat()
                base_quat_wxyz = np.array([base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]])
                physics.data.qpos[3:7] = base_quat_wxyz
                
                physics.data.qpos[7:11] = optimal_arm_angles
                
                physics.forward()
                actual_ee_pos = physics.data.xpos[self.ee_bid][:3]
                ee_position_error = np.linalg.norm(actual_ee_pos - target_ee_pos)
                
                print(f"   Iteration {iteration + 1}: Cost: {total_cost:.3f}, EE error: {ee_position_error:.4f}m")
                
                if ee_position_error < convergence_threshold:
                    print(f"✅ Converged after {iteration + 1} iterations!")
                    break
            else:
                print(f"⚠️  Did not converge after {max_iterations} iterations, final error: {ee_position_error:.4f}m")
            
            actual_ee_quat = physics.data.xquat[self.ee_bid]
            
            print(f"✅ MPC optimization complete!")
            print(f"   Final base position: {optimal_base_pos}")
            print(f"   Final base euler: {optimal_base_euler}")
            print(f"   Final arm angles: {optimal_arm_angles}")
            print(f"   Final EE positioning error: {ee_position_error:.4f}m")
            print(f"   Actual EE position (after): {actual_ee_pos}")
            print(f"   Actual EE quaternion (after): {actual_ee_quat}")
            
            self._save_mpc_warmstart_if_missing()
        except Exception as e:
            print(f"❌ MPC initialization failed: {e}")
            print("Falling back to manual translation method")
            
            current_ee_pos = physics.data.xpos[self.ee_bid][:3].copy()
            delta = target_ee_pos - current_ee_pos
            physics.data.qpos[0:3] += delta
            physics.forward()

    def get_qpos(self, physics, task_name="sim_pick_and_place_scripted"):
        qpos_raw = physics.data.qpos.copy()
        uav_base_pos = qpos_raw[:3]
        uav_quat = qpos_raw[3:7]
        uav_quat_xyzw = np.array([uav_quat[1], uav_quat[2], uav_quat[3], uav_quat[0]])
        uav_euler = R.from_quat(uav_quat_xyzw).as_euler('zyx', degrees=False)[::-1]
        manipulator_joints = qpos_raw[7:11]
        fk_input = np.concatenate((uav_base_pos, uav_euler, manipulator_joints))
        ee_state = self.ik_planner.forward_kinematics(fk_input)
        
        ee_bid = physics.model.name2id("ee", "body")
        ee_pos = physics.data.xpos[ee_bid][:3].copy()
        
        ee_quat_xyzw = ee_state[3:7]
        
        current_rot = R.from_quat(ee_quat_xyzw)
        coord_frame_transform_inv = R.from_rotvec([-np.pi/2, 0, 0])
        transformed_rot = coord_frame_transform_inv * current_rot
        
        FIX_QUAT_XYZW = np.array([-0.5, -0.5,  0.5,  0.5])
        fix_rot = R.from_quat(FIX_QUAT_XYZW)
        final_rot = fix_rot * transformed_rot
        
        final_quat_xyzw = final_rot.as_quat()
        final_quat_wxyz = np.array([final_quat_xyzw[3], final_quat_xyzw[0], 
                                     final_quat_xyzw[1], final_quat_xyzw[2]])
        final_quat_wxyz = np.array([final_quat_wxyz[0], -final_quat_wxyz[3], final_quat_wxyz[2], -final_quat_wxyz[1]])
        
        transformed_ee_state = np.concatenate([ee_pos, final_quat_wxyz])
        gripper_qpos = [1-UAM_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[12])]
        return np.concatenate([transformed_ee_state, gripper_qpos])
        
    @staticmethod
    def get_env_state(physics):
        """Return full environment state (all qpos)."""
        return physics.data.qpos.copy()

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics, self.task_name)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = {name: physics.render(480, 640, camera_id=name)
                         for name in self.camera_names}
        
        WALL_STATE_IDX = 13
        obs["wall_pos"] = physics.data.qpos[WALL_STATE_IDX:WALL_STATE_IDX+3].copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError

    def compute_trajectory_cost(self, physics, trajectory):
        """Compute MPC cost for a given reference EE trajectory without applying any control.

        Args:
            physics: MuJoCo physics instance from the environment.
            trajectory (np.ndarray): Waypoint array of shape (N, 7)
                [x,y,z, qw,qx,qy,qz].

        Returns:
            float: total MPC cost for the trajectory.
            tuple: (force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt)
                   All optimization outputs for reuse.
        """
        if trajectory is None or len(trajectory) == 0:
            return np.inf, None

        raw_state = physics.data.qpos.copy()
        current_base_pos = raw_state[:3]
        current_base_ori = np.array([raw_state[4], raw_state[5], raw_state[6], raw_state[3]])
        base_euler = R.from_quat(current_base_ori).as_euler('zyx', degrees=False)[::-1]
        current_manipulator_joint_angles = raw_state[7:11]
        
        base_vel = physics.data.qvel[0:3].copy()
        base_ang_vel = physics.data.qvel[3:6].copy()

        ee_pos_ref = trajectory[:, :3]
        
        ee_quat_ref = []
        coord_frame_transform = R.from_rotvec([np.pi/2, 0, 0])
        for quat_wxyz in trajectory[:, 3:7]:
            quat_xyzw = np.array([
                quat_wxyz[1],
                quat_wxyz[3],
                -quat_wxyz[2],
                quat_wxyz[0]
            ])
            rot = R.from_quat(quat_xyzw)
            transformed_rot = coord_frame_transform * rot
            quat_xyzw_t = transformed_rot.as_quat()
            quat_wxyz_t = quat_xyzw_t[[3, 0, 1, 2]]
            ee_quat_ref.append(quat_wxyz_t)
        ee_quat_ref = np.array(ee_quat_ref)
        # Run MPC planner with updated interface for 16-state system
        force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt, total_cost = self.mpc_planner.optimize(
            current_base_pos.copy(),
            base_vel,
            current_manipulator_joint_angles,
            base_euler,
            base_ang_vel,  # Add angular velocity for 16-state MPC
            ee_pos_ref,
            ee_quat_ref,
            self.u_prev,
        )
        return float(total_cost), (force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt)

    def compute_trajectory_cost_with_tracking(self, physics, trajectory):
        """Compute both total MPC cost and tracking-only cost for a given reference EE trajectory.

        Args:
            physics: MuJoCo physics instance from the environment.
            trajectory (np.ndarray): Waypoint array of shape (N, 7)
                [x,y,z, qw,qx,qy,qz].

        Returns:
            tuple: (total_cost, tracking_cost, predicted_ee_trajectory)
                   total_cost: float - total MPC cost
                   tracking_cost: float - tracking-only cost
                   predicted_ee_trajectory: np.ndarray - MPC predicted EE poses (N+1, 7)
        """
        if trajectory is None or len(trajectory) == 0:
            return np.inf, np.inf, None

        raw_state = physics.data.qpos.copy()
        current_base_pos = raw_state[:3]
        current_base_ori = np.array([raw_state[4], raw_state[5], raw_state[6], raw_state[3]])
        base_euler = R.from_quat(current_base_ori).as_euler('zyx', degrees=False)[::-1]
        current_manipulator_joint_angles = raw_state[7:11]
        
        base_vel = physics.data.qvel[0:3].copy()
        base_ang_vel = physics.data.qvel[3:6].copy()

        ee_pos_ref = trajectory[:, :3]
        
        ee_quat_ref = []
        coord_frame_transform = R.from_rotvec([np.pi/2, 0, 0])
        for quat_wxyz in trajectory[:, 3:7]:
            quat_xyzw = np.array([
                quat_wxyz[1],
                quat_wxyz[3],
                -quat_wxyz[2],
                quat_wxyz[0]
            ])
            rot = R.from_quat(quat_xyzw)
            transformed_rot = coord_frame_transform * rot
            quat_xyzw_t = transformed_rot.as_quat()
            quat_wxyz_t = quat_xyzw_t[[3, 0, 1, 2]]
            ee_quat_ref.append(quat_wxyz_t)
        ee_quat_ref = np.array(ee_quat_ref)

        # Run MPC planner with tracking cost extraction
        force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt, total_cost, tracking_cost, predicted_ee_trajectory = self.mpc_planner.optimize_with_tracking_cost(
            current_base_pos.copy(),
            base_vel,
            current_manipulator_joint_angles,
            base_euler,
            base_ang_vel,  # Add angular velocity for 16-state MPC
            ee_pos_ref,
            ee_quat_ref,
            self.u_prev,
        )
        
        return float(total_cost), float(tracking_cost), predicted_ee_trajectory

    def compute_trajectory_cost_with_tracking_gradient(self, physics, trajectory):
        """Compute the gradient of the tracking cost for a given reference EE trajectory.

        Args:
            physics: MuJoCo physics instance from the environment.
            trajectory (np.ndarray): Waypoint array of shape (N, 7)
                [x,y,z, qw,qx,qy,qz].

        Returns:
            tuple: (total_cost, tracking_cost, gradient, mpc_x_opt)
                - total_cost (float): Total MPC cost
                - tracking_cost (float): Tracking component of MPC cost  
                - gradient (np.ndarray): Gradient of the tracking cost with respect to the initial state.
                - mpc_x_opt (np.ndarray): MPC optimized states for end-effector position extraction
        """
        raw_state = physics.data.qpos.copy()
        current_base_pos = raw_state[:3]
        current_base_ori = np.array([raw_state[4], raw_state[5], raw_state[6], raw_state[3]])
        base_euler = R.from_quat(current_base_ori).as_euler('zyx', degrees=False)[::-1]
        current_manipulator_joint_angles = raw_state[7:11]
        
        base_vel = physics.data.qvel[0:3].copy()
        base_ang_vel = physics.data.qvel[3:6].copy()

        ee_pos_ref = trajectory[:, :3]
        
        ee_quat_ref = []
        coord_frame_transform = R.from_rotvec([np.pi/2, 0, 0])
        for quat_wxyz in trajectory[:, 3:7]:
            quat_xyzw = np.array([
                quat_wxyz[1],
                quat_wxyz[3],
                -quat_wxyz[2],
                quat_wxyz[0]
            ])
            rot = R.from_quat(quat_xyzw)
            transformed_rot = coord_frame_transform * rot
            quat_xyzw_t = transformed_rot.as_quat()
            quat_wxyz_t = quat_xyzw_t[[3, 0, 1, 2]]
            ee_quat_ref.append(quat_wxyz_t)
        ee_quat_ref = np.array(ee_quat_ref)
        
        # Run MPC planner with tracking cost extraction
        total_cost, tracking_cost, tracking_gradient, mpc_x_opt = self.mpc_planner_mini.optimize_with_tracking_gradient(
            current_base_pos.copy(),
            base_vel,
            current_manipulator_joint_angles,
            base_euler,
            base_ang_vel,  # Add angular velocity for 16-state MPC
            ee_pos_ref,
            ee_quat_ref,
            self.u_prev,
        )
        
        return total_cost, tracking_cost, tracking_gradient, mpc_x_opt

# ============================================================================
# Task Classes 
# ============================================================================

class RotateValveTask:
    """Shared logic for rotate valve task across all robot types."""
    
    max_reward = 1.0
    
    def _setup_valve_objects(self, physics):
        """Initialize valve-specific object IDs using dynamic lookups."""
        self._valve_hinge_jid = physics.model.name2id("valve_joint", "joint")
        self._valve_hinge_addr = physics.model.jnt_qposadr[self._valve_hinge_jid]
        self._valve_body_id = physics.model.name2id("valve", "body")
    
    def sample_valve_pose(self):
        """Sample valve position within standard range."""
        ranges = np.vstack([[0.975, 0.975], [-0.15, 0.15], [1.15, 1.25]])
        return np.random.uniform(ranges[:, 0], ranges[:, 1])
    
    def _initialize_valve_state(self, physics):
        """Initialize valve position and angle."""
        valve_pos = self.sample_valve_pose()
        physics.model.body_pos[self._valve_body_id] = valve_pos
        physics.data.qpos[self._valve_hinge_addr] = 1.57
    
    def _get_initial_gripper_state(self):
        """Return initial gripper state for this task (OPEN for valve rotation)."""
        return 'OPEN'
    
    def get_reward(self, physics):
        """Return 1.0 if valve rotated >= π radians beyond initial position."""
        angle = physics.data.qpos[self._valve_hinge_addr]
        return 1.0 if (angle - 1.57) / np.pi >= 1.0 else 0.0

class PegInHoleTask:
    """Shared logic for peg-in-hole task across all robot types."""
    
    max_reward = 1.0
    
    def _setup_hole_objects(self, physics):
        """Initialize wall/hole object IDs using dynamic lookups."""
        wall_jid = physics.model.name2id("wall_joint", "joint")
        self._wall_qpos_addr = physics.model.jnt_qposadr[wall_jid]
    
    def sample_hole_pose(self):
        """Sample wall position (determines hole location)."""
        ranges = np.vstack([[1.0, 1.0], [-0.3, 0.3], [0.4, 0.4]])
        wall_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        wall_quat = np.array([1, 0, 0, 0])
        return np.concatenate([wall_position, wall_quat])
    
    def _initialize_hole_state(self, physics):
        """Initialize wall position."""
        wall_pose = self.sample_hole_pose()
        np.copyto(physics.data.qpos[self._wall_qpos_addr:self._wall_qpos_addr+3], wall_pose[:3])
    
    def _get_initial_gripper_state(self):
        """Return initial gripper state for this task (CLOSED - holding peg)."""
        return 'CLOSED'

    def get_reward(self, physics):
        """Return 1.0 if peg touches the pin (reward box)."""
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            all_contact_pairs.append((name_geom_1, name_geom_2))
        
        pin_touched = ("obj_geom", "reward_box_geom") in all_contact_pairs
        return 1.0 if pin_touched else 0.0

class PickAndPlaceTask:
    """Shared logic for pick-and-place task across all robot types."""
    
    max_reward = 1.0
    
    def _setup_pnp_objects(self, physics):
        """Initialize can/bowl object IDs using dynamic lookups."""
        self._can_bid = physics.model.name2id("can", "body")
        self._bowl_bid = physics.model.name2id("bowl", "body")
        
        # Dynamic joint address lookups
        can_jid = physics.model.name2id("can", "joint")
        bowl_jid = physics.model.name2id("bowl", "joint")
        self._can_qpos_addr = physics.model.jnt_qposadr[can_jid]
        self._bowl_qpos_addr = physics.model.jnt_qposadr[bowl_jid]
        
        # For UAM/UR10e gravity forces
        self._can_mass = physics.model.body_mass[self._can_bid]
        self._bowl_mass = physics.model.body_mass[self._bowl_bid]
        self._grav_vec = np.array([0.0, 0.0, -0.01])
        
    def sample_can_bowl_poses(self):
        """Sample can and bowl XY positions."""
        can_base_xy = np.array([1.0, 0.3])
        bowl_base_xy = np.array([1.0, 0.1])
        can_xy = can_base_xy + np.random.uniform(-0.05, 0.05, size=2)
        bowl_xy = bowl_base_xy + np.random.uniform(-0.05, 0.05, size=2)
        return can_xy, bowl_xy
    
    def _initialize_pnp_state(self, physics):
        """Initialize can and bowl positions."""
        can_xy, bowl_xy = self.sample_can_bowl_poses()
        np.copyto(physics.data.qpos[self._can_qpos_addr:self._can_qpos_addr+2], can_xy)
        np.copyto(physics.data.qpos[self._bowl_qpos_addr:self._bowl_qpos_addr+2], bowl_xy)
    
    def _get_initial_gripper_state(self):
        """Return initial gripper state for this task (OPEN for pick and place)."""
        return 'OPEN'
    
    def _apply_forces(self, physics):
        """Apply gravity forces to can and bowl (for UAM/UR10e tasks)."""
        physics.data.xfrc_applied[self._can_bid, :3] = self._grav_vec
        physics.data.xfrc_applied[self._bowl_bid, :3] = self._grav_vec * 10
    
    def get_reward(self, physics):
        """Return 1.0 if can within 7cm of bowl."""
        can_xyz = physics.data.xpos[self._can_bid][:3]
        bowl_xyz = physics.data.xpos[self._bowl_bid][:3]
        return 1.0 if np.linalg.norm(can_xyz - bowl_xyz) < 0.07 else 0.0

class CabinetTask:
    """Shared logic for cabinet task across all robot types."""
    
    max_reward = 1.0
    
    # Success region constants
    CAB_TOP_X_RANGE = (0.6, 1.2)
    CAB_TOP_Y_RANGE = (-0.6, 0.6)
    CAB_TOP_Z_THRESH = 1.4
    
    def _setup_cabinet_objects(self, physics):
        """Initialize cabinet/can object IDs using dynamic lookups."""
        self._can_bid = physics.model.name2id("can", "body")
        
        # Dynamic joint address lookup
        can_jid = physics.model.name2id("can", "joint")
        self._can_qpos_addr = physics.model.jnt_qposadr[can_jid]
        
        # For UAM/UR10e gravity
        self._grav_vec = np.array([0, 0, -0.01])
    
    def sample_can_pose(self):
        """Sample can XY position."""
        can_base_xy = np.array([0.7, 0.25])
        return can_base_xy + np.random.uniform(-0.05, 0.05, size=2)
    
    def _initialize_cabinet_state(self, physics):
        """Initialize can position."""
        can_xy = self.sample_can_pose()
        np.copyto(physics.data.qpos[self._can_qpos_addr:self._can_qpos_addr+2], can_xy)
    
    def _get_initial_gripper_state(self):
        """Return initial gripper state for this task (OPEN for cabinet)."""
        return 'OPEN'
    
    def _apply_forces(self, physics):
        """Apply gravity force to can (for UAM/UR10e tasks)."""
        physics.data.xfrc_applied[self._can_bid, :3] = self._grav_vec
    
    def get_reward(self, physics):
        """Return 1.0 if can is on cabinet top surface."""
        can_pos = physics.data.xpos[self._can_bid]
        x_ok = self.CAB_TOP_X_RANGE[0] <= can_pos[0] <= self.CAB_TOP_X_RANGE[1]
        y_ok = self.CAB_TOP_Y_RANGE[0] <= can_pos[1] <= self.CAB_TOP_Y_RANGE[1]
        z_ok = can_pos[2] >= self.CAB_TOP_Z_THRESH
        return 1.0 if (x_ok and y_ok and z_ok) else 0.0

# ============================================================================
# Robot-Task Classes 
# ============================================================================

class PegInHoleEETask_4DoF(PegInHoleTask, UAMBaseTask):
    """Peg-in-hole task for 4-DoF drone + arm (MPC control)."""
    
    def __init__(self, physics, task_name="sim_peg_in_hole_scripted", random=None, camera_names=["ee"], disturbance_enabled=False):
        UAMBaseTask.__init__(self, physics, task_name, random=random, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
        self._setup_hole_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and hole state."""
        self.initialize_robots(physics, self.DEFAULT_REF_EE_POS)
        self._initialize_hole_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

class PickAndPlaceEETask_4DoF(PickAndPlaceTask, UAMBaseTask):
    """Pick-and-place task for 4-DoF drone + arm (MPC control)."""
    
    def __init__(self, physics, task_name="sim_pick_and_place_scripted", random=None, camera_names=["ee"], disturbance_enabled=False):
        UAMBaseTask.__init__(self, physics, task_name, random=random, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
        self._setup_pnp_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and pick-and-place objects."""
        self.initialize_robots(physics, self.DEFAULT_REF_EE_POS)
        self._initialize_pnp_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

    def before_step(self, action, physics, trajectory=None):
        """Run parent control and apply gravity forces."""
        super().before_step(action, physics, trajectory)
        self._apply_forces(physics)

class RotateValveEETask_4DoF(RotateValveTask, UAMBaseTask):
    """Rotate valve task for 4-DoF drone + arm (MPC control)."""
    
    def __init__(self, physics, task_name="sim_rotate_valve_scripted", random=None, camera_names=["ee"], disturbance_enabled=False):
        UAMBaseTask.__init__(self, physics, task_name, random=random, camera_names=camera_names, disturbance_enabled=disturbance_enabled)
        self._setup_valve_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and valve state."""
        self.initialize_robots(physics, self.DEFAULT_REF_EE_POS)
        self._initialize_valve_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()
    
class CabinetEETask_4DoF(CabinetTask, UAMBaseTask):
    """Cabinet pick-and-place task for 4-DoF drone + arm (MPC control).""" 
    def __init__(self, physics, task_name="sim_cabinet_umi", random=None, camera_names=["ee"], disturbance_enabled=False, acados_build_dir=None):
        UAMBaseTask.__init__(self, physics, task_name, random=random, camera_names=camera_names, disturbance_enabled=disturbance_enabled, acados_build_dir=acados_build_dir)
        self._setup_cabinet_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and cabinet state."""
        self.initialize_robots(physics, self.DEFAULT_REF_EE_POS)
        self._initialize_cabinet_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

    def before_step(self, action, physics, trajectory=None):
        """Run parent control and apply gravity forces."""
        super().before_step(action, physics, trajectory)
        self._apply_forces(physics)


class CabinetUR10eEETask(CabinetTask, UR10eBaseTask):
    """Cabinet pick-and-place task for UR10e + UMI gripper (IK control)."""

    DEFAULT_REF_EE_POS = np.array([0.1, 0.3, 1.2])
    BASE_POS = np.array([-0.2, -0.1, 0.9])

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UR10eBaseTask.__init__(self, physics, random=random, camera_names=camera_names)
        self._setup_cabinet_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and cabinet state."""
        self._initialize_ur10e_robot(physics, self.BASE_POS, self.DEFAULT_REF_EE_POS)
        self._initialize_cabinet_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

    def before_step(self, action, physics, trajectory=None):
        """Run parent control and apply gravity forces."""
        super().before_step(action, physics, trajectory)
        self._apply_forces(physics)

class PickAndPlaceUR10eEETask(PickAndPlaceTask, UR10eBaseTask):
    """Pick-and-place task for UR10e + UMI gripper (IK control)."""

    DEFAULT_REF_EE_POS = np.array([0.4, 0.3, 1.0])
    BASE_POS = np.array([0.4, 0.9, 0.9])

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UR10eBaseTask.__init__(self, physics, random=random, camera_names=camera_names)
        self._setup_pnp_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and pick-and-place objects."""
        self._initialize_ur10e_robot(physics, self.BASE_POS, self.DEFAULT_REF_EE_POS)
        self._initialize_pnp_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

    def before_step(self, action, physics, trajectory=None):
        """Run parent control and apply gravity forces."""
        super().before_step(action, physics, trajectory)
        self._apply_forces(physics)

class RotateValveUR10eEETask(RotateValveTask, UR10eBaseTask):
    """Rotate valve task for UR10e + UMI gripper (IK control)."""

    DEFAULT_REF_EE_POS = np.array([0.0, 0.0, 1.2])
    BASE_POS = np.array([0.3, -0.4, 0.6])

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UR10eBaseTask.__init__(self, physics, random=random, camera_names=camera_names)
        self._setup_valve_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and valve state."""
        self._initialize_ur10e_robot(physics, self.BASE_POS, self.DEFAULT_REF_EE_POS)
        self._initialize_valve_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

class PegInHoleUR10eEETask(PegInHoleTask, UR10eBaseTask):
    """Peg-in-hole task for UR10e + UMI gripper (IK control)."""

    DEFAULT_REF_EE_POS = np.array([0.0, 0.0, 1.2])
    BASE_POS = np.array([0.3, -0.4, 1.2])

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UR10eBaseTask.__init__(self, physics, random=random, camera_names=camera_names)
        self._setup_hole_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and hole state."""
        self._initialize_ur10e_robot(physics, self.BASE_POS, self.DEFAULT_REF_EE_POS)
        self._initialize_hole_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()


class CabinetUMIOracleTask(CabinetTask, UMIOracleBaseTask):
    """Cabinet pick-and-place task for stand-alone UMI robot (mocap control)."""

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UMIOracleBaseTask.__init__(self, physics, random, camera_names, disturbance_enabled)
        self._setup_cabinet_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and cabinet state."""
        self._initialize_episode_start(physics)
        self._initialize_cabinet_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

    def before_step(self, action, physics, trajectory=None):
        """Run parent control and apply gravity forces."""
        super().before_step(action, physics, trajectory)
        self._apply_forces(physics)

class PegInHoleUMIOracleTask(PegInHoleTask, UMIOracleBaseTask):
    """Peg-in-hole task for stand-alone UMI robot (mocap control)."""

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UMIOracleBaseTask.__init__(self, physics, random, camera_names, disturbance_enabled)
        self._setup_hole_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and hole state."""
        self._initialize_episode_start(physics)
        self._initialize_hole_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

class PickAndPlaceUMIOracleTask(PickAndPlaceTask, UMIOracleBaseTask):
    """Pick-and-place task for stand-alone UMI robot (mocap control)."""

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UMIOracleBaseTask.__init__(self, physics, random, camera_names, disturbance_enabled)
        self._setup_pnp_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and pick-and-place objects."""
        self._initialize_episode_start(physics)
        self._initialize_pnp_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()

    def before_step(self, action, physics, trajectory=None):
        """Run parent control and apply gravity forces."""
        super().before_step(action, physics, trajectory)
        self._apply_forces(physics)

class RotateValveUMIOracleTask(RotateValveTask, UMIOracleBaseTask):
    """Rotate valve task for stand-alone UMI robot (mocap control)."""

    def __init__(self, physics, random=None, camera_names=("ee",), disturbance_enabled=False):
        UMIOracleBaseTask.__init__(self, physics, random, camera_names, disturbance_enabled)
        self._setup_valve_objects(physics)

    def initialize_episode(self, physics):
        """Initialize robot and valve state."""
        self._initialize_episode_start(physics)
        self._initialize_valve_state(physics)
        self._set_gripper_state(physics, self._get_initial_gripper_state())
        physics.forward()
