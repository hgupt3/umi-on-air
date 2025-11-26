"""
Evaluation script for UMI diffusion policies in simulation environments.
Supports 3D visualization, disturbances, and guided diffusion.
"""
import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, Any, Optional, Tuple

import cv2
import dill
import hydra
import mujoco.viewer
import numpy as np
import tkinter as tk
import torch
from omegaconf import OmegaConf
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
from Xlib import display

sys.path.append("../universal_manipulation_interface")
sys.path.append("..")

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from universal_manipulation_interface.umi.real_world.real_inference_util import (
    get_real_umi_action, 
    get_real_umi_obs_dict
)

from constants import DT, STATE_DIM, ACTION_DIM
from utils import set_seed
from visualize_episodes import save_videos
from ee_sim_env import make_ee_sim_env
from imitate_tracking_utils import (
    EpisodeMetricsTracker, 
    ExperimentSummary, 
    save_diffusion_analysis,
    pad_trajectory_segment,
    process_completed_diffusion_analysis,
    finalize_remaining_analysis
)
from imitate_sim_utils import (
    get_image, 
    convert_action_buffer_to_trajectory, 
    extract_trajectory_positions, 
    add_trajectory_to_viewer, 
    get_ee_pos, 
    interpolate_action_sequence, 
    extrapolate_action_buffer, 
    TrajectoryAwareEnvWrapper, 
    add_disturbance_arrows,
    quat_wxyz_to_axis_angle,
    extract_robot_state,
    prepare_umi_image_observation,
    ObservationBufferManager
)

OmegaConf.register_new_resolver("eval", eval, replace=True)

# Constants
STEPS_UNTIL_STOP_ON_SUCCESS = int(0.1 / DT)
ACTUAL_TRAJECTORY_SAMPLE_RATE = 32
GREY_DOT_HISTORY_SIZE = 16
STATUS_PRINT_INTERVAL = 32
PAUSE_UPDATE_RATE = 0.05
DDIM_INFERENCE_STEPS = 16
VIEWER_CLEANUP_DELAY = 0.1

# MPC cost thresholds for early restart / late failure
MPC_COST_THRESHOLD = 10.0        # If MPC cost exceeds this, take action
MPC_RESTART_WINDOW = 500        # Timesteps before switching from restart to failure

def is_window_focused(window_keywords):
    """Check if a window matching keywords has focus"""
    if not window_keywords:
        return False
    
    display_obj = display.Display()
    window = display_obj.get_input_focus().focus
    wmname = window.get_wm_name()
    
    if wmname is None:
        return False
    
    title_lower = wmname.lower()
    return any(kw.lower() in title_lower for kw in window_keywords)


def should_exit_episode(key_flags: Dict[str, bool]) -> bool:
    """Check if episode should exit based on user input."""
    return key_flags['quit'] or key_flags['discard'] or key_flags['save']


def build_config(args: Dict[str, Any], task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build configuration dict from args and task config."""
    return {
        'ckpt_dir': args['ckpt_dir'],
        'episode_len': task_config['episode_len'],
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'onscreen_render': args['onscreen_render'],
        'use_3d_viewer': args['use_3d_viewer'],
        'policy_config': {'use_workspace_loading': True},
        'task_name': args['task_name'],
        'camera_names': task_config['camera_names'],
        'load_ckpt_file_path': args['load_ckpt_file_path'],
        'num_rollouts': args['num_rollouts'],
        'output_dir': args['output_dir'],
        'disturb': args['disturb'],
        'guidance': args['guidance'],
        'guided_steps': args['guided_steps'],
        'log_diffusion': args['log_diffusion'],
        'acados_build_dir': args['acados_build_dir'],
        'scale': args['scale'],
        'resume': args['resume']
    }


def auto_detect_checkpoint_dir(task_name: str) -> Optional[str]:
    """
    Auto-detect checkpoint directory based on task name.
    
    Task naming format: EMBODIMENT_TASK (e.g., 'uam_cabinet', 'ur10e_peg')
    Looks for checkpoints in: checkpoints/umi_{TASK}/
    
    Args:
        task_name: Task name in format 'embodiment_task'
    
    Returns:
        Path to checkpoint directory if found, None otherwise
    """
    # Extract task name (part after embodiment prefix)
    # E.g., 'uam_cabinet' -> 'cabinet', 'ur10e_peg' -> 'peg'
    parts = task_name.split('_', 1)
    if len(parts) < 2:
        return None
    
    task = parts[1]  # Get task name (cabinet, peg, pick, valve)
    
    # Get workspace root (2 levels up from policy_learning)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Look for checkpoints/umi_{task}/ directory
    checkpoint_dir = os.path.join(workspace_root, 'checkpoints', f'umi_{task}')
    
    if os.path.exists(checkpoint_dir):
        print(f"✨ Auto-detected checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir
    
    return None


def get_timestamp() -> str:
    """Get current timestamp in YYYY-MM-DD_HH-MM-SS format"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def find_latest_timestamped_dir(base_path: str) -> Optional[str]:
    """Find the most recent timestamped directory"""
    if not os.path.exists(base_path):
        return None
    dirs = [d for d in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, d))]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return os.path.join(base_path, dirs[0])


def resolve_checkpoint_path(ckpt_dir: str, provided_path: Optional[str]) -> Optional[str]:
    """Resolve checkpoint path with auto-detection."""
    if provided_path is not None:
        return provided_path
    
    auto_ckpt_path = os.path.join(ckpt_dir, 'checkpoints', 'latest.ckpt')
    if os.path.exists(auto_ckpt_path):
        print(f"Auto-detected checkpoint: {auto_ckpt_path}")
        return auto_ckpt_path
    return None


def handle_episode_crash(episode_tracker: EpisodeMetricsTracker) -> None:
    """Handle episode crash by cleaning up episode directory."""
    if os.path.exists(episode_tracker.episode_dir):
        import shutil
        shutil.rmtree(episode_tracker.episode_dir, ignore_errors=True)


def handle_pause_mode(key_flags: Dict[str, bool], use_3d_viewer: bool, viewer) -> bool:
    """Handle pause mode - returns True if should break episode loop."""
    print("⏸️  Simulation paused. Press SPACE to resume.")
    while key_flags['paused']:
        if use_3d_viewer and viewer is not None:
            viewer.sync()
        time.sleep(PAUSE_UPDATE_RATE)
        
        if key_flags['quit']:
            print("Exiting program per user request")
            return True
        if key_flags['discard'] or key_flags['save']:
            print(f"User requested {'discard' if key_flags['discard'] else 'save'} and reset")
            return True
    
    print("▶️  Simulation resumed.")
    return False


def render_3d_viewer(viewer, success_detected: bool, success_counter: int, 
                     rollout_id: int, step: int, max_timesteps: int) -> None:
    """Render 3D viewer with status message."""
    if success_detected:
        remaining = max(0, (STEPS_UNTIL_STOP_ON_SUCCESS - success_counter) * DT)
        status_msg = f"SUCCESS stopping in {remaining:.1f}s"
    else:
        status_msg = f"EP{rollout_id} {step+1}/{max_timesteps} ESC-quit"
    
    if step % STATUS_PRINT_INTERVAL == 0:
        print(status_msg)
    
    viewer.sync()


def render_opencv_preview(env, onscreen_cam: str, screen_width: int, screen_height: int, 
                         window_name: str, success_detected: bool, success_counter: int,
                         rollout_id: int, step: int, max_timesteps: int, 
                         fullscreen_mode: bool) -> Tuple[bool, bool]:
    """Render OpenCV preview window. Returns (exit_program, new_fullscreen_mode)."""
    frame = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (screen_width, screen_height))
    
    if success_detected:
        remaining = max(0, (STEPS_UNTIL_STOP_ON_SUCCESS - success_counter) * DT)
        overlay = f"SUCCESS stopping in {remaining:.1f}s"
    else:
        overlay = f"EP{rollout_id} {step+1}/{max_timesteps} ESC-quit"
    
    cv2.putText(frame_bgr, overlay, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
    cv2.imshow(window_name, frame_bgr)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        return True, fullscreen_mode
    elif key in (ord('f'), ord('F')):
        if fullscreen_mode:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return False, not fullscreen_mode
    
    return False, fullscreen_mode


def run_policy_inference(policy, obs_dict, env_obs_stacked, action_pose_repr, query_frequency,
                        env, log_diffusion, has_acados_cost):
    """
    Run policy inference and return action buffer.
    
    Returns:
        tuple: (action_buffer, reference_trajectory, mpc_cost)
            - mpc_cost is 0.0 if ACADOS cost is not available
    """
    with torch.no_grad():
        res = policy.predict_action(obs_dict)
        raw_action = res['action_pred'][0].detach().to('cpu').numpy()
        raw_action[...,:3] -= raw_action[0][:3]
        action = get_real_umi_action(raw_action, env_obs_stacked, action_pose_repr)

    action_buffer = interpolate_action_sequence(action, query_frequency)
    reference_trajectory = convert_action_buffer_to_trajectory(action_buffer)
    
    # Compute MPC cost if available (for restart/failure detection)
    mpc_cost = 0.0
    if has_acados_cost and reference_trajectory is not None:
        try:
            total_cost, tracking_cost, _ = env.task.compute_trajectory_cost_with_tracking(
                env._physics, reference_trajectory
            )
            mpc_cost = total_cost
        except Exception:
            mpc_cost = 0.0
    
    return action_buffer, reference_trajectory, mpc_cost


def update_3d_viewer_trajectory(policy, use_3d_viewer, viewer, robot_pos, query_frequency, 
                                actions_consumed, pregrad_full_buffer):
    """
    Update 3D viewer with pre-gradient trajectory visualization.
    
    Returns:
        tuple: (current_pregrad_trajectory, updated_pregrad_full_buffer)
    """
    if not (use_3d_viewer and viewer is not None):
        return None, pregrad_full_buffer
    
    current_ee_pos = robot_pos
    
    if hasattr(policy, 'last_pregrad_trajectory') and policy.last_pregrad_trajectory is not None:
        pregrad_full = policy.interpolate_trajectory(policy.last_pregrad_trajectory, target_horizon=query_frequency)
        pregrad_full[:,:3] -= pregrad_full[0,:3]
        pregrad_full[:,:3] += robot_pos
        pregrad_full_buffer = [pregrad_full[i] for i in range(pregrad_full.shape[0])]
        
        remaining_pregrad_actions = pregrad_full_buffer[actions_consumed:]
        if remaining_pregrad_actions:
            current_pregrad_trajectory = extract_trajectory_positions(remaining_pregrad_actions, current_ee_pos)
        else:
            current_pregrad_trajectory = None
    else:
        current_pregrad_trajectory = None
    
    return current_pregrad_trajectory, pregrad_full_buffer


def main(args: Dict[str, Any]) -> None:
    """Main entry point for evaluation."""
    from constants import SIM_TASK_CONFIGS
    
    task_config = SIM_TASK_CONFIGS[args['task_name']]
    config = build_config(args, task_config)
    
    ckpt_name = 'policy_best.ckpt'
    success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
    print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    
    sys.exit(0)

def make_policy(policy_config: Dict[str, Any]):
    """Create and configure UMI policy from workspace."""
    payload = policy_config['payload']
    cfg = payload['cfg']
    
    # Check if guided inference is requested
    guidance = policy_config.get('guidance', 0.0)
    scale = policy_config.get('scale', 0.0)
    guided_steps = policy_config.get('guided_steps', 0)
    
    use_guided_inference = (guidance > 0 or scale > 0)
    
    # Auto-switch to EADP policy if guided inference is requested
    # Training always uses vanilla policy, but inference can use EADP for guidance
    if use_guided_inference:
        original_target = cfg.policy._target_
        if 'diffusion_unet_timm_policy' in original_target and 'eadp' not in original_target:
            cfg.policy._target_ = original_target.replace(
                'diffusion_unet_timm_policy',
                'eadp_unet_timm_policy'
            )
            print(f"🌟 Auto-switching to EADP policy for guided inference")
            print(f"   Original: {original_target}")
            print(f"   Switched: {cfg.policy._target_}")
    
    print(f"Hydra workspace target: {cfg._target_}")
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.num_inference_steps = DDIM_INFERENCE_STEPS
    
    # Set policy parameters from config
    if 'action_pose_repr' in policy_config:
        policy.action_pose_repr = policy_config['action_pose_repr']
    
    # Set guidance parameters if policy supports them
    if hasattr(policy, 'scale'):
        policy.scale = scale
    if hasattr(policy, 'guidance'):
        policy.guidance = guidance
    if hasattr(policy, 'guided_steps'):
        policy.guided_steps = guided_steps
    
    return policy


def eval_bc(config, ckpt_name, save_episode=True):
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    onscreen_render = config['onscreen_render']
    use_3d_viewer = config['use_3d_viewer']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    onscreen_cam = 'ee'
    load_ckpt_file_path = config['load_ckpt_file_path']
    disturbance_enabled = config.get('disturb', False)
    log_diffusion = config['log_diffusion']
    guided_steps_param = config.get('guided_steps')
    acados_build_dir = config.get('acados_build_dir', None)
    set_seed(np.random.randint(1000000))

    # Show disturbance status
    if disturbance_enabled:
        print("🌪️  DISTURBANCES ENABLED: Simulation will include environmental disturbances")
    else:
        print("🔒 DISTURBANCES DISABLED: Clean simulation environment")

    # Load UMI payload
    # Use load_ckpt_file_path if provided, otherwise use default ckpt_dir/ckpt_name
    if load_ckpt_file_path is not None:
        ckpt_path = load_ckpt_file_path
    else:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    # Load payload with dill exactly like eval_real.py
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    policy_config['payload'] = payload
    # Pass diffusion logging flag to policy
    policy_config['log_diffusion'] = config['log_diffusion']
    
    # Extract UMI-specific parameters from config
    shape_meta = cfg.task.shape_meta
    img_obs_horizon = shape_meta.obs.camera0_rgb.horizon  # Image observation horizon
    low_dim_obs_horizon = shape_meta.obs.robot0_eef_pos.horizon  # Low-dim observation horizon
    camera_shape = shape_meta.obs.camera0_rgb.shape  # Get camera resolution (C, H, W)
    action_horizon = shape_meta.action.horizon
    obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    obs_down_sample_steps = cfg.task.obs_down_sample_steps
    
    print(f"UMI image obs horizon: {img_obs_horizon}")
    print(f"UMI low-dim obs horizon: {low_dim_obs_horizon}")
    print(f"UMI camera shape: {camera_shape}")
    print(f"UMI action horizon: {action_horizon}")
    print(f"obs_pose_repr: {obs_pose_repr}")
    print(f"action_pose_repr: {action_pose_repr}")
    print(f"obs_down_sample_steps: {obs_down_sample_steps}")

    # Pass action_pose_repr to policy config for coordinate conversion
    policy_config['action_pose_repr'] = action_pose_repr
    # Pass guidance parameters to policy config for auto-switching and initialization
    policy_config['scale'] = config['scale']
    policy_config['guidance'] = config['guidance']
    policy_config['guided_steps'] = config.get('guided_steps', 0)

    # Create policy (auto-switches to EADP if guidance params are non-zero)
    policy = make_policy(policy_config)
    
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    
    # Show guided diffusion modes
    scale = config.get('scale', 0.0)
    guidance = config.get('guidance', 0.0)
    
    if scale > 0.0:
        print(f"🌟 GUIDED DIFFUSION: Alpha-based scaling throughout diffusion process (scale={scale:.3f})")
        print("   Guidance strength increases as noise decreases (stronger at clean timesteps)")
    if guidance > 0.0:
        guided_steps = config.get('guided_steps', 0)
        print(f"🔧 REFINEMENT GUIDANCE: Post-diffusion iterative refinement (guidance={guidance:.3f}, steps={guided_steps})")
    if scale == 0.0 and guidance == 0.0:
        print("🚫 NO GUIDANCE: Pure diffusion sampling without MPC guidance")
    
    # Show 3D viewer status
    if use_3d_viewer:
        print("📺 3D VIEWER: Enabled")
    
    env = make_ee_sim_env(task_name, camera_names=camera_names, disturbance_enabled=disturbance_enabled, acados_build_dir=acados_build_dir)
    env_max_reward = env.task.max_reward
            
    # Wrap environment with trajectory-aware wrapper for UMI policies to enable MPC cost evaluation
    env = TrajectoryAwareEnvWrapper(env)

    # Capability flags: 4DoF tasks expose ACADOS-based cost functions; UR10e does not
    has_acados_cost = hasattr(env.task, 'compute_trajectory_cost_with_tracking')

    # ✅ Attach environment handle to policy for downstream access
    policy.env = env

    # Handle query frequency for UMI policy
    # UMI runs inference every (action_horizon * obs_down_sample_steps) environment timesteps
    # This accounts for the fact that each predicted action is repeated obs_down_sample_steps times
    query_frequency = action_horizon * obs_down_sample_steps
    
    print(f"UMI query frequency: {query_frequency} (effective_action_horizon={action_horizon} * obs_down_sample_steps={obs_down_sample_steps})")

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = config.get('num_rollouts', 10)

    # Directory for saving rollout artefacts (videos, plots, results)
    resume_mode = config.get('resume', False)
    output_dir = config.get('output_dir', None)
    
    if output_dir is None:
        # Get workspace root (2 levels up from policy_learning)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(script_dir))
        
        # For UAM tasks with disturbance, use uam-disturb_* folder naming
        folder_task_name = task_name
        if disturbance_enabled and task_name.startswith('uam_'):
            folder_task_name = task_name.replace('uam_', 'uam-disturb_', 1)
        
        task_base = os.path.join(workspace_root, 'results', 'eval', folder_task_name)
        
        if resume_mode:
            # Auto-find most recent run
            output_dir = find_latest_timestamped_dir(task_base)
            if output_dir is None:
                print(f"❌ Error: --resume specified but no previous runs found in {task_base}")
                sys.exit(1)
            print(f"📂 Resuming from: {output_dir}")
        else:
            # Create new timestamped directory
            timestamp = get_timestamp()
            output_dir = os.path.join(task_base, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)

    # Initialize experiment summary tracker
    experiment_summary = ExperimentSummary(output_dir)

    # Progress tracking setup
    guidance = config.get('guidance', 0.0)
    guided_steps = config.get('guided_steps', 0)
    experiment_id = f"g{guidance}_s{guided_steps}"
    progress_file = os.path.join(output_dir, 'progress_status.json')
    
    def update_progress_status(current_episode, successful_episodes, avg_episode_duration, status="running"):
        """Update progress status file for monitoring by ablation runner"""
        progress_data = {
            "experiment_id": experiment_id,
            "current_episode": current_episode + total_existing_episodes,
            "total_episodes": num_rollouts,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / current_episode if current_episode > 0 else 0.0,
            "avg_episode_duration": avg_episode_duration,
            "estimated_remaining_time": avg_episode_duration * (episodes_needed - current_episode),
            "status": status,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            # Don't let progress tracking failures break the experiment
            pass

    # Determine next episode index based on existing episodes that are already present on disk.
    # This lets us resume an interrupted run and continue numbering without overwriting.
    if resume_mode:
        existing_episode_dirs = [d for d in os.listdir(output_dir) if d.startswith('episode_')]
        existing_ids = []
        for d in existing_episode_dirs:
            try:
                existing_ids.append(int(d.split('_')[1]))
            except (IndexError, ValueError):
                continue  # Skip malformed names
        
        # Count total existing episodes
        total_existing_episodes = len(existing_ids)
        next_episode_id = max(existing_ids) + 1 if existing_ids else 0
        
        # Check if we already have enough episodes
        if total_existing_episodes >= num_rollouts:
            print(f"✅ Already have {total_existing_episodes} episodes (target: {num_rollouts}). No additional episodes needed.")
            success_rate = 0.0  # Will be calculated from existing episodes if needed
            avg_return = 0.0    # Will be calculated from existing episodes if needed
            return success_rate, avg_return
        
        # Calculate how many more episodes we actually need to reach the target
        episodes_needed = num_rollouts - total_existing_episodes
        print(f"📊 Found {total_existing_episodes} existing episodes, need {episodes_needed} more to reach target of {num_rollouts}")
    else:
        # Fresh start
        total_existing_episodes = 0
        next_episode_id = 0
        episodes_needed = num_rollouts

    # Track how many successfully recorded episodes we have completed in this session.
    completed_episodes = 0
    
    # Track failure statistics
    can_drop_failure_count = 0
    mpc_restart_count = 0
    
    episode_returns = []
    highest_rewards = []
    
    # Track episode timing for progress estimates
    episode_start_times = []
    episode_durations = []
    successful_episode_count = 0

    # setup keyboard listener for manual episode control
    key_flags = {'discard': False, 'save': False, 'quit': False, 'paused': False}
    
    # Set window keywords for focus detection
    if use_3d_viewer:
        window_keywords = ['mujoco', 'simulate']
    elif onscreen_render:
        window_keywords = ['Simulation Preview']
    else:
        window_keywords = []

    def on_press(key):
        if not is_window_focused(window_keywords):
            return
        
        if key == keyboard.Key.esc:
            key_flags['quit'] = True
            return False
        elif key == keyboard.Key.space:
            key_flags['paused'] = not key_flags['paused']
            if key_flags['paused']:
                print("⏸️  PAUSED - Simulation frozen (press SPACE to resume)")
            else:
                print("▶️  RESUMED - Simulation continuing")
        elif hasattr(key, 'char'):
            if key.char == '1':
                print("🔄 Discard episode and reset")
                key_flags['discard'] = True
            elif key.char == '2':
                print("💾 Save episode and reset")
                key_flags['save'] = True
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Initialize progress status
    update_progress_status(0, 0, 0.0, "starting")

    # ------------------------------------------------------------------
    # 3D Viewer setup (similar to record_episodes_keyboard.py)
    # ------------------------------------------------------------------
    viewer = None
    if use_3d_viewer:
        print("Setting up 3D MuJoCo viewer for evaluation...")
        print("3D Viewer Controls:")
        print("  Mouse: Rotate camera")
        print("  Mouse wheel: Zoom")
        print("  Ctrl+Mouse: Pan camera")
        print("  ESC - exit evaluation")
        print("  SPACE - pause/resume simulation")
        print("  1 - discard episode and reset")
        print("  2 - save episode and reset")
        
        # We'll create the viewer after the first env.reset() to have access to physics
        
    while completed_episodes < episodes_needed:
        # Current episode id is based on how many episodes already exist on disk plus the number
        # of completed episodes in this session so far.
        rollout_id = next_episode_id + completed_episodes
        # reset episode control flags
        key_flags['discard'] = False
        key_flags['save'] = False
        
        # Episode failure/restart flags
        episode_failed_can = False
        episode_restart_mpc = False

        # Initialize episode metrics tracker
        episode_tracker = EpisodeMetricsTracker(rollout_id, output_dir)

        # Track episode start time for progress estimates
        episode_start_time = time.time()
        episode_start_times.append(episode_start_time)

        timestep = env.reset()

        if has_acados_cost:
            print(f"🔄 Resetting MPC solvers for episode {rollout_id}")
            env.task.mpc_planner.reset(reset_qp_solver_mem=1)
            env.task.mpc_planner_mini.reset(reset_qp_solver_mem=1)
        
        # Create 3D viewer after first reset if needed
        if use_3d_viewer and viewer is None:
            viewer = mujoco.viewer.launch_passive(
                model=env.physics.model.ptr, 
                data=env.physics.data.ptr
            )

            print("3D Viewer enabled with trajectory visualization:")
            print("  - Colored spheres: Planned trajectory from action buffer")
            print("  - Grey dots: Actual executed trajectory (accumulates over time)")
            print("  - Green: Current position | Yellow: Near future | Red: Far future")
            print("  - Cyan (transparent): Extrapolated trajectory when buffer is short")
        actual_trajectory = []
        actual_trajectory_counter = 0
        policy.reset()

        # OpenCV full-screen preview setup (only if not using 3D viewer)
        window_name = 'Simulation Preview'
        if onscreen_render and not use_3d_viewer:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Start in fullscreen mode
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            fullscreen_mode = True  # track current fullscreen state
            # Get screen resolution
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            # show first frame
            first_frame = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            first_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            first_bgr = cv2.resize(first_bgr, (screen_width, screen_height))
            cv2.imshow(window_name, first_bgr)
        
        target_img_size = (camera_shape[1], camera_shape[2])
        obs_buffer_manager = ObservationBufferManager(img_obs_horizon, low_dim_obs_horizon)
        action_buffer = []
        last_inference_step = -query_frequency
        
        initial_obs = timestep.observation
        _, initial_robot_pos, initial_robot_quat_wxyz, _ = extract_robot_state(initial_obs)
        initial_robot_rot_axis_angle = quat_wxyz_to_axis_angle(initial_robot_quat_wxyz)
        episode_start_pose = [np.concatenate([initial_robot_pos, initial_robot_rot_axis_angle])]

        image_list = []
        qpos_list = []
        target_qpos_list = []
        rewards = []
        all_costs = []
        actions_consumed = 0
        current_pregrad_trajectory = None
        pregrad_full_buffer = None
        pending_diffusion_analysis = []
        all_actual_ee_positions = []
        all_actual_ee_quaternions = []
        success_detected = False
        success_counter = 0
        exit_program = False
        episode_crashed = False
        steps_until_stop_success = STEPS_UNTIL_STOP_ON_SUCCESS
        with torch.inference_mode():
            for step in range(max_timesteps):
                if key_flags['quit']:
                    print("Exiting program per user request")
                    listener.stop()
                    sys.exit(0)
                
                if should_exit_episode(key_flags):
                    print(f"User requested {'discard' if key_flags['discard'] else 'save'} and reset")
                    break
                
                if key_flags['paused']:
                    if handle_pause_mode(key_flags, use_3d_viewer, viewer):
                        if key_flags['quit']:
                            listener.stop()
                            sys.exit(0)
                        break
                    
                if use_3d_viewer or onscreen_render:
                    time.sleep(DT)
                
                if use_3d_viewer:
                    render_3d_viewer(viewer, success_detected, success_counter, 
                                   rollout_id, step, max_timesteps)
                elif onscreen_render:
                    exit_program, fullscreen_mode = render_opencv_preview(
                        env, onscreen_cam, screen_width, screen_height, window_name,
                        success_detected, success_counter, rollout_id, step, max_timesteps,
                        fullscreen_mode
                    )
                    if exit_program:
                        break
                else:
                    if step % 100 == 0:
                        print(f"step: {step}")
                obs = timestep.observation
                if 'images' in obs:
                    ee_img = obs['images'].get('ee') if isinstance(obs['images'], dict) else None
                    image_list.append({'ee': ee_img} if ee_img is not None else obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                
                qpos_8d, robot_pos, robot_quat_wxyz, robot_gripper = extract_robot_state(obs)
                current_actual_pos = robot_pos.copy()
                current_actual_quat = robot_quat_wxyz.copy()
                robot_rot_axis_angle = quat_wxyz_to_axis_angle(robot_quat_wxyz)
                
                curr_image = get_image(timestep, camera_names, target_img_size)

                should_update_obs = (step % obs_down_sample_steps == 0)
                
                if should_update_obs:
                    img_hwc = prepare_umi_image_observation(curr_image)
                    obs_buffer_manager.add_observation(img_hwc, robot_pos, robot_rot_axis_angle, robot_gripper)
                    env_obs_stacked = obs_buffer_manager.get_stacked_observation()
                    
                    obs_dict_np = get_real_umi_obs_dict(
                        env_obs=env_obs_stacked,
                        shape_meta=shape_meta,
                        obs_pose_repr=obs_pose_repr,
                        tx_robot1_robot0=None,
                        episode_start_pose=episode_start_pose
                    )
                    
                    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).cuda())
                    policy.original_env_obs_stacked = env_obs_stacked
                
                should_infer = (step % query_frequency == 0)
                if should_infer:
                    last_inference_step = step
                    
                    action_buffer, reference_trajectory, mpc_cost = run_policy_inference(
                        policy, obs_dict, env_obs_stacked, action_pose_repr, query_frequency,
                        env, log_diffusion, has_acados_cost
                    )
                    
                    # Check MPC cost for restart condition (early in episode only)
                    if has_acados_cost and mpc_cost > MPC_COST_THRESHOLD:
                        if step < MPC_RESTART_WINDOW:
                            print(f"🔄 MPC cost {mpc_cost:.3f} > {MPC_COST_THRESHOLD} at step {step} - restarting episode")
                            episode_restart_mpc = True
                            break
                        else:
                            # Later in episode: just log warning, don't fail (let episode complete)
                            print(f"⚠️ MPC cost {mpc_cost:.3f} > {MPC_COST_THRESHOLD} at step {step} (past restart window)")
                    
                    current_pregrad_trajectory, pregrad_full_buffer = update_3d_viewer_trajectory(
                        policy, use_3d_viewer, viewer, robot_pos, query_frequency, 
                        actions_consumed, pregrad_full_buffer
                    )
                    
                    actions_consumed = 0
                    current_actual_trajectory_start = len(all_actual_ee_positions)
                    if log_diffusion:
                        pending_diffusion_analysis.append((step, current_actual_trajectory_start))

                if len(action_buffer) > 0:
                    remaining_trajectory = convert_action_buffer_to_trajectory(action_buffer)
                    if remaining_trajectory is not None:
                        env.set_trajectory(remaining_trajectory)
                    
                    raw_action = action_buffer.pop(0)
                    actions_consumed += 1
                    
                    if len(action_buffer) < query_frequency:
                        action_buffer = extrapolate_action_buffer(action_buffer, query_frequency)
                else:
                    raise RuntimeError(
                        f"Action buffer is empty at step {step}! "
                        f"Last inference at step {last_inference_step}, query_frequency={query_frequency}. "
                        f"This indicates a bug in action chunking logic."
                    )
                
                target_qpos = raw_action

                try:
                    timestep = env.step(target_qpos)
                    episode_tracker.num_timesteps += 1
                except Exception as e:
                    print(f"\n🚨 SIMULATION CRASH at rollout {rollout_id}, step {step}: {e}")
                    print(f"Full traceback:")
                    traceback.print_exc()
                    print(f"target_qpos shape: {target_qpos.shape if hasattr(target_qpos, 'shape') else type(target_qpos)}")
                    print(f"target_qpos: {target_qpos}")
                    episode_crashed = True
                    break

                if timestep.reward == -1:
                    print(f"❌ Episode failed")
                    episode_failed_can = True
                    break

                if step > 0 and current_actual_pos is not None:
                    target_pos = target_qpos[:3]
                    target_quat = target_qpos[3:7] if len(target_qpos) >= 7 else np.array([1, 0, 0, 0])
                    
                    episode_tracker.add_tracking_error(
                        timestep=step-1,
                        target_pos=target_pos,
                        actual_pos=current_actual_pos,
                        target_quat=target_quat,
                        actual_quat=current_actual_quat
                    )

                actual_ee_pos = get_ee_pos(env._physics)
                all_actual_ee_positions.append(robot_pos.copy())
                all_actual_ee_quaternions.append(robot_quat_wxyz.copy())
                
                if use_3d_viewer and viewer is not None:
                    actual_trajectory_counter += 1
                    if actual_trajectory_counter % ACTUAL_TRAJECTORY_SAMPLE_RATE == 0:
                        actual_trajectory.append(actual_ee_pos.copy())
                        if len(actual_trajectory) > GREY_DOT_HISTORY_SIZE:
                            actual_trajectory.pop(0)
                    
                    # Update trajectory visualization with both planned and actual trajectories
                    if len(action_buffer) > 0:
                        # Get current end effector tip position for planned trajectory
                        current_ee_pos = get_ee_pos(env._physics)

                        # The original buffer size minus actions consumed gives us where the original buffer ends in current indexing
                        effective_original_size = query_frequency - actions_consumed
                        trajectory_points = extract_trajectory_positions(action_buffer, current_ee_pos)
                        
                        # Compute remaining pregrad trajectory for fading visualization
                        if pregrad_full_buffer is not None:
                            remaining_pregrad_actions = pregrad_full_buffer[actions_consumed:]
                            if remaining_pregrad_actions:
                                current_pregrad_trajectory = extract_trajectory_positions(remaining_pregrad_actions, current_ee_pos)
                            else:
                                current_pregrad_trajectory = None
                        else:
                            current_pregrad_trajectory = None

                        viewer.user_scn.ngeom = 0
                        add_trajectory_to_viewer(
                            viewer,
                            trajectory_points,
                            actual_trajectory,
                            effective_original_size,
                            None,  # no candidate trajectories since NUM_CANDIDATES=1
                            current_pregrad_trajectory,
                            None,  # vanilla trajectories removed
                            None   # guided trajectories removed
                        )

                        # Append disturbance arrows after trajectory geoms (if enabled and available)
                        if getattr(env.task, 'disturbance_enabled', False) and \
                           hasattr(env.task, '_wind') and hasattr(env.task, '_torque'):
                            add_disturbance_arrows(
                                viewer,
                                env._physics,
                                env.task._wind,
                                env.task._torque
                            )

                if log_diffusion:
                    process_completed_diffusion_analysis(
                        pending_diffusion_analysis, all_actual_ee_positions, all_actual_ee_quaternions,
                        query_frequency, policy, episode_tracker
                    )

                if not success_detected and timestep.reward >= env_max_reward:
                    success_detected = True
                    success_counter = 0
                if success_detected:
                    success_counter += 1
                    if success_counter >= steps_until_stop_success:
                        break

                qpos_list.append(qpos_8d)
                target_qpos_list.append(target_qpos)
                rewards.append(timestep.reward)
                episode_tracker.add_qpos_data(target_qpos, qpos_8d)

            if exit_program:
                break

        if key_flags['discard']:
            print("🔄 Episode discarded by user (will retry same episode index)")
            handle_episode_crash(episode_tracker)
            continue
        
        # Handle MPC restart condition (early high cost - retry same episode)
        if episode_restart_mpc:
            print("🔄 Episode restarted due to high MPC cost (will retry same episode index)")
            mpc_restart_count += 1
            handle_episode_crash(episode_tracker)
            continue
        
        if episode_failed_can:
            print("❌ Episode failed due to can dropping below threshold")
            can_drop_failure_count += 1

        if log_diffusion:
            finalize_remaining_analysis(
                pending_diffusion_analysis,
                all_actual_ee_positions, all_actual_ee_quaternions,
                query_frequency, policy, episode_tracker
            )

        failure_reasons = []
        if episode_failed_can:
            failure_reasons.append("can_dropped_below_threshold")

        # Compute success from local rewards list FIRST (single source of truth)
        if len(rewards) == 0:
            episode_return = 0.0
            episode_highest_reward = -1.0
        else:
            rewards_np = np.asarray(rewards, dtype=float)
            episode_return = float(np.sum(rewards_np))
            episode_highest_reward = float(np.max(rewards_np))
        
        # Determine success once, use everywhere for consistency
        episode_success = (episode_highest_reward >= env_max_reward)
        
        # Pass all reward-based metrics from main loop to ensure consistency
        episode_tracker.finalize_episode(
            env_max_reward, 
            crashed=episode_crashed, 
            failure_reasons=failure_reasons, 
            success=episode_success,
            episode_return=episode_return,
            highest_reward=episode_highest_reward
        )
        episode_metrics = episode_tracker.save_metrics()
        experiment_summary.add_episode(episode_metrics)

        if save_episode:
            video_path = os.path.join(episode_tracker.episode_dir, 'video.mp4')
            save_videos(image_list, DT, video_path=video_path)

        episode_returns.append(episode_return)
        highest_rewards.append(episode_highest_reward)
        
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_durations.append(episode_duration)
        
        if episode_success:
            successful_episode_count += 1
        
        if episode_crashed:
            print("🔄 Episode crashed – retrying the same index")
            handle_episode_crash(episode_tracker)
            continue

        completed_episodes += 1
        avg_episode_duration = np.mean(episode_durations) if episode_durations else 0.0
        update_progress_status(completed_episodes, successful_episode_count, avg_episode_duration)

    if use_3d_viewer and viewer:
        viewer.close()
        time.sleep(VIEWER_CLEANUP_DELAY)
    elif onscreen_render:
        cv2.destroyAllWindows()

    experiment_summary.save_summary()
    update_progress_status(completed_episodes, successful_episode_count, 
                          np.mean(episode_durations) if episode_durations else 0.0, 
                          status="completed")

    success_rate = np.mean(np.array(highest_rewards) >= env_max_reward)
    avg_return = np.mean(episode_returns)
    total_episodes_this_session = len(episode_returns)
    total_episodes = total_existing_episodes + total_episodes_this_session

    crashed_episodes = sum(1 for ep in experiment_summary.episode_metrics if ep.get('crashed', False))
    crash_rate = crashed_episodes / total_episodes_this_session if total_episodes_this_session > 0 else 0.0
    
    summary_str = f'\n{"="*60}\n'
    summary_str += f'🎯 FINAL EXPERIMENT RESULTS\n'
    summary_str += f'{"="*60}\n'
    if total_existing_episodes > 0:
        summary_str += f'📁 Existing episodes: {total_existing_episodes}\n'
        summary_str += f'🆕 New episodes (this session): {total_episodes_this_session}\n'
        summary_str += f'📊 Total episodes: {total_episodes}\n'
        summary_str += f'{"="*60}\n'
        summary_str += f'📈 THIS SESSION RESULTS:\n'
    summary_str += f'Success rate: {success_rate:.1%} ({int(success_rate * total_episodes_this_session)}/{total_episodes_this_session})\n'
    summary_str += f'Average return: {avg_return:.3f}\n'
    if crashed_episodes > 0: summary_str += f'🚨 Crashed episodes: {crashed_episodes}/{total_episodes_this_session} ({crash_rate:.1%})\n'
    if mpc_restart_count > 0: summary_str += f'🔄 MPC restarts: {mpc_restart_count}\n'
    if can_drop_failure_count > 0: summary_str += f'❌ Can drop failures: {can_drop_failure_count}/{total_episodes_this_session}\n'
    summary_str += f'{"="*60}\n'
    
    for r in range(int(env_max_reward) + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / total_episodes_this_session if total_episodes_this_session > 0 else 0.0
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{total_episodes_this_session} = {more_or_equal_r_rate*100:.1f}%\n'

    print(summary_str)
    return success_rate, avg_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--use_3d_viewer', action='store_true', help='Use 3D MuJoCo viewer instead of OpenCV preview')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='Checkpoint directory (optional - will auto-detect from task_name if not provided)', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--load_ckpt_file_path', action='store', type=str, help='load_ckpt_file_path', required=False)
    parser.add_argument('--num_rollouts', action='store', type=int, default=10, help='Number of evaluation rollouts to execute')
    parser.add_argument('--output_dir', action='store', type=str, help='Directory to save rollout artefacts', required=False)
    parser.add_argument('--disturb', action='store_true', help='Enable disturbances in the simulation environment')
    parser.add_argument('--guidance', action='store', type=float, default=0.0, help='Guidance factor for MPC')
    parser.add_argument('--guided_steps', action='store', type=int, default=0, help='Number of diffusion timesteps during which guidance is applied')
    parser.add_argument('--log_diffusion', action='store_true', help='Enable logging & plotting of diffusion trajectories/costs (disabled by default)')
    parser.add_argument('--acados_build_dir', default=None, type=str, help='Unique directory for ACADOS build files')
    parser.add_argument('--scale', action='store', type=float, default=0.0, help='Alpha-based scaling factor for guided diffusion throughout the process (0=disabled)')
    parser.add_argument('--resume', action='store_true', help='Resume from most recent run (auto-finds latest timestamp)')
    args = vars(parser.parse_args())
    
    # Auto-detect checkpoint directory if not provided
    if args['ckpt_dir'] is None:
        args['ckpt_dir'] = auto_detect_checkpoint_dir(args['task_name'])
        if args['ckpt_dir'] is None:
            print(f"❌ Error: Could not auto-detect checkpoint directory for task '{args['task_name']}'")
            print(f"   Please provide --ckpt_dir explicitly or ensure checkpoints/umi_{{task}}/ exists")
            sys.exit(1)
    
    args['load_ckpt_file_path'] = resolve_checkpoint_path(
        args['ckpt_dir'], 
        args['load_ckpt_file_path']
    )
    
    main(args)
