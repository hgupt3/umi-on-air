"""ARKit-based episode recorder using MujocoAR.

This script records teleoperation episodes using an iPhone running the MujocoAR app.
The iPhone's AR data controls the robot end-effector, and episodes are saved to HDF5 files.

Usage:
    python record_episodes_arkit.py --task_name umi_cabinet --onscreen_render
    python record_episodes_arkit.py --task_name umi_cabinet --use_3d_viewer

Controls (via MujocoAR app on iPhone):
    - Phone motion: Controls end-effector pose
    - Button press: Close gripper
    - Button release: Open gripper
    - Toggle: Start/stop recording

Requires: pip install mujoco_ar
"""

import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from constants import DT, XML_DIR, START_UAM_POSE, SIM_TASK_CONFIGS

import cv2
import tkinter as tk
import mujoco.viewer

from ee_sim_env import make_ee_sim_env

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

# Import ARKit policy
from arkit_policy import ARKitPolicy


def main(args):
    """ARKit tele-operation episode recorder with OpenCV preview or 3D viewer.

    Features:
    - Uses MujocoAR to receive pose data from iPhone ARKit
    - Supports both OpenCV window (fullscreen) and 3D MuJoCo viewer
    - Recording controlled via toggle on iPhone app
    - Haptic feedback on success detection
    - Automatic early stop after success held for 3 seconds
    - Episodes written atomically to prevent corruption
    """
    # ------------------------------------------------------------------
    # Config & Environment init
    # ------------------------------------------------------------------
    task_name = args['task_name']
    onscreen_render = args['onscreen_render']
    use_3d_viewer = args['use_3d_viewer']
    arkit_port = args.get('port', 8888)
    arkit_scale = args.get('scale', 1.0)

    task_cfg = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_cfg['dataset_dir']
    episode_len = task_cfg['episode_len']
    camera_names = task_cfg['camera_names']
    render_cam_name = 'ee'

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    disturbance = args.get('disturb', False)
    env = make_ee_sim_env(
        task_name,
        camera_names=camera_names,
        disturbance_enabled=disturbance
    )

    # Initialize ARKit policy
    policy = ARKitPolicy(
        port=arkit_port,
        scale=arkit_scale,
        task_name=task_name
    )

    # 3-second grace period after success signal before stopping
    steps_until_stop_success = int(3.0 / DT)

    # ------------------------------------------------------------------
    # Viewer setup - either OpenCV or 3D MuJoCo viewer
    # ------------------------------------------------------------------
    viewer = None
    screen_w, screen_h = 1920, 1080  # Default values

    if use_3d_viewer:
        print("Setting up 3D MuJoCo viewer...")
        print("3D Viewer Controls:")
        print("  Mouse: Rotate camera")
        print("  Mouse wheel: Zoom")
        print("  Ctrl+Mouse: Pan camera")
        print("  iPhone ARKit controls the robot end-effector")
        print("  Toggle on iPhone: Start/stop recording")

        viewer = mujoco.viewer.launch_passive(
            model=env.physics.model.ptr,
            data=env.physics.data.ptr
        )

        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

        mujoco.mjv_defaultFreeCamera(env.physics.model.ptr, viewer.cam)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -30.0
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 1.0

    elif onscreen_render:
        window_name = 'ARKit Teleoperation'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        _root = tk.Tk()
        screen_w, screen_h = _root.winfo_screenwidth(), _root.winfo_screenheight()
        _root.destroy()

    # ------------------------------------------------------------------
    # Helper - draw controls legend on bottom right (like keyboard version)
    # ------------------------------------------------------------------
    def _draw_controls_legend(img, screen_w, screen_h, is_recording=False, toggle_on=False):
        """Draw ARKit controls legend on bottom right corner."""
        legend_x = screen_w - 400
        legend_y_start = screen_h - 300
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (screen_w - 420, screen_h - 340), (screen_w - 10, screen_h - 10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Title
        cv2.putText(img, "ARKit CONTROLS", (legend_x, legend_y_start - 10),
                   font, 0.8, (255, 255, 255), thickness + 1)
        
        # Colors
        movement_color = (100, 200, 255)
        gripper_color = (100, 180, 255)
        record_color = (255, 150, 255)
        
        # Toggle status indicator
        toggle_status = "ON" if toggle_on else "OFF"
        toggle_color = (0, 255, 0) if toggle_on else (100, 100, 100)
        
        controls = [
            ("Phone", "Move EE position", movement_color),
            ("Rotate", "Rotate EE orientation", movement_color),
            ("", "", (0, 0, 0)),  # Spacer
            ("Button", "Close gripper (hold)", gripper_color),
            ("Release", "Open gripper", gripper_color),
            ("", "", (0, 0, 0)),  # Spacer
            ("Toggle", f"Record [{toggle_status}]", record_color),
            ("ESC", "Exit program", (200, 200, 200)),
        ]
        
        y = legend_y_start + 25
        for key, action, color in controls:
            if key == "":
                y += line_height // 2
                continue
            cv2.putText(img, key, (legend_x, y), font, font_scale, color, thickness)
            cv2.putText(img, action, (legend_x + 90, y), font, font_scale - 0.05, 
                       (220, 220, 220), thickness - 1)
            y += line_height
        
        return img

    # ------------------------------------------------------------------
    # Helper - draw status overlay on screen
    # ------------------------------------------------------------------
    def _draw_status_overlay(img, screen_w, screen_h, is_recording, episode_idx,
                              step=0, episode_len=0, success_detected=False,
                              remaining_time=0, connected=False, toggle_on=False,
                              wait_for_toggle_off=False):
        """Draw status information on the preview image."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Connection status
        conn_color = (0, 255, 0) if connected else (0, 0, 255)
        conn_text = "iPhone: CONNECTED" if connected else "iPhone: WAITING..."
        cv2.putText(img, conn_text, (50, 50), font, 1.2, conn_color, 3)

        # Episode number
        cv2.putText(img, f"Episode {episode_idx}", (50, screen_h - 60),
                    font, 2.0, (255, 255, 0), 4)

        if wait_for_toggle_off:
            cv2.putText(img, "SUCCESS! Toggle OFF to continue", (50, 100), 
                        font, 2.0, (0, 255, 0), 4)
        elif is_recording:
            if success_detected:
                overlay = f"SUCCESS - stopping in {remaining_time:.1f}s"
                color = (0, 255, 0)
            else:
                overlay = f"RECORDING: {step + 1}/{episode_len}"
                color = (0, 0, 255)
            cv2.putText(img, overlay, (50, 100), font, 2.0, color, 4)
        else:
            cv2.putText(img, "READY - Toggle ON to start",
                        (50, 100), font, 1.5, (0, 255, 255), 3)

        # Draw controls legend
        _draw_controls_legend(img, screen_w, screen_h, is_recording, toggle_on)

        return img

    # ------------------------------------------------------------------
    # Helper - atomic episode writer
    # ------------------------------------------------------------------
    def _save_episode(idx, episode, action_traj):
        """Write episode to <dataset_dir>/episode_<idx>.hdf5 atomically."""
        if len(action_traj) == 0:
            print("[WARN] No timesteps - nothing to save.")
            return

        max_T = len(action_traj)

        qpos_arr = np.stack([episode[t].observation['qpos'] for t in range(max_T)])
        act_arr = np.stack(action_traj)

        img_arrays = {
            cam: np.stack([episode[t].observation['images'][cam] for t in range(max_T)])
            for cam in camera_names
        }

        tmp_path = os.path.join(dataset_dir, f'episode_{idx}.tmp')
        final_path = os.path.join(dataset_dir, f'episode_{idx}.hdf5')

        t0 = time.time()
        with h5py.File(tmp_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['success'] = True

            obs_grp = root.create_group('observations')
            obs_grp.create_dataset('qpos', data=qpos_arr)
            root.create_dataset('action', data=act_arr)

            img_grp = obs_grp.create_group('images')
            for cam, arr in img_arrays.items():
                img_grp.create_dataset(cam, data=arr, dtype='uint8', chunks=(1,) + arr.shape[1:])

        os.replace(tmp_path, final_path)
        print(f"Saved to {final_path} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Main loop - record successive episodes until user quits
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ARKit Episode Recorder")
    print("=" * 60)
    print("Waiting for iPhone connection...")
    print("Use Toggle on iPhone to start/stop recording")
    print("Press ESC in the viewer window to exit")
    print("=" * 60 + "\n")

    while True:
        print(f"\n{'=' * 40}")
        print(f"Ready to record episode_{episode_idx}.hdf5")
        print(f"{'=' * 40}")

        exit_program = False
        restart_episode = False

        # Reset environment
        ts = env.reset()

        # Set policy reference pose from environment's initial state
        init_qpos = ts.observation['qpos']
        policy.set_reference_pose(
            ee_pos=init_qpos[0:3],
            ee_quat=init_qpos[3:7]
        )
        policy.gripper_status = init_qpos[7]
        policy.recording = False  # Reset recording state

        episode = [ts]
        action_traj = []

        # ------------------------------------------------------------------
        # Preview/wait loop - wait for recording to start via toggle
        # ------------------------------------------------------------------
        print("Preview mode - use Toggle on iPhone to start recording")

        recording_started = False
        prev_recording_state = False

        while not recording_started:
            preview_start_time = time.time()

            # Get action from ARKit
            action_preview = policy.get_action()
            ts_preview = env.step(action_preview)

            # Check if recording was toggled on
            if policy.is_recording() and not prev_recording_state:
                # Recording just started
                recording_started = True
                print("Recording started!")
                # Note: Not resetting calibration so pose stays continuous
                episode = [ts_preview]
                break

            prev_recording_state = policy.is_recording()

            # Check for connection status
            connected = policy.is_connected()

            if use_3d_viewer:
                viewer.sync()

                # Print status periodically
                if int(time.time()) % 2 == 0:
                    status = "CONNECTED" if connected else "WAITING"
                    print(f"\r[{status}] Waiting for toggle to start recording...", end="")

            elif onscreen_render:
                img_bgr = cv2.cvtColor(
                    ts_preview.observation['images'][render_cam_name],
                    cv2.COLOR_RGB2BGR
                )
                img_res = cv2.resize(img_bgr, (screen_w, screen_h))

                _draw_status_overlay(
                    img_res, screen_w, screen_h,
                    is_recording=False,
                    episode_idx=episode_idx,
                    connected=connected,
                    toggle_on=policy.is_recording()
                )

                cv2.imshow(window_name, img_res)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    exit_program = True
                    break

            # Maintain timing
            elapsed = time.time() - preview_start_time
            if elapsed < DT:
                time.sleep(DT - elapsed)

        if exit_program:
            break

        # ------------------------------------------------------------------
        # Recording loop
        # ------------------------------------------------------------------
        step = 0
        success_detected = False
        success_counter = 0
        haptic_triggered = False

        print("Recording in progress...")

        while True:
            step_start_time = time.time()

            action = policy.get_action()
            action_traj.append(action)
            ts = env.step(action)
            episode.append(ts)

            # Check if recording was toggled off
            if not policy.is_recording():
                print("Recording stopped via toggle")
                restart_episode = True
                break

            if use_3d_viewer:
                viewer.sync()

                if step % 50 == 0:
                    if success_detected:
                        remaining = max(0, (steps_until_stop_success - success_counter) * DT)
                        print(f"SUCCESS - stopping in {remaining:.1f}s")
                    else:
                        print(f"Recording EP{episode_idx}: {step + 1}/{episode_len}")

            elif onscreen_render:
                img_bgr = cv2.cvtColor(
                    ts.observation['images'][render_cam_name],
                    cv2.COLOR_RGB2BGR
                )
                img_res = cv2.resize(img_bgr, (screen_w, screen_h))

                remaining = max(0, (steps_until_stop_success - success_counter) * DT) if success_detected else 0

                _draw_status_overlay(
                    img_res, screen_w, screen_h,
                    is_recording=True,
                    episode_idx=episode_idx,
                    step=step,
                    episode_len=episode_len,
                    success_detected=success_detected,
                    remaining_time=remaining,
                    connected=True,
                    toggle_on=policy.is_recording()
                )

                cv2.imshow(window_name, img_res)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    exit_program = True
                    break

            # Success detection
            if not success_detected and ts.reward == env.task.max_reward:
                success_detected = True
                success_counter = 0
                print("Success reached - finishing in 3s")

                # Trigger haptic feedback on success
                if not haptic_triggered:
                    policy.vibrate(sharpness=1.0, intensity=1.0, duration=0.1)
                    haptic_triggered = True

            if success_detected:
                success_counter += 1
                if success_counter >= steps_until_stop_success:
                    print("Success countdown elapsed - stopping recording")
                    break

            step += 1
            if step >= episode_len:
                break

            # Maintain timing
            elapsed = time.time() - step_start_time
            if elapsed < DT:
                time.sleep(DT - elapsed)

        if exit_program:
            break

        # ------------------------------------------------------------------
        # Decide what to do with the recorded episode
        # ------------------------------------------------------------------
        success_episode = success_detected and success_counter >= steps_until_stop_success

        if restart_episode or not success_episode:
            if restart_episode:
                print("Episode restarted - ready for new attempt.")
            else:
                print("Episode did not reach success - discarding.")
            continue

        # Save & advance index
        _save_episode(episode_idx, episode[:-1], action_traj)
        episode_idx += 1

        # ------------------------------------------------------------------
        # Wait for user to toggle OFF before continuing to next episode
        # ------------------------------------------------------------------
        if policy.is_recording():
            print("Toggle OFF to continue to next episode...")
            while policy.is_recording():
                # Keep simulation running and showing status
                action = policy.get_action()
                ts = env.step(action)
                
                if use_3d_viewer:
                    viewer.sync()
                elif onscreen_render:
                    img_bgr = cv2.cvtColor(
                        ts.observation['images'][render_cam_name],
                        cv2.COLOR_RGB2BGR
                    )
                    img_res = cv2.resize(img_bgr, (screen_w, screen_h))
                    _draw_status_overlay(
                        img_res, screen_w, screen_h,
                        is_recording=False,
                        episode_idx=episode_idx,
                        connected=True,
                        toggle_on=True,
                        wait_for_toggle_off=True
                    )
                    cv2.imshow(window_name, img_res)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        exit_program = True
                        break
                
                time.sleep(DT)
            
            if exit_program:
                break
            print("Toggle OFF detected - ready for next episode")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    print("\nShutting down...")
    policy.cleanup()
    if use_3d_viewer and viewer:
        viewer.close()
    elif onscreen_render:
        cv2.destroyAllWindows()


def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    """Get the next available episode index."""
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARKit-based episode recorder using MujocoAR')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Task name (e.g., umi_cabinet, umi_peg)')
    parser.add_argument('--episode_idx', type=int, default=None,
                        help='Starting episode index (auto-detected if not specified)')
    parser.add_argument('--onscreen_render', action='store_true',
                        help='Enable OpenCV preview window')
    parser.add_argument('--use_3d_viewer', action='store_true',
                        help='Use 3D MuJoCo viewer instead of OpenCV preview')
    parser.add_argument('--disturb', action='store_true',
                        help='Enable disturbance in simulation')
    parser.add_argument('--port', type=int, default=8888,
                        help='WebSocket port for MujocoAR connection (default: 8888)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Position scaling factor for ARKit input (default: 1.0)')

    main(vars(parser.parse_args()))

