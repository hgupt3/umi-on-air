import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from constants import DT, XML_DIR, START_UAM_POSE, SIM_TASK_CONFIGS

import cv2
import tkinter as tk  # NEW: for fullscreen sizing
import mujoco.viewer  # NEW: for 3D viewer support

from ee_sim_env import make_ee_sim_env

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

# Import from our new policy file
from keyboard_policy import MinecraftStyleKeyboardPolicy
import matplotlib.pyplot as plt

# Disable matplotlib key bindings that might interfere with our keyboard controls
plt.rcParams['keymap.save'] = []  # Remove 's' key binding for saving
plt.rcParams['keymap.quit'] = []  # Remove 'q' key binding for quitting
plt.rcParams['keymap.grid'] = []  # Remove 'g' key binding for grid
plt.rcParams['keymap.yscale'] = []  # Remove 'l' key binding for log scale

WAITING = 0
TAKEOFF = 1
PAUSE = 2
LAND = 3
FREEFLIGHT = 4


DEBUG = True

def main(args):
    """Keyboard tele-operation episode recorder with OpenCV preview or 3D viewer and READY screen.

    Key improvements vs. the previous implementation:
    •   Uses either OpenCV window (fullscreen by default) or 3D MuJoCo viewer for preview.
    •   READY screen – waits for P, shows countdown before starting.
    •   Hot-keys while READY:  P-start  |  ESC-quit  |  X-delete last  |  F-toggle fullscreen (OpenCV only).
    •   Recording loop supports ESC (quit), R (restart/abort current), and automatic early stop
        once max reward is reached for 3 s (steps_until_stop_success).
    •   Episodes are written atomically via a temporary "tmp" file to avoid corruption.
    """
    # ------------------------------------------------------------------
    # Config & Environment init
    # ------------------------------------------------------------------
    task_name        = args['task_name']
    onscreen_render  = args['onscreen_render']
    use_3d_viewer    = args['use_3d_viewer']  # NEW: 3D viewer flag

    task_cfg         = SIM_TASK_CONFIGS[task_name]
    dataset_dir      = task_cfg['dataset_dir']
    episode_len      = task_cfg['episode_len']
    camera_names     = task_cfg['camera_names']
    render_cam_name  = 'ee'

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    disturbance = args.get('disturb', False)
    env    = make_ee_sim_env(task_name,
                            camera_names=camera_names,
                            disturbance_enabled=disturbance)
    
    # Set window keywords for focus detection
    if use_3d_viewer:
        window_keywords = ['mujoco', 'simulate']
    elif onscreen_render:
        window_keywords = ['Simulation Preview']
    else:
        window_keywords = []
    
    policy = MinecraftStyleKeyboardPolicy(task_name=task_name, window_keywords=window_keywords)

    # 3-second grace period after success signal before stopping.
    steps_until_stop_success = int(3.0 / DT)

    # ------------------------------------------------------------------
    # Viewer setup - either OpenCV or 3D MuJoCo viewer
    # ------------------------------------------------------------------
    viewer = None
    if use_3d_viewer:
        # Set up 3D MuJoCo viewer
        print("Setting up 3D MuJoCo viewer...")
        print("3D Viewer Controls:")
        print("  Mouse: Rotate camera")
        print("  Mouse wheel: Zoom")
        print("  Ctrl+Mouse: Pan camera")
        print("  Keyboard controls for robot motion still active via MinecraftStyleKeyboardPolicy")
        print("  P - start recording | ESC - exit | R - restart episode | X - delete last")
        
        # Create the 3D viewer
        viewer = mujoco.viewer.launch_passive(
            model=env.physics.model.ptr, 
            data=env.physics.data.ptr
        )
        
        # Configure viewer options
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        
        # Set up camera position for good view of the scene
        mujoco.mjv_defaultFreeCamera(env.physics.model.ptr, viewer.cam)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -30.0
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 1.0
        
    elif onscreen_render:
        # Original OpenCV setup
        window_name = 'Simulation Preview'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Query screen resolution once for resizing.
        _root = tk.Tk(); screen_w, screen_h = _root.winfo_screenwidth(), _root.winfo_screenheight(); _root.destroy()

    # ------------------------------------------------------------------
    # Helper – draw control legend on bottom right
    # ------------------------------------------------------------------
    def _draw_controls_legend(img, screen_w, screen_h, is_recording=False):
        """Draw keyboard controls legend on bottom right corner."""
        # Configuration
        legend_x = screen_w - 480  # Right margin (wider for more content)
        legend_y_start = screen_h - 450  # Bottom margin (taller for more content)
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (screen_w - 510, screen_h - 500), (screen_w - 10, screen_h - 10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Title
        cv2.putText(img, "CONTROLS", (legend_x, legend_y_start - 10),
                   font, 0.9, (255, 255, 255), thickness + 1)
        
        # Control lines organized by category
        # Movement controls (blue/cyan)
        movement_color = (100, 200, 255)
        # Gripper controls (orange)
        gripper_color = (100, 180, 255)
        # Rotation controls (green)
        rotation_color = (150, 255, 150)
        # Speed controls (yellow)
        speed_color = (100, 255, 255)
        # Recording controls (magenta/pink)
        record_color = (255, 150, 255)
        
        controls = [
            ("W/A/S/D", "Move horizontal", movement_color),
            ("Space", "Move up", movement_color),
            ("Shift", "Move down", movement_color),
            ("", "", (0, 0, 0)),  # Spacer
            ("Q", "Close gripper", gripper_color),
            ("E", "Open gripper", gripper_color),
            ("", "", (0, 0, 0)),  # Spacer
            ("Arrows", "Rotate pitch/yaw", rotation_color),
            ("Z/C", "Roll left/right", rotation_color),
            ("", "", (0, 0, 0)),  # Spacer
            ("1-3", "Adjust speed", speed_color),
            ("", "", (0, 0, 0)),  # Spacer
            ("P", "Start recording" if not is_recording else "Recording...", record_color),
            ("R", "Reset scene" if not is_recording else "Restart/abort", record_color),
            ("X", "Delete last", record_color),
            ("ESC", "Exit program", (200, 200, 200)),
        ]
        
        y = legend_y_start + 25
        for key, action, color in controls:
            if key == "":  # Spacer
                y += line_height // 2
                continue
            # Key name (bold)
            cv2.putText(img, key, (legend_x, y), font, font_scale, color, thickness)
            # Action description
            cv2.putText(img, action, (legend_x + 110, y), font, font_scale - 0.05, 
                       (220, 220, 220), thickness - 1)
            y += line_height
        
        return img
    
    # ------------------------------------------------------------------
    # Helper – atomic episode writer
    # ------------------------------------------------------------------
    def _save_episode(idx, episode, action_traj):
        """Write episode to <dataset_dir>/episode_<idx>.hdf5 atomically."""
        if len(action_traj) == 0:
            print("[WARN] No timesteps – nothing to save.")
            return

        max_T = len(action_traj)

        # Build arrays first
        qpos_arr = np.stack([episode[t].observation['qpos'] for t in range(max_T)])  # (T,8)
        act_arr  = np.stack(action_traj)                                             # (T,8)

        img_arrays = {cam: np.stack([episode[t].observation['images'][cam] for t in range(max_T)])
                      for cam in camera_names}

        tmp_path   = os.path.join(dataset_dir, f'episode_{idx}.tmp')
        final_path = os.path.join(dataset_dir, f'episode_{idx}.hdf5')

        t0 = time.time()
        with h5py.File(tmp_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['success'] = True  # only save successful episodes

            obs_grp = root.create_group('observations')
            obs_grp.create_dataset('qpos', data=qpos_arr)
            root.create_dataset('action', data=act_arr)

            img_grp = obs_grp.create_group('images')
            for cam, arr in img_arrays.items():
                img_grp.create_dataset(cam, data=arr, dtype='uint8', chunks=(1,) + arr.shape[1:])

        os.replace(tmp_path, final_path)
        print(f"💾 Saved to {final_path} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Main loop – record successive episodes until user quits.
    # ------------------------------------------------------------------
    while True:
        print(f"\n==============================\nRecording episode_{episode_idx}.hdf5\n==============================")
        exit_program   = False
        restart_episode = False

        ts = env.reset()
        # ------------------------------------------------------------------
        # Reset keyboard-policy internal state for fresh episode
        # ------------------------------------------------------------------
        init_qpos = ts.observation['qpos']
        policy.target_ee_state[0:3] = init_qpos[0:3]        # EE position
        policy.target_ee_state[3:7] = init_qpos[3:7]        # EE orientation quaternion
        policy.gripper_status          = init_qpos[7]       # Read initial gripper state from task

        episode      = [ts]
        action_traj  = []

        # READY screen ---------------------------------------------------
        if use_3d_viewer:
            # 3D viewer mode - show the environment and wait for input
            print("🎬 Ready to record!  P – start | ESC – exit | R – reset scene | X – delete last")
            recording_started = False
            countdown_start_time = None
            last_countdown_sec = None
            
            while not recording_started:
                preview_start_time = time.time()
                
                # Update physics and viewer in preview mode
                action_preview = policy.get_action()
                ts_preview     = env.step(action_preview)
                
                # Sync the 3D viewer
                viewer.sync()
                
                # Check for user inputs through policy
                if not policy.running:  # ESC was pressed
                    exit_program = True
                    break
                elif countdown_start_time is None and not policy.recording and 'p' in policy.current_keys:
                    # P key pressed to start countdown
                    countdown_start_time = time.time()
                    last_countdown_sec = 3
                    print(f"Starting in {last_countdown_sec}...")
                    policy.current_keys.discard('p')  # Clear to avoid repeated triggers
                
                # Handle countdown (non-blocking)
                if countdown_start_time is not None:
                    elapsed = time.time() - countdown_start_time
                    remaining_sec = 3 - int(elapsed)
                    
                    if remaining_sec > 0 and remaining_sec != last_countdown_sec:
                        print(f"Starting in {remaining_sec}...")
                        last_countdown_sec = remaining_sec
                    elif elapsed >= 3.0:
                        recording_started = True
                        print("🔴 Started recording")
                        policy.recording = True  # enable policy recording flag
                        episode = [ts_preview]
                        break
                
                if 'r' in policy.current_keys:
                    # R key pressed to reset scene
                    print("🔄 Resetting scene...")
                    ts = env.reset()
                    # Reset policy state
                    init_qpos = ts.observation['qpos']
                    policy.target_ee_state[0:3] = init_qpos[0:3]
                    policy.target_ee_state[3:7] = init_qpos[3:7]
                    policy.gripper_status = init_qpos[7]
                    ts_preview = ts
                    episode = [ts]
                    # Clear the key press
                    policy.current_keys.discard('r')
                    print("✅ Scene reset complete")
                
                if 'x' in policy.current_keys:
                    # X key pressed to delete last episode
                    if episode_idx == 0:
                        print("🚫 No previous episode to delete.")
                    else:
                        last_idx  = episode_idx - 1
                        last_path = os.path.join(dataset_dir, f'episode_{last_idx}.hdf5')
                        if os.path.isfile(last_path):
                            os.remove(last_path)
                            episode_idx -= 1
                            print(f"🗑️  Deleted {last_path}. Now recording episode_{episode_idx}.")
                    # Clear the key press to avoid repeated deletion
                    policy.current_keys.discard('x')
                
                # Maintain consistent timing in preview
                elapsed = time.time() - preview_start_time
                if elapsed < DT:
                    time.sleep(DT - elapsed)
                
        elif onscreen_render:
            # Original OpenCV preview mode
            img0_bgr = cv2.cvtColor(ts.observation['images'][render_cam_name], cv2.COLOR_RGB2BGR)
            img0_res = cv2.resize(img0_bgr, (screen_w, screen_h))
            cv2.imshow(window_name, img0_res)

            print("🎬 Ready to record!  P – start | ESC – exit | F – fullscreen | R – reset scene | X – delete last")
            recording_started = False
            countdown_start_time = None
            last_countdown_sec = None
            
            while not recording_started:
                # Update live preview (not yet recording)
                action_preview = policy.get_action()
                ts_preview     = env.step(action_preview)
                img_bgr        = cv2.cvtColor(ts_preview.observation['images'][render_cam_name], cv2.COLOR_RGB2BGR)
                img_res        = cv2.resize(img_bgr, (screen_w, screen_h))

                # Check if countdown is active
                if countdown_start_time is not None:
                    elapsed = time.time() - countdown_start_time
                    remaining_sec = 3 - int(elapsed)
                    
                    if elapsed >= 3.0:
                        # Countdown complete, start recording
                        recording_started = True
                        print("🔴 Started recording")
                        policy.recording = True
                        episode = [ts_preview]
                        break
                    else:
                        # Show countdown on screen
                        if remaining_sec > 0 and remaining_sec != last_countdown_sec:
                            print(f"Starting in {remaining_sec}...")
                            last_countdown_sec = remaining_sec
                        
                        # Display large countdown number
                        cv2.putText(img_res, str(max(1, remaining_sec)), (screen_w//2 - 80, screen_h//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 15)
                else:
                    # Normal ready screen - simplified
                    cv2.putText(img_res, f"EPISODE {episode_idx}", (50, screen_h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 5)
                    cv2.putText(img_res, "READY TO RECORD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (0, 255, 255), 6)
                    
                    # Speed indicator (kept for visibility)
                    speed_level = policy.current_speed_idx + 1
                    speed_value = policy.speeds[policy.current_speed_idx]
                    cv2.putText(img_res, f"Speed: {speed_level}/3 ({speed_value:.3f} m/s)", (50, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 255), 3)
                
                # Draw controls legend on bottom right (always visible)
                _draw_controls_legend(img_res, screen_w, screen_h, is_recording=False)
                cv2.imshow(window_name, img_res)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('p'), ord('P')) and countdown_start_time is None:
                    # Start countdown
                    countdown_start_time = time.time()
                    last_countdown_sec = 3
                    print("Starting in 3...")
                elif key == 27:  # ESC
                    exit_program = True
                    break
                elif key in (ord('f'), ord('F')):
                    prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    mode = cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
                elif 'r' in policy.current_keys:
                    # R key pressed to reset scene
                    print("🔄 Resetting scene...")
                    ts = env.reset()
                    # Reset policy state
                    init_qpos = ts.observation['qpos']
                    policy.target_ee_state[0:3] = init_qpos[0:3]
                    policy.target_ee_state[3:7] = init_qpos[3:7]
                    policy.gripper_status = init_qpos[7]
                    ts_preview = ts
                    episode = [ts]
                    # Update the displayed image
                    img_bgr = cv2.cvtColor(ts.observation['images'][render_cam_name], cv2.COLOR_RGB2BGR)
                    img_res = cv2.resize(img_bgr, (screen_w, screen_h))
                    # Clear the key press
                    policy.current_keys.discard('r')
                    print("✅ Scene reset complete")
                elif key in (ord('x'), ord('X')):
                    if episode_idx == 0:
                        print("🚫 No previous episode to delete.")
                    else:
                        last_idx  = episode_idx - 1
                        last_path = os.path.join(dataset_dir, f'episode_{last_idx}.hdf5')
                        if os.path.isfile(last_path):
                            os.remove(last_path)
                            episode_idx -= 1
                            print(f"🗑️  Deleted {last_path}. Now recording episode_{episode_idx}.")
                # else: ignore other keys

        if exit_program:
            break  # Exit outer while True

        # ---------------------------------------------------------------
        # Recording loop - maintain proper timing for consistent simulation speed
        # ---------------------------------------------------------------
        step              = 0
        success_detected  = False
        success_counter   = 0
        while True:
            step_start_time = time.time()
            
            action_traj.append(policy.get_action())
            ts = env.step(action_traj[-1])
            episode.append(ts)

            if use_3d_viewer:
                # Update 3D viewer
                viewer.sync()
                
                # Check for keyboard input through policy
                if not policy.running:  # ESC was pressed
                    exit_program = True
                    break
                elif 'r' in policy.current_keys:
                    print("🔁 Restart requested – discarding current episode.")
                    restart_episode = True
                    # Clear the key press
                    policy.current_keys.discard('r')
                    break
                    
                # Print status to console since we don't have overlay in 3D viewer
                if step % 50 == 0:  # Print every 50 steps to avoid spam
                    if success_detected:
                        remaining = max(0, (steps_until_stop_success - success_counter) * DT)
                        print(f"SUCCESS – stopping in {remaining:.1f}s")
                    else:
                        print(f"EP{episode_idx} REC {step+1}/{episode_len} | R-abort | ESC-exit")
                        
            elif onscreen_render:
                # Original OpenCV rendering
                img_bgr = cv2.cvtColor(ts.observation['images'][render_cam_name], cv2.COLOR_RGB2BGR)
                img_res = cv2.resize(img_bgr, (screen_w, screen_h))

                if success_detected:
                    remaining = max(0, (steps_until_stop_success - success_counter) * DT)
                    overlay   = f"SUCCESS – stopping in {remaining:.1f}s"
                    color = (0, 255, 0)
                else:
                    overlay = f"RECORDING EP{episode_idx}: {step+1}/{episode_len}"
                    color = (0, 0, 255)
                cv2.putText(img_res, overlay, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                
                # Speed indicator during recording
                speed_level = policy.current_speed_idx + 1
                speed_value = policy.speeds[policy.current_speed_idx]
                cv2.putText(img_res, f"Speed: {speed_level}/3 ({speed_value:.3f} m/s)", 
                            (50, screen_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 255), 3)
                
                # Draw controls legend on bottom right
                _draw_controls_legend(img_res, screen_w, screen_h, is_recording=True)
                cv2.imshow(window_name, img_res)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    exit_program = True
                    break
                # Use policy key state for restart to avoid arrow-up (82/'R') collision from OpenCV
                elif 'r' in policy.current_keys:
                    print("🔁 Restart requested – discarding current episode.")
                    restart_episode = True
                    # Clear the key press to prevent repeated triggers
                    policy.current_keys.discard('r')
                    break

            # Success detection
            if not success_detected and ts.reward == env.task.max_reward:
                success_detected = True
                success_counter  = 0
                print("✔️  Success reached – finishing in 3s")

            if success_detected:
                success_counter += 1
                if success_counter >= steps_until_stop_success:
                    print("Success countdown elapsed – stopping recording")
                    break

            step += 1
            if step >= episode_len:
                break
            if not policy.is_recording():
                break
            
            # Maintain consistent timing: ensure each step takes exactly DT seconds
            elapsed = time.time() - step_start_time
            if elapsed < DT:
                time.sleep(DT - elapsed)

        if exit_program:
            break

        # ---------------------------------------------------------------
        # Decide what to do with the episode just recorded
        # ---------------------------------------------------------------
        success_episode = success_detected and success_counter >= steps_until_stop_success

        if restart_episode or not success_episode:
            if restart_episode:
                print("🔄 Episode restarted – ready for new attempt.")
            else:
                print("⚠️  Episode did not reach success – discarding.")
            continue  # Start next attempt with same idx

        # Save & advance idx
        _save_episode(episode_idx, episode[:-1], action_traj)
        episode_idx += 1
        # Loop again for next episode
        continue

    # ------------------------------------------------------------------
    # Shutdown – user pressed ESC
    # ------------------------------------------------------------------
    policy.cleanup()
    if use_3d_viewer and viewer:
        viewer.close()
    elif onscreen_render:
        cv2.destroyAllWindows()

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--use_3d_viewer', action='store_true', help='Use 3D MuJoCo viewer instead of OpenCV preview')
    parser.add_argument('--disturb', action='store_true', help='Enable disturbance in simulation')

    main(vars(parser.parse_args()))
