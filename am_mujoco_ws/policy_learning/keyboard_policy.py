import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import threading
from pynput import keyboard

# Debug setting
DEBUG = True

def is_window_focused(window_keywords):
    """Check if a window matching keywords has focus"""
    if not window_keywords:
        return False
    
    from Xlib import display
    
    d = display.Display()
    window = d.get_input_focus().focus
    wmname = window.get_wm_name()
    
    if wmname is None:
        return False
    
    title_lower = wmname.lower()
    return any(kw.lower() in title_lower for kw in window_keywords)

# Base policy class copied from sim_teleop_policy_4dof.py but without ROS dependencies
class SimTeleopBasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.ee_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.ee_trajectory[0]['t'] == self.step_count:
            self.curr_ee_waypoint = self.ee_trajectory.pop(0)
        next_ee_waypoint = self.ee_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        ee_xyz, ee_quat, ee_gripper = self.interpolate(self.curr_ee_waypoint, next_ee_waypoint, self.step_count)
        
        # Inject noise
        if self.inject_noise:
            scale = 0.01
            ee_xyz = ee_xyz + np.random.uniform(-scale, scale, ee_xyz.shape)

        action_ee = np.concatenate([ee_xyz, ee_quat, [ee_gripper]])
        
        self.step_count += 1
        return action_ee


# Keyboard-only control policy using pynput
class MinecraftStyleKeyboardPolicy(SimTeleopBasePolicy):
    def __init__(self, inject_noise=False, task_name=None, window_keywords=None):
        super().__init__(inject_noise)
        
        # Task-aware setup
        self.task_name = task_name or ""
        self.is_umi_task = "umi" in self.task_name.lower()
        
        # Window focus tracking
        self.window_keywords = window_keywords if window_keywords is not None else []
        
        # Initial state - Match the robot's natural starting orientation to avoid initial movement
        # Robot starts with EE at [0.377, 0.0, 1.124] with identity orientation
        # Coordinate frame transformation is now handled in the control system for 4DOF
        self.target_ee_state = np.array([0.0, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0])  # Position + quaternion (w,x,y,z)
        self.gripper_status = 0.0  # UMI-style: 0=open, 1=closed (start open)
        self.recording = False
        
        # Movement settings
        # SPEED ADJUSTMENT: You can modify these values to increase or decrease movement speeds
        self.speeds = [0.001, 0.002, 0.004]  # Movement speeds for levels 1-3 (m/s)
        self.current_speed_idx = 1  # Default to speed level 2
        # ROTATION SPEED: You can modify this value to increase or decrease rotation speed
        self.rotation_speed = 0.008
        
        # Key state tracking
        self.current_keys = set()  # Set of currently pressed keys
        self.lock = threading.Lock()  # Thread safety
        
        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release)
        self.listener.start()
        
        # Store the current running state
        self.running = True
        
        # Ensure cleanup happens on exit
        import atexit
        atexit.register(self.cleanup)
        
        # Instructions for the user - task-aware
        print("Keyboard Controls:")
        print("  WASD: Move horizontally")
        print("  Space: Move up")
        print("  Shift: Move down")
        print("  Up/Down arrows: Rotate pitch (look up/down)")
        print("  Left/Right arrows: Rotate yaw (turn left/right)")
        print("  Z/C: Roll left/right")
        print("  E: Close gripper gradually (hold)")
        print("  Q: Open  gripper gradually (hold)")
        print("  R: Start/stop recording")
        print("  1-3: Adjust speed")
        print("  Esc: Exit")
        if self.is_umi_task:
            print(f"  Task: {self.task_name} (UMI robot - using original coordinate frame)")
        else:
            print(f"  Task: {self.task_name} (4DOF robot - using transformed coordinate frame)")
    
    def _on_press(self, key):
        """Handle key press events"""
        if not is_window_focused(self.window_keywords):
            return
        
        with self.lock:
            try:
                # For normal keys like 'a', 'b', etc.
                if hasattr(key, 'char') and key.char:
                    self.current_keys.add(key.char.lower())
                    
                    # One-shot actions
                    if key.char == 'r':
                        self.recording = not self.recording
                        print(f"Recording {'started' if self.recording else 'stopped'}")
                    elif key.char in '123':
                        self.current_speed_idx = int(key.char) - 1
                        print(f"Speed set to level {self.current_speed_idx + 1} ({self.speeds[self.current_speed_idx]} m/s)")
                # For special keys like arrow keys, space, shift, etc.
                else:
                    self.current_keys.add(key)
                    
                    # Exit on Escape
                    if key == keyboard.Key.esc:
                        self.running = False
                        return False  # Stop listener
                        
            except AttributeError:
                # Some keys might not have expected attributes
                pass
                
    def _on_release(self, key):
        """Handle key release events"""
        if not is_window_focused(self.window_keywords):
            return
        
        with self.lock:
            try:
                # For normal keys
                if hasattr(key, 'char') and key.char:
                    if key.char.lower() in self.current_keys:
                        self.current_keys.remove(key.char.lower())
                # For special keys
                else:
                    if key in self.current_keys:
                        self.current_keys.remove(key)
            except (AttributeError, KeyError):
                # Key might not be in the set or have expected attributes
                pass
    
    def is_recording(self):
        """Check if recording is active"""
        return self.recording
    
    def get_action(self):
        """Get action based on current key states"""
        if not self.running:
            import sys
            print("Exiting due to ESC key press")
            sys.exit(0)
            
        # Create rotation matrix from current orientation
        quat = self.target_ee_state[3:7]
        rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        
        # Forward vector is the first column of the rotation matrix
        forward = rot_matrix[:, 0]
        # Right vector is the second column of the rotation matrix
        right = rot_matrix[:, 1]
        # Up vector is the third column of the rotation matrix
        up = rot_matrix[:, 2]
        
        # Use thread-safe access to current_keys
        with self.lock:
            speed = self.speeds[self.current_speed_idx]
            
            # Print current keys for debugging
            if DEBUG and self.current_keys:
                print(f"Current keys: {self.current_keys}")
            
            # Movement controls
            if {'w', 'g'} & self.current_keys:  # Forward (also 'g')
                self.target_ee_state[0:3] += forward * speed
            if {'s', 'b'} & self.current_keys:  # Backward (also 'b')
                self.target_ee_state[0:3] -= forward * speed
            if {'a', 'v'} & self.current_keys:  # Left (also 'v')
                self.target_ee_state[0:3] += right * speed
            if {'d', 'n'} & self.current_keys:  # Right (also 'n')
                self.target_ee_state[0:3] -= right * speed
            if keyboard.Key.space in self.current_keys:  # Up
                self.target_ee_state[0:3] += up * speed * 0.5
            if keyboard.Key.shift in self.current_keys:  # Down
                self.target_ee_state[0:3] -= up * speed * 0.5
                
            # Gripper incremental control (hold E to close, Q to open)
            grip_speed = 0.02  # change per timestep (~20 steps full swing)
            if 'e' in self.current_keys:
                self.gripper_status = max(0.0, self.gripper_status - grip_speed)
            if 'q' in self.current_keys:
                self.gripper_status = min(1.0, self.gripper_status + grip_speed)
            
            # Rotation controls - Task-aware mapping
            rotation_applied = False

            current_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            
            # UMI robot has built-in pitch90 wrapper, use original coordinate frame
            # Pitch (up/down rotation) - Y-axis
            if keyboard.Key.up in self.current_keys:
                rot = R.from_rotvec(np.array([0,  -self.rotation_speed, 0]))  # Y-axis rotation
                current_rot = current_rot * rot
                rotation_applied = True
            if keyboard.Key.down in self.current_keys:
                rot = R.from_rotvec(np.array([0, self.rotation_speed, 0]))  # Y-axis rotation
                current_rot = current_rot * rot
                rotation_applied = True

            # Yaw (left/right arrow) - Z-axis
            if keyboard.Key.left in self.current_keys:
                rot = R.from_rotvec(np.array([0, 0,  self.rotation_speed]))  # Z-axis rotation
                current_rot = current_rot * rot
                rotation_applied = True
            if keyboard.Key.right in self.current_keys:
                rot = R.from_rotvec(np.array([0, 0, -self.rotation_speed]))  # Z-axis rotation
                current_rot = current_rot * rot
                rotation_applied = True

            # Roll (Z / C keys) - X-axis remains the same for both robots
            if 'z' in self.current_keys:
                rot = R.from_rotvec(np.array([ -self.rotation_speed, 0, 0]))
                current_rot = current_rot * rot
                rotation_applied = True
            if 'c' in self.current_keys:
                rot = R.from_rotvec(np.array([ self.rotation_speed, 0, 0]))
                current_rot = current_rot * rot
                rotation_applied = True

            if rotation_applied:
                new_quat = current_rot.as_quat()
                self.target_ee_state[3:7] = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
        
        # Normalize quaternion if rotation was applied
        if rotation_applied:
            quat_norm = np.linalg.norm(self.target_ee_state[3:7])
            if quat_norm > 1e-10:  # Prevent division by zero
                self.target_ee_state[3:7] /= quat_norm
        
        # Create final action *after* gripper status update
        # Order: [x, y, z, qw, qx, qy, qz, gripper]
        action = np.zeros(8)
        action[0:3] = self.target_ee_state[0:3]
        action[3:7] = self.target_ee_state[3:7]
        action[7]   = self.gripper_status

        return action
        
    def generate_trajectory(self, ts_first):
        """Initialize the trajectory waypoints for the base policy"""
        qpos_first = ts_first.observation['qpos']
        self.curr_ee_waypoint = {
            't': 0,
            'xyz': qpos_first[0:3],
            'quat': qpos_first[3:7],
            'gripper': qpos_first[7],
        }
        self.ee_trajectory = [{
            't': 1,
            'xyz': qpos_first[0:3],
            'quat': qpos_first[3:7],
            'gripper': qpos_first[7],
        }]
    
    def cleanup(self):
        """Cleanup keyboard listener resources"""
        if hasattr(self, 'listener') and self.listener.running:
            self.listener.stop() 