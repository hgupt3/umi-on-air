"""ARKit-based teleoperation policy using MujocoAR.

This module provides an ARKitPolicy class that receives pose data from an iPhone
running the MujocoAR app and converts it to robot end-effector actions.

Requires: pip install mujoco_ar
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import atexit
from mujoco_ar import MujocoARConnector


class ARKitPolicy:
    """ARKit-based teleoperation policy using MujocoAR.
    
    This policy receives pose data from an iPhone running the MujocoAR app
    and converts it to 8D robot actions [x, y, z, qw, qx, qy, qz, gripper].
    
    Controls:
        - Phone position/rotation: Controls end-effector pose
        - Button (pressed): Close gripper (1.0)
        - Button (released): Open gripper (0.0)
        - Toggle: Start/stop recording
    """
    
    def __init__(self, port=8888, scale=1.0, task_name=None):
        """Initialize ARKit policy.
        
        Args:
            port: WebSocket port for MujocoAR connection (default: 8888)
            scale: Position scaling factor (default: 1.0)
            task_name: Name of the task (for logging)
        """
        self.port = port
        self.scale = scale
        self.task_name = task_name or ""
        
        # Initialize MujocoAR connector (flexible setup - no MuJoCo model linking)
        self.connector = MujocoARConnector(port=port)
        
        # Calibration state
        self.calibrated = False
        self.reference_arkit_pos = None
        self.reference_arkit_rot = None
        self.reference_ee_pos = np.array([0.0, 0.0, 1.2])  # Default EE position
        self.reference_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (wxyz)
        
        # Current state
        self.target_ee_state = np.array([0.0, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0])  # [x,y,z, qw,qx,qy,qz]
        self.gripper_status = 0.0  # 0=open, 1=closed (start open)
        self.grip_speed = 0.05  # Gradual gripper change per timestep
        
        # Recording state (directly mapped to toggle: true=recording, false=not)
        self.recording = False
        
        # Running state
        self.running = True
        
        # Thread lock for state access
        self.lock = threading.Lock()
        
        # Start the connector
        self.connector.start()
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        # Print instructions
        print("=" * 60)
        print("ARKit Teleoperation Policy")
        print("=" * 60)
        print(f"  WebSocket server running on port {port}")
        print(f"  Task: {self.task_name}")
        print()
        print("  Connect your iPhone:")
        print("    1. Open MujocoAR app on iPhone")
        print("    2. Enter this computer's IP address and port")
        print("    3. Tap Connect")
        print()
        print("  Controls:")
        print("    Phone motion  → End-effector pose")
        print("    Button press  → Close gripper")
        print("    Button release → Open gripper")
        print("    Toggle        → Start/stop recording")
        print("=" * 60)
    
    def set_reference_pose(self, ee_pos, ee_quat=None):
        """Set the reference end-effector pose for calibration.
        
        This should be called with the robot's initial EE pose before
        starting teleoperation. The ARKit pose will be relative to this.
        
        Args:
            ee_pos: Reference EE position [x, y, z]
            ee_quat: Reference EE quaternion [qw, qx, qy, qz] (optional)
        """
        self.reference_ee_pos = np.array(ee_pos)
        if ee_quat is not None:
            self.reference_ee_quat = np.array(ee_quat)
        
        # Reset calibration so next ARKit frame becomes the reference
        self.calibrated = False
        self.reference_arkit_pos = None
        self.reference_arkit_rot = None
        
        # Also update current target state
        self.target_ee_state[0:3] = self.reference_ee_pos.copy()
        self.target_ee_state[3:7] = self.reference_ee_quat.copy()
    
    def reset_calibration(self):
        """Reset calibration - next ARKit frame will be the new reference."""
        self.calibrated = False
        self.reference_arkit_pos = None
        self.reference_arkit_rot = None
        
        # Also call MujocoAR's reset_position for its internal calibration
        try:
            self.connector.reset_position()
        except Exception:
            pass  # May fail if not connected yet
        
        print("[ARKit] Calibration reset - next frame will be reference")
    
    def vibrate(self, sharpness=0.8, intensity=0.4, duration=0.01):
        """Trigger haptic feedback on the connected iPhone.
        
        Args:
            sharpness: Vibration sharpness (0.0 to 1.0)
            intensity: Vibration intensity (0.0 to 1.0)
            duration: Vibration duration in seconds
        """
        try:
            self.connector.vibrate(sharpness=sharpness, intensity=intensity, duration=duration)
        except Exception:
            pass  # May fail if not connected
    
    def is_recording(self):
        """Check if recording is active (controlled by toggle)."""
        return self.recording
    
    def is_connected(self):
        """Check if an iPhone is connected with valid data."""
        try:
            data = self.connector.get_latest_data()
            if data is None:
                return False
            raw_pos = data.get("position")
            if raw_pos is None or None in np.array(raw_pos).flatten():
                return False
            return True
        except:
            return False
    
    def get_action(self):
        """Get current action based on ARKit data.
        
        Returns:
            8D numpy array: [x, y, z, qw, qx, qy, qz, gripper]
        """
        # Get latest ARKit data
        data = self.connector.get_latest_data()
        
        if data is None:
            # No data yet - return current state
            action = np.zeros(8)
            action[0:3] = self.target_ee_state[0:3]
            action[3:7] = self.target_ee_state[3:7]
            action[7] = self.gripper_status
            return action
        
        # Extract data from MujocoAR
        # position: (3,) array, rotation: (3,3) matrix, button: bool, toggle: bool
        raw_pos = data.get("position")
        raw_rot = data.get("rotation")
        button_pressed = data.get("button", False)
        toggle_state = data.get("toggle", False)
        
        # Check if position/rotation data is valid (not None)
        if raw_pos is None or raw_rot is None or None in np.array(raw_pos).flatten():
            # Invalid data - return current state
            action = np.zeros(8)
            action[0:3] = self.target_ee_state[0:3]
            action[3:7] = self.target_ee_state[3:7]
            action[7] = self.gripper_status
            return action
        
        arkit_pos = np.array(raw_pos).flatten()
        arkit_rot_matrix = np.array(raw_rot)
        
        with self.lock:
            # Handle calibration on first frame with valid data
            if not self.calibrated:
                self.reference_arkit_pos = arkit_pos.copy()
                self.reference_arkit_rot = arkit_rot_matrix.copy()
                self.calibrated = True
                print(f"[ARKit] Calibrated! Reference ARKit pos: {arkit_pos}")
            
            # Calculate relative position from ARKit reference
            arkit_delta_pos = arkit_pos - self.reference_arkit_pos
            
            # Apply scaling and add to reference EE position
            ee_pos = self.reference_ee_pos + arkit_delta_pos * self.scale
            
            # Calculate relative rotation
            # R_relative = R_current * R_reference^(-1)
            ref_rot = R.from_matrix(self.reference_arkit_rot)
            current_rot = R.from_matrix(arkit_rot_matrix)
            relative_rot = current_rot * ref_rot.inv()
            
            # Apply relative rotation to reference EE orientation
            ref_ee_rot = R.from_quat([
                self.reference_ee_quat[1],  # x
                self.reference_ee_quat[2],  # y
                self.reference_ee_quat[3],  # z
                self.reference_ee_quat[0],  # w (scipy uses xyzw)
            ])
            final_rot = relative_rot * ref_ee_rot
            
            # Convert back to wxyz quaternion format
            quat_xyzw = final_rot.as_quat()
            ee_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # wxyz
            
            # Normalize quaternion
            quat_norm = np.linalg.norm(ee_quat)
            if quat_norm > 1e-10:
                ee_quat /= quat_norm
            
            # Update target state
            self.target_ee_state[0:3] = ee_pos
            self.target_ee_state[3:7] = ee_quat
            
            # Gripper control: button pressed = gradually close, released = gradually open
            if button_pressed:
                self.gripper_status = min(1.0, self.gripper_status + self.grip_speed)
            else:
                self.gripper_status = max(0.0, self.gripper_status - self.grip_speed)
            
            # Toggle state directly controls recording (true=recording, false=not)
            if toggle_state != self.recording:
                self.recording = toggle_state
                print(f"[ARKit] Recording {'started' if self.recording else 'stopped'}")
        
        # Build action array: [x, y, z, qw, qx, qy, qz, gripper]
        action = np.zeros(8)
        action[0:3] = self.target_ee_state[0:3]
        action[3:7] = self.target_ee_state[3:7]
        action[7] = self.gripper_status
        
        return action
    
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        print("[ARKit] Policy cleanup complete")
