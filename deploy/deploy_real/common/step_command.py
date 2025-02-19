import time
import math
import numpy as np
import matplotlib.pyplot as plt


def linear_map(val, in_min, in_max, out_min, out_max):
    """Linearly map val from [in_min, in_max] to [out_min, out_max]."""
    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)

def quaternion_multiply(q1, q2):
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=np.float32)

def quaternion_rotate(q, v):
    """Rotate vector v by quaternion q."""
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    v_q = np.concatenate(([0.0], v))
    rotated = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
    return rotated[1:]

def yaw_to_quaternion(yaw):
    """Convert yaw angle (radians) to a quaternion (w, x, y, z)."""
    half_yaw = yaw / 2.0
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float32)

def combine_frame_transforms(pos, quat, rel_pos, rel_quat):
    """
    Combine two transforms:
      T_new = T * T_rel
    where T is given by (pos, quat) and T_rel by (rel_pos, rel_quat).
    """
    new_pos = pos + quaternion_rotate(quat, rel_pos)
    new_quat = quaternion_multiply(quat, rel_quat)
    return new_pos, new_quat

# ----------------------
# StepCommand Class
# ----------------------
class StepCommand:
    def __init__(self, current_left_pose, current_right_pose):
        """
        Initialize with the current foot poses.
        Each pose is a 7-dimensional vector: [x, y, z, qw, qx, qy, qz].
        Both next_ctarget_left and next_ctarget_right are initialized to these values.
        Also, store the maximum ranges for x, y, and theta.
          - x_range: (-0.2, 0.2)
          - y_range: (0.2, 0.4)
          - theta_range: (-0.3, 0.3)
        """
        self.next_ctarget_left = current_left_pose.copy()
        self.next_ctarget_right = current_right_pose.copy()
        self.next_ctime_left = 0.8
        self.next_ctime_right = 1.2
        self.delta_ctime = 0.4  # Fixed time delta for a new step
        self.max_range = {
            'x_range': (-0.2, 0.2),
            'y_range': (0.2, 0.4),
            'theta_range': (-0.3, 0.3)
        }

    def compute_relstep_left(self, lx, ly, rx):
        """
        Compute the left foot relative step based on remote controller inputs.
        
        Mapping:
          - x: map ly in [-1,1] to self.max_range['x_range'].
          - y: baseline for left is self.max_range['y_range'][0]. If lx > 0,
               add an offset mapping lx in [0,1] to [0, self.max_range['y_range'][1]-self.max_range['y_range'][0]].
          - z: fixed at 0.
          - rotation: map rx in [-1,1] to self.max_range['theta_range'] and convert to quaternion.
        """
        delta_x = linear_map(ly, -1, 1, self.max_range['x_range'][0], self.max_range['x_range'][1])
        baseline_left = self.max_range['y_range'][0]
        extra_y = linear_map(lx, 0, 1, 0, self.max_range['y_range'][1] - self.max_range['y_range'][0]) if lx > 0 else 0.0
        delta_y = baseline_left + extra_y
        delta_z = 0.0
        theta = linear_map(rx, -1, 1, self.max_range['theta_range'][0], self.max_range['theta_range'][1])
        q = yaw_to_quaternion(theta)
        return np.array([delta_x, delta_y, delta_z, q[0], q[1], q[2], q[3]], dtype=np.float32)

    def compute_relstep_right(self, lx, ly, rx):
        """
        Compute the right foot relative step based on remote controller inputs.
        
        Mapping:
          - x: map ly in [-1,1] to self.max_range['x_range'].
          - y: baseline for right is the negative of self.max_range['y_range'][0]. If lx < 0,
               add an offset mapping lx in [-1,0] to [- (self.max_range['y_range'][1]-self.max_range['y_range'][0]), 0].
          - z: fixed at 0.
          - rotation: map rx in [-1,1] to self.max_range['theta_range'] and convert to quaternion.
        """
        delta_x = linear_map(ly, -1, 1, self.max_range['x_range'][0], self.max_range['x_range'][1])
        baseline_right = -self.max_range['y_range'][0]
        extra_y = linear_map(lx, -1, 0, -(self.max_range['y_range'][1] - self.max_range['y_range'][0]), 0) if lx < 0 else 0.0
        delta_y = baseline_right + extra_y
        delta_z = 0.0
        theta = linear_map(rx, -1, 1, self.max_range['theta_range'][0], self.max_range['theta_range'][1])
        q = yaw_to_quaternion(theta)
        return np.array([delta_x, delta_y, delta_z, q[0], q[1], q[2], q[3]], dtype=np.float32)

    def get_next_ctarget(self, remote_controller, count):
        """
        Given the remote controller inputs and elapsed time (count),
        compute relative step commands for left and right feet and update
        the outdated targets accordingly.

        Update procedure:
          - When the left foot is due (count > next_ctime_left), update it by combining
            the right foot target with the left relative step.
          - Similarly, when the right foot is due (count > next_ctime_right), update it using
            the left foot target and the right relative step.
        
        Returns:
            A concatenated 14-dimensional vector:
            [left_foot_target (7D), right_foot_target (7D)]
        """
        lx = remote_controller.lx
        ly = remote_controller.ly
        rx = remote_controller.rx

        # Compute relative steps using the internal methods.
        relstep_left = self.compute_relstep_left(lx, ly, rx)
        relstep_right = self.compute_relstep_right(lx, ly, rx)
        from icecream import ic

        # Update left foot target if its scheduled time has elapsed.
        if count > self.next_ctime_left:
            self.next_ctime_left = self.next_ctime_right + self.delta_ctime
            new_pos, new_quat = combine_frame_transforms(
                self.next_ctarget_right[:3],
                self.next_ctarget_right[3:7],
                relstep_left[:3],
                relstep_left[3:7],
            )
            self.next_ctarget_left[:3] = new_pos
            self.next_ctarget_left[3:7] = new_quat

        # Update right foot target if its scheduled time has elapsed.
        if count > self.next_ctime_right:
            self.next_ctime_right = self.next_ctime_left + self.delta_ctime
            new_pos, new_quat = combine_frame_transforms(
                self.next_ctarget_left[:3],
                self.next_ctarget_left[3:7],
                relstep_right[:3],
                relstep_right[3:7],
            )
            self.next_ctarget_right[:3] = new_pos
            self.next_ctarget_right[3:7] = new_quat

        # Return the concatenated target: left (7D) followed by right (7D).
        return (np.concatenate((self.next_ctarget_left, self.next_ctarget_right)),
                (self.next_ctime_left - count), 
                (self.next_ctarget_right - count))



# For testing purposes, we define a dummy remote controller that mimics the attributes lx, ly, and rx.
class DummyRemoteController:
    def __init__(self, lx=0.0, ly=0.0, rx=0.0):
        self.lx = lx  # lateral command input in range [-1,1]
        self.ly = ly  # forward/backward command input in range [-1,1]
        self.rx = rx  # yaw command input in range [-1,1]

if __name__ == "__main__":
    # Initial foot poses (7D each): [x, y, z, qw, qx, qy, qz]
    current_left_pose = np.array([0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    current_right_pose = np.array([0.0, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Create an instance of StepCommand with the initial poses.
    step_command = StepCommand(current_left_pose, current_right_pose)

    # Create a dummy remote controller.
    dummy_remote = DummyRemoteController()

    # Set up matplotlib for interactive plotting.
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Footstep Target Visualization")

    print("Starting test. Press Ctrl+C to exit.")
    start_time = time.time()
    try:
        while True:
            elapsed = time.time() - start_time

            # For demonstration, vary the controller inputs over time:
            #   - ly oscillates between -1 and 1 (forward/backward)
            #   - lx oscillates between -1 and 1 (lateral left/right)
            #   - rx is held at 0 (no yaw command)
            # dummy_remote.ly = math.sin(elapsed)    # forward/backward command
            # dummy_remote.lx = math.cos(elapsed)    # lateral command
            dummy_remote.ly = 0.0
            dummy_remote.lx = 0.0
            dummy_remote.rx = 1.                  # no yaw

            # Get the current footstep target (14-dimensional)
            ctarget = step_command.get_next_ctarget(dummy_remote, elapsed)
            print("Time: {:.2f} s, ctarget: {}".format(elapsed, ctarget))

            # Extract left foot and right foot positions:
            # Left foot: indices 0:7 (position: [0:3], quaternion: [3:7])
            left_pos = ctarget[0:3]  # [x, y, z]
            left_quat = ctarget[3:7] # [qw, qx, qy, qz]
            # Right foot: indices 7:14 (position: [7:10], quaternion: [10:14])
            right_pos = ctarget[7:10]
            right_quat = ctarget[10:14]

            # For visualization, we use only the x and y components.
            left_x, left_y = left_pos[0], left_pos[1]
            right_x, right_y = right_pos[0], right_pos[1]

            # Assuming rotation only about z, compute yaw angle from quaternion:
            # yaw = 2 * atan2(qz, qw)
            left_yaw = 2 * math.atan2(left_quat[3], left_quat[0])
            right_yaw = 2 * math.atan2(right_quat[3], right_quat[0])

            # Clear and redraw the plot.
            ax.cla()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Footstep Target Visualization")

            # Plot the left and right foot positions.
            ax.plot(left_x, left_y, 'bo', label='Left Foot')
            ax.plot(right_x, right_y, 'ro', label='Right Foot')

            # Draw an arrow for each foot to indicate orientation.
            arrow_length = 0.1
            ax.arrow(left_x, left_y,
                     arrow_length * math.cos(left_yaw),
                     arrow_length * math.sin(left_yaw),
                     head_width=0.03, head_length=0.03, fc='b', ec='b')
            ax.arrow(right_x, right_y,
                     arrow_length * math.cos(right_yaw),
                     arrow_length * math.sin(right_yaw),
                     head_width=0.03, head_length=0.03, fc='r', ec='r')

            ax.legend()
            plt.pause(0.001)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Test terminated by user.")
    finally:
        plt.ioff()
        plt.show()