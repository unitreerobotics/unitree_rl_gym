# Unitree H1 Description (URDF & MJCF)

## Overview

This package includes a streamlined robot description (URDF & MJCF) for the [Unitree H1](https://www.unitree.com/h1/), developed by [Unitree Robotics](https://www.unitree.com/).

<p align="center">
  <img src="h1_5.png" width="500"/>
</p>

Unitree H1 have 51 DOFs:

```text
root [⚓] => /pelvis/
    left_hip_yaw_joint [⚙+Z] => /left_hip_yaw_link/
        left_hip_pitch_joint [⚙+Y] => /left_hip_pitch_link/
            left_hip_roll_joint [⚙+X] => /left_hip_roll_link/
                left_knee_joint [⚙+Y] => /left_knee_link/
                    left_ankle_pitch_joint [⚙+Y] => /left_ankle_pitch_link/
                        left_ankle_roll_joint [⚙+X] => /left_ankle_roll_link/
    right_hip_yaw_joint [⚙+Z] => /right_hip_yaw_link/
        right_hip_pitch_joint [⚙+Y] => /right_hip_pitch_link/
            right_hip_roll_joint [⚙+X] => /right_hip_roll_link/
                right_knee_joint [⚙+Y] => /right_knee_link/
                    right_ankle_pitch_joint [⚙+Y] => /right_ankle_pitch_link/
                        right_ankle_roll_joint [⚙+X] => /right_ankle_roll_link/
    torso_joint [⚙+Z] => /torso_link/
        left_shoulder_pitch_joint [⚙+Y] => /left_shoulder_pitch_link/
            left_shoulder_roll_joint [⚙+X] => /left_shoulder_roll_link/
                left_shoulder_yaw_joint [⚙+Z] => /left_shoulder_yaw_link/
                    left_elbow_pitch_joint [⚙+Y] => /left_elbow_pitch_link/
                        left_elbow_roll_joint [⚙+X] => /left_elbow_roll_link/
                            left_wrist_pitch_joint [⚙+Y] => /left_wrist_pitch_link/
                                left_wrist_yaw_joint [⚙+Z] => /left_wrist_yaw_link/
                                    L_base_link_joint [⚓] => /L_hand_base_link/
                                        L_thumb_proximal_yaw_joint [⚙+Z] => /L_thumb_proximal_base/
                                            L_thumb_proximal_pitch_joint [⚙-Z] => /L_thumb_proximal/
                                                L_thumb_intermediate_joint [⚙-Z] => /L_thumb_intermediate/
                                                    L_thumb_distal_joint [⚙-Z] => /L_thumb_distal/
                                        L_index_proximal_joint [⚙-Z] => /L_index_proximal/
                                            L_index_intermediate_joint [⚙-Z] => /L_index_intermediate/
                                        L_middle_proximal_joint [⚙-Z] => /L_middle_proximal/
                                            L_middle_intermediate_joint [⚙-Z] => /L_middle_intermediate/
                                        L_ring_proximal_joint [⚙-Z] => /L_ring_proximal/
                                            L_ring_intermediate_joint [⚙-Z] => /L_ring_intermediate/
                                        L_pinky_proximal_joint [⚙-Z] => /L_pinky_proximal/
                                            L_pinky_intermediate_joint [⚙-Z] => /L_pinky_intermediate/
        right_shoulder_pitch_joint [⚙+Y] => /right_shoulder_pitch_link/
            right_shoulder_roll_joint [⚙+X] => /right_shoulder_roll_link/
                right_shoulder_yaw_joint [⚙+Z] => /right_shoulder_yaw_link/
                    right_elbow_pitch_joint [⚙+Y] => /right_elbow_pitch_link/
                        right_elbow_roll_joint [⚙+X] => /right_elbow_roll_link/
                            right_wrist_pitch_joint [⚙+Y] => /right_wrist_pitch_link/
                                right_wrist_yaw_joint [⚙+Z] => /right_wrist_yaw_link/
                                    R_base_link_joint [⚓] => /R_hand_base_link/
                                        R_thumb_proximal_yaw_joint [⚙-Z] => /R_thumb_proximal_base/
                                            R_thumb_proximal_pitch_joint [⚙+Z] => /R_thumb_proximal/
                                                R_thumb_intermediate_joint [⚙+Z] => /R_thumb_intermediate/
                                                    R_thumb_distal_joint [⚙+Z] => /R_thumb_distal/
                                        R_index_proximal_joint [⚙+Z] => /R_index_proximal/
                                            R_index_intermediate_joint [⚙+Z] => /R_index_intermediate/
                                        R_middle_proximal_joint [⚙+Z] => /R_middle_proximal/
                                            R_middle_intermediate_joint [⚙+Z] => /R_middle_intermediate/
                                        R_ring_proximal_joint [⚙+Z] => /R_ring_proximal/
                                            R_ring_intermediate_joint [⚙+Z] => /R_ring_intermediate/
                                        R_pinky_proximal_joint [⚙+Z] => /R_pinky_proximal/
                                            R_pinky_intermediate_joint [⚙+Z] => /R_pinky_intermediate/
```

## Visulization with [MuJoCo](https://github.com/google-deepmind/mujoco)

1. Open MuJoCo Viewer

   ```bash
   pip install mujoco
   python -m mujoco.viewer
   ```

2. Drag and drop the MJCF model file (`scene.xml`) to the MuJoCo Viewer.
