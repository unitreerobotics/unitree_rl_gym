# DOF order reference

Every joint-indexed vector in this repo - observations, actions, default
angles, gains - follows the **asset DOF order**: the order movable joints are
declared in the URDF file. The Python configs map joint *names* to values
(`default_joint_angles`, `stiffness`, ...), so the configs themselves never
show the order; the URDF is the only source of index truth. This page lists
it for every shipped URDF so it does not have to be reverse-engineered
(asked in #94, #32, and others).

## Isaac Gym order is NOT Isaac Sim order

Two orderings exist for the same robot, and mixing them is a classic silent
deployment bug:

- **Isaac Gym** (this repo's training): DOF order = URDF declaration order -
  the tables below. Ground truth at runtime: `gym.get_asset_dof_names(asset)`.
- **Isaac Sim / PhysX articulations**: DOF order = breadth-first traversal of
  the kinematic tree (e.g. `left_hip_pitch, right_hip_pitch, torso,
  left_hip_roll, ...` for a humanoid). Ground truth: `articulation.dof_names`.

A policy trained here and deployed in Isaac Sim (or any other stack) must be
remapped **by name**, never by copying index lists:

```python
# gym_names: from gym.get_asset_dof_names(asset) at training time
# sim_names: from the deployment side
perm = [gym_names.index(n) for n in sim_names]   # sim index -> gym index
```

## Regenerating this page

```python
import re
text = open("path/to/robot.urdf").read()
joints = re.findall(r'<joint[^>]*name="([^"]+)"[^>]*type="(?:revolute|prismatic|continuous)"', text)
print(list(enumerate(joints)))
```


## g1_12dof.urdf (12 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |

## g1_23dof.urdf / g1_23dof_rev_1_0.urdf (23 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `left_shoulder_pitch_joint` |
| 14 | `left_shoulder_roll_joint` |
| 15 | `left_shoulder_yaw_joint` |
| 16 | `left_elbow_joint` |
| 17 | `left_wrist_roll_joint` |
| 18 | `right_shoulder_pitch_joint` |
| 19 | `right_shoulder_roll_joint` |
| 20 | `right_shoulder_yaw_joint` |
| 21 | `right_elbow_joint` |
| 22 | `right_wrist_roll_joint` |

## g1_29dof.urdf / g1_29dof_rev_1_0.urdf (29 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `waist_roll_joint` |
| 14 | `waist_pitch_joint` |
| 15 | `left_shoulder_pitch_joint` |
| 16 | `left_shoulder_roll_joint` |
| 17 | `left_shoulder_yaw_joint` |
| 18 | `left_elbow_joint` |
| 19 | `left_wrist_roll_joint` |
| 20 | `left_wrist_pitch_joint` |
| 21 | `left_wrist_yaw_joint` |
| 22 | `right_shoulder_pitch_joint` |
| 23 | `right_shoulder_roll_joint` |
| 24 | `right_shoulder_yaw_joint` |
| 25 | `right_elbow_joint` |
| 26 | `right_wrist_roll_joint` |
| 27 | `right_wrist_pitch_joint` |
| 28 | `right_wrist_yaw_joint` |

## g1_29dof_lock_waist.urdf / g1_29dof_lock_waist_rev_1_0.urdf (27 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `left_shoulder_pitch_joint` |
| 14 | `left_shoulder_roll_joint` |
| 15 | `left_shoulder_yaw_joint` |
| 16 | `left_elbow_joint` |
| 17 | `left_wrist_roll_joint` |
| 18 | `left_wrist_pitch_joint` |
| 19 | `left_wrist_yaw_joint` |
| 20 | `right_shoulder_pitch_joint` |
| 21 | `right_shoulder_roll_joint` |
| 22 | `right_shoulder_yaw_joint` |
| 23 | `right_elbow_joint` |
| 24 | `right_wrist_roll_joint` |
| 25 | `right_wrist_pitch_joint` |
| 26 | `right_wrist_yaw_joint` |

## g1_29dof_with_hand.urdf / g1_29dof_with_hand_rev_1_0.urdf (43 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `waist_roll_joint` |
| 14 | `waist_pitch_joint` |
| 15 | `left_shoulder_pitch_joint` |
| 16 | `left_shoulder_roll_joint` |
| 17 | `left_shoulder_yaw_joint` |
| 18 | `left_elbow_joint` |
| 19 | `left_wrist_roll_joint` |
| 20 | `left_wrist_pitch_joint` |
| 21 | `left_wrist_yaw_joint` |
| 22 | `left_hand_thumb_0_joint` |
| 23 | `left_hand_thumb_1_joint` |
| 24 | `left_hand_thumb_2_joint` |
| 25 | `left_hand_middle_0_joint` |
| 26 | `left_hand_middle_1_joint` |
| 27 | `left_hand_index_0_joint` |
| 28 | `left_hand_index_1_joint` |
| 29 | `right_shoulder_pitch_joint` |
| 30 | `right_shoulder_roll_joint` |
| 31 | `right_shoulder_yaw_joint` |
| 32 | `right_elbow_joint` |
| 33 | `right_wrist_roll_joint` |
| 34 | `right_wrist_pitch_joint` |
| 35 | `right_wrist_yaw_joint` |
| 36 | `right_hand_thumb_0_joint` |
| 37 | `right_hand_thumb_1_joint` |
| 38 | `right_hand_thumb_2_joint` |
| 39 | `right_hand_middle_0_joint` |
| 40 | `right_hand_middle_1_joint` |
| 41 | `right_hand_index_0_joint` |
| 42 | `right_hand_index_1_joint` |

## g1_dual_arm.urdf (14 DOFs)

| idx | joint |
|---|---|
| 0 | `left_shoulder_pitch_joint` |
| 1 | `left_shoulder_roll_joint` |
| 2 | `left_shoulder_yaw_joint` |
| 3 | `left_elbow_joint` |
| 4 | `left_wrist_roll_joint` |
| 5 | `left_wrist_pitch_joint` |
| 6 | `left_wrist_yaw_joint` |
| 7 | `right_shoulder_pitch_joint` |
| 8 | `right_shoulder_roll_joint` |
| 9 | `right_shoulder_yaw_joint` |
| 10 | `right_elbow_joint` |
| 11 | `right_wrist_roll_joint` |
| 12 | `right_wrist_pitch_joint` |
| 13 | `right_wrist_yaw_joint` |

## go2.urdf (12 DOFs)

| idx | joint |
|---|---|
| 0 | `FL_hip_joint` |
| 1 | `FL_thigh_joint` |
| 2 | `FL_calf_joint` |
| 3 | `FR_hip_joint` |
| 4 | `FR_thigh_joint` |
| 5 | `FR_calf_joint` |
| 6 | `RL_hip_joint` |
| 7 | `RL_thigh_joint` |
| 8 | `RL_calf_joint` |
| 9 | `RR_hip_joint` |
| 10 | `RR_thigh_joint` |
| 11 | `RR_calf_joint` |

## h1.urdf (10 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_yaw_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_pitch_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_joint` |
| 5 | `right_hip_yaw_joint` |
| 6 | `right_hip_roll_joint` |
| 7 | `right_hip_pitch_joint` |
| 8 | `right_knee_joint` |
| 9 | `right_ankle_joint` |

## h1_2.urdf (51 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_yaw_joint` |
| 1 | `left_hip_pitch_joint` |
| 2 | `left_hip_roll_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_yaw_joint` |
| 7 | `right_hip_pitch_joint` |
| 8 | `right_hip_roll_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `torso_joint` |
| 13 | `left_shoulder_pitch_joint` |
| 14 | `left_shoulder_roll_joint` |
| 15 | `left_shoulder_yaw_joint` |
| 16 | `left_elbow_pitch_joint` |
| 17 | `left_elbow_roll_joint` |
| 18 | `left_wrist_pitch_joint` |
| 19 | `left_wrist_yaw_joint` |
| 20 | `right_shoulder_pitch_joint` |
| 21 | `right_shoulder_roll_joint` |
| 22 | `right_shoulder_yaw_joint` |
| 23 | `right_elbow_pitch_joint` |
| 24 | `right_elbow_roll_joint` |
| 25 | `right_wrist_pitch_joint` |
| 26 | `right_wrist_yaw_joint` |
| 27 | `L_thumb_proximal_yaw_joint` |
| 28 | `L_thumb_proximal_pitch_joint` |
| 29 | `L_thumb_intermediate_joint` |
| 30 | `L_thumb_distal_joint` |
| 31 | `L_index_proximal_joint` |
| 32 | `L_index_intermediate_joint` |
| 33 | `L_middle_proximal_joint` |
| 34 | `L_middle_intermediate_joint` |
| 35 | `L_ring_proximal_joint` |
| 36 | `L_ring_intermediate_joint` |
| 37 | `L_pinky_proximal_joint` |
| 38 | `L_pinky_intermediate_joint` |
| 39 | `R_thumb_proximal_yaw_joint` |
| 40 | `R_thumb_proximal_pitch_joint` |
| 41 | `R_thumb_intermediate_joint` |
| 42 | `R_thumb_distal_joint` |
| 43 | `R_index_proximal_joint` |
| 44 | `R_index_intermediate_joint` |
| 45 | `R_middle_proximal_joint` |
| 46 | `R_middle_intermediate_joint` |
| 47 | `R_ring_proximal_joint` |
| 48 | `R_ring_intermediate_joint` |
| 49 | `R_pinky_proximal_joint` |
| 50 | `R_pinky_intermediate_joint` |

## h1_2_12dof.urdf (12 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_yaw_joint` |
| 1 | `left_hip_pitch_joint` |
| 2 | `left_hip_roll_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_yaw_joint` |
| 7 | `right_hip_pitch_joint` |
| 8 | `right_hip_roll_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |

## h1_2_simplified.urdf (21 DOFs)

| idx | joint |
|---|---|
| 0 | `left_hip_yaw_joint` |
| 1 | `left_hip_pitch_joint` |
| 2 | `left_hip_roll_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_yaw_joint` |
| 7 | `right_hip_pitch_joint` |
| 8 | `right_hip_roll_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `torso_joint` |
| 13 | `left_shoulder_pitch_joint` |
| 14 | `left_shoulder_roll_joint` |
| 15 | `left_shoulder_yaw_joint` |
| 16 | `left_elbow_pitch_joint` |
| 17 | `right_shoulder_pitch_joint` |
| 18 | `right_shoulder_roll_joint` |
| 19 | `right_shoulder_yaw_joint` |
| 20 | `right_elbow_pitch_joint` |
