# H1_2 RL Example (Preview)

## Simplified URDF

This task utilizes a simplified version of URDF. We fix some joints and ignore most of the collisions.

### Fixed Joints

We fix all the joints in the hands, wrists and the elbow roll joints, since those joints have very limited effect on the whole body dynamics and are commonly controlled by other controllers.

### Collisions

We only keep the collision of foot roll links, knee links and the base. Early termination is majorly checked by the angular position of the base.

## Dynamics

Free "light" end effectors can lead to unstable simulation. Thus please be carefull with the control parameters for the joints that may affect such end effectors.

## Results

https://github.com/user-attachments/assets/d6cdee70-8f8a-4a50-b219-df31b269b083

## Preview Stage

**The reward functions are not well tuned and cannot produce satisfactory results at the current stage. A feasible version is comming soon.**
