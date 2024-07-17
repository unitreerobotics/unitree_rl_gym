# H1_2 RL Example (Preview)

## Simplified URDF

This task utilizes a simplified version of URDF. We fix some joints and ignore most of the collisions.

### Fixed Joints

We fix all the joints in the hands, wrist and the elbow roll joints, since those joints have very limited effect on the whole body dynamics and are commonly controlled by other controllers.

### Collision

We only keep the collision of foot roll links, knee links and base. Early termination is majorly check by angular position of the base.

## Dynamics

Free "light" end effectors can lead to unstable simulation. Thus please be carefull with the control parameters for the joints that may affect such end effectors.

## Preview Stage

**The reward functions is not well tuned and cannot produce satisfactory results at the current stage. A feasible version is comming soon.**