"""Robot-only reachability tolerances; no scene or contact observations."""
import numpy as np
from scipy.spatial.transform import Rotation

from ..execution_limits import EEF_MAX_STEPS

POSITION_TOLERANCE_M = 0.002
ROTATION_TOLERANCE_RAD = 0.02
JOINT_STEP_LIMIT_RAD = 0.05
JOINT_NUMERICAL_TOLERANCE_RAD = 1e-6


def limits():
    return dict(position_tolerance_m=POSITION_TOLERANCE_M,
                rotation_tolerance_rad=ROTATION_TOLERANCE_RAD,
                joint_step_limit_rad=JOINT_STEP_LIMIT_RAD,
                joint_numerical_tolerance_rad=JOINT_NUMERICAL_TOLERANCE_RAD,
                max_total_steps=EEF_MAX_STEPS)


def checked_pose(position, quaternion):
    p, q = np.asarray(position), np.asarray(quaternion)
    if (p.shape != (3,) or q.shape != (4,) or
            p.dtype.kind not in 'iuf' or q.dtype.kind not in 'iuf'):
        raise ValueError('Finite numeric position[3] and quaternion_wxyz[4] required')
    p, q = p.astype(float), q.astype(float)
    if not np.isfinite(np.r_[p, q]).all() or abs(np.linalg.norm(q)-1) > 1e-4:
        raise ValueError('Finite position and unit wxyz quaternion required')
    return p, q


def checked_targets(targets):
    if not isinstance(targets, dict) or set(targets) != {'left', 'right'}:
        raise ValueError('Explicit left and right EEF targets required')
    result = {}
    for arm in ('left', 'right'):
        target = targets[arm]
        if not isinstance(target, dict):
            raise ValueError('EEF target must be an object')
        p, q = checked_pose(target.get('position'), target.get('quaternion_wxyz'))
        if type(target.get('gripper_closed')) is not bool:
            raise ValueError('Explicit boolean gripper_closed required')
        result[arm] = dict(position=p.tolist(), quaternion_wxyz=q.tolist(),
                           gripper_closed=target['gripper_closed'])
    return result


def pose_errors(position, quaternion, target):
    p, q = checked_pose(position, quaternion)
    goal_p, goal_q = checked_pose(target['position'], target['quaternion_wxyz'])
    distance = float(np.linalg.norm(goal_p-p))
    angle = float((Rotation.from_quat(goal_q[[1, 2, 3, 0]]) *
                   Rotation.from_quat(q[[1, 2, 3, 0]]).inv()).magnitude())
    return dict(position_error_m=distance, rotation_error_rad=angle,
                within_tolerance=bool(distance <= POSITION_TOLERANCE_M and
                                      angle <= ROTATION_TOLERANCE_RAD))
