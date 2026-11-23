"""Deterministic EEF easing and ACK-owned joint command filtering.

This smooths commands, not measured contact dynamics. Collision validation must
run AFTER the joint filter; a rejected action always takes priority over easing.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np

VERSION = 'wall_eef_joint_smoothing_v1'
JOINT_ALPHA = 0.5
ARMS = ('left', 'right')


def metadata():
    return dict(version=VERSION, eef_position='quintic_minimum_jerk',
        eef_rotation='shortest_arc_slerp_with_quintic_time',
        extra_stages=0, extra_controls=0,
        joint_filter='acknowledged_command_ema', joint_alpha=JOINT_ALPHA,
        gripper_filter=False, final_commands_rechecked=True,
        acceleration_bound_guaranteed=False, full_scene_collision_checked=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def verified_schedule(nodes, detail):
    """Execute exactly the simulator's preflight targets, including its float rounding."""
    expected = eef_schedule(nodes, detail['start_poses'])
    transmitted = 'eef_schedule' in detail
    schedule = detail['eef_schedule'] if transmitted else expected
    if digest(schedule) != detail.get('schedule_sha256'):
        raise RuntimeError('EEF smoothing schedule digest mismatch')
    if not transmitted or digest(expected) == detail['schedule_sha256']:
        return schedule
    # Older simulators send only the digest and must match exactly. For a
    # transmitted trajectory independently check each value against the plan.
    if type(schedule) is not list or len(schedule) != len(expected):
        raise RuntimeError('EEF smoothing schedule step count mismatch')
    for step, reference in zip(schedule, expected):
        if not isinstance(step, dict) or step.keys() != reference.keys():
            raise RuntimeError('EEF smoothing schedule arm mismatch')
        for arm, ref in reference.items():
            pose = step[arm]
            if not isinstance(pose, dict) or pose.keys() != ref.keys():
                raise RuntimeError('EEF smoothing schedule target fields mismatch')
            if pose['gripper_closed'] is not ref['gripper_closed']:
                raise RuntimeError('EEF smoothing schedule gripper mismatch')
            for key in ('position', 'quaternion_wxyz'):
                values = pose[key]
                if (type(values) is not list or len(values) != len(ref[key])
                        or not all(type(v) in (int, float) for v in values)
                        or not np.allclose(values, ref[key], rtol=0, atol=1e-14)):
                    raise RuntimeError('EEF smoothing schedule differs from requested ' + key)
    return schedule


def slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        raise ValueError('Finite wxyz quaternions required')
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        raise ValueError('Zero quaternion')
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        q = (1-t)*a+t*b
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        q = (np.sin((1-t)*angle)*a+np.sin(t*angle)*b)/np.sin(angle)
    return (q/np.linalg.norm(q)).tolist()


def eef_schedule(nodes, start_poses):
    """Same exact target schedule is used by preflight and the real client.

    Ease out of the starting pose and into each existing waypoint. No new
    stage, dwell, or control step is inserted. The gripper
    command takes effect at that waypoint's FIRST control, as before.
    """
    previous = deepcopy(start_poses)
    result = []
    for node in nodes:
        count = node['steps']
        if type(count) is not int or count < 1:
            raise ValueError('Positive integer waypoint duration required')
        for i in range(count):
            t = (i+1)/count
            u = t*t*t*(10+t*(-15+6*t))
            target = deepcopy(node['target'])
            for arm in ARMS:
                start, end = previous[arm], node['target'][arm]
                if t < 1:
                    target[arm]['position'] = (
                        (1-u)*np.asarray(start['position'], float)+
                        u*np.asarray(end['position'], float)).tolist()
                    target[arm]['quaternion_wxyz'] = slerp(
                        start['quaternion_wxyz'], end['quaternion_wxyz'], u)
            result.append(target)
        previous = deepcopy(node['target'])
    return result


def filter_joint_command(current, desired, last_command, max_step, bounds):
    """EMA on commanded positions, bounded relative to measured joints.

    No new velocity/acceleration or physical tracking guarantee is asserted.
    The previous command is updated only after a simulator ACK, never on a
    solver call or during preflight. Clip to URDF limits before collision check.
    """
    current, desired = np.asarray(current, float), np.asarray(desired, float)
    previous = current if last_command is None else np.asarray(last_command, float)
    bounds = np.asarray(bounds, float)
    if (current.shape != desired.shape or previous.shape != current.shape or
            bounds.shape != (len(current), 2) or
            not np.isfinite(np.r_[current, desired, previous, bounds.ravel()]).all() or
            not np.isfinite(max_step) or max_step <= 0):
        raise ValueError('Invalid joint filter input')
    low = np.maximum(bounds[:, 0], current-max_step)
    high = np.minimum(bounds[:, 1], current+max_step)
    if np.any(low > high):
        raise ValueError('Measured joints outside reachable bounds')
    filtered = np.clip(previous+JOINT_ALPHA*(desired-previous), low, high)
    return filtered, dict(version=VERSION, alpha=JOINT_ALPHA,
        raw_joint_target=desired.tolist(), filtered_joint_target=filtered.tolist(),
        maximum_joint_step_rad=float(np.max(np.abs(filtered-current))))
