"""Bounded, step-scheduled EEF stages; no contact-driven switching.

All indexes in expanded records are zero-based. Validation is performed before
any physical action, relative to both the measured initial EEF pose and each
preceding waypoint. The controller additionally requires numerical reachability
before execution and measured arrival at each stage boundary. Execution still
recomputes robot-only IK after every ACK; arrival does not shorten a stage.
"""
from copy import deepcopy

import numpy as np

from ..execution_limits import EEF_MAX_STEPS
from .validation import _contains_cjk, validate_response


ARMS = ('left', 'right')
TARGET_FIELDS = {'position', 'quaternion_wxyz', 'gripper_closed'}


def _step_count(value):
    if type(value) is not int or not 1 <= value <= EEF_MAX_STEPS:
        raise ValueError(f'EEF waypoint execution steps must be an integer in 1..{EEF_MAX_STEPS}')
    return value


def _dual_target(target, *, strict=False):
    if not isinstance(target, dict) or set(target) != set(ARMS):
        raise ValueError('Each EEF waypoint requires explicit left and right targets')
    for arm in ARMS:
        pose = target[arm]
        if not isinstance(pose, dict) or not TARGET_FIELDS <= set(pose):
            raise ValueError(f'Waypoint {arm} requires position, quaternion_wxyz and gripper_closed')
        if strict and set(pose) != TARGET_FIELDS:
            raise ValueError(f'Unexpected fields in waypoint {arm} target')
    return target


def _targets_equal(left, right):
    """Exact JSON target agreement, allowing equivalent list/tuple coordinates."""
    if set(left) != set(right):
        return False
    for arm in ARMS:
        if set(left[arm]) != set(right[arm]):
            return False
        for key in left[arm]:
            if key == 'gripper_closed' and (type(left[arm][key]) is not bool or
                                            type(right[arm][key]) is not bool):
                return False
            if not np.array_equal(left[arm][key], right[arm][key]):
                return False
    return True


def _scheduled_nodes(response):
    if not isinstance(response, dict):
        raise ValueError('EEF response must be an object')
    nodes = response.get('waypoints', [])
    if not isinstance(nodes, list):
        raise ValueError('waypoints must be an array; null is not supported')
    if response.get('mode') != 'eef':
        if nodes:
            raise ValueError('Nonempty waypoints require mode=eef')
        return []
    total = _step_count(response.get('steps'))
    target = _dual_target(response.get('target'))
    if not nodes:
        return [dict(label='target', steps=total, target=deepcopy(target))]
    if len(nodes) > EEF_MAX_STEPS:
        raise ValueError(f'At most {EEF_MAX_STEPS} EEF waypoints are allowed')
    result = []
    for node in nodes:
        if not isinstance(node, dict) or set(node) != {'label', 'steps', 'target'}:
            raise ValueError('Each waypoint requires exactly label, steps and target')
        label = node['label']
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise ValueError('Waypoint label must be a nonempty English string of at most 80 characters')
        if _contains_cjk(label):
            raise ValueError('Public waypoint labels must be written in English')
        result.append(dict(label=label, steps=_step_count(node['steps']),
                           target=deepcopy(_dual_target(node['target'], strict=True))))
    if sum(node['steps'] for node in result) != total:
        raise ValueError('EEF steps must equal the sum of waypoint steps')
    if not _targets_equal(target, result[-1]['target']):
        raise ValueError('EEF target must equal the final waypoint target')
    return result


def validated_waypoints(response, request):
    """Validate dual-arm stages and return independent node dictionaries.

    Legacy EEF (missing/empty ``waypoints``) returns one ``label='target'``
    node. Other modes require empty/missing waypoints and return []. The caller
    remains responsible for their usual student/edit/stop validation.
    """
    nodes = _scheduled_nodes(response)
    if not nodes:
        return []
    if request.get('gate_policy') == 'gpt_only' and response.get('waypoints'):
        raise ValueError('Waypoints are unavailable in GPT-only evaluation')
    measured = request.get('current_eef')
    if not isinstance(measured, dict) or set(measured) != set(ARMS):
        raise ValueError('Measured left and right EEF poses are required')
    previous = measured
    for index, node in enumerate(nodes):
        for arm in ARMS:
            local = dict(response, target=node['target'][arm], steps=node['steps'])
            for reference_name, reference in (('initial measured pose', measured),
                                               ('preceding waypoint', previous)):
                try:
                    validate_response(local, dict(request, current_eef=reference[arm]))
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(
                        f'Waypoint {index} ({node["label"]}), {arm}, '
                        f'relative to {reference_name}: {error}') from error
        previous = node['target']
    return nodes


def expanded_waypoints(response):
    """Expand a validated EEF schedule, with zero-based stage/step indexes.

    Checks structure and step accounting; geometric checks require the initial
    observation and must first use ``validated_waypoints(response, request)``.
    Returns [] for non-EEF modes with empty/missing waypoints.
    """
    return [dict(target=deepcopy(node['target']), waypoint_index=index,
                 waypoint_label=node['label'], waypoint_step=step)
            for index, node in enumerate(_scheduled_nodes(response))
            for step in range(node['steps'])]


def target_at_step(response, index):
    """Return the target for a zero-based executed EEF control-step index."""
    expanded = expanded_waypoints(response)
    if type(index) is not int or not 0 <= index < len(expanded):
        raise ValueError('EEF control-step index is outside the waypoint schedule')
    return expanded[index]['target']
