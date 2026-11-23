"""Copied action/gate validation from colleague baseline 5ca5512."""
import numpy as np
from scipy.spatial.transform import Rotation
from .action_edit_kinematics import edited_targets
from .gate_assessment import validate_assessment
from ..execution_limits import HYBRID_STEP_LIMITS, EEF_MAX_STEPS
from .active_view import reject_active_view

def angles(p):
    q = np.asarray(p['quaternion_wxyz'])
    return dict(p, rpy_deg=Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler('xyz', degrees=True).tolist())


def _contains_cjk(text):
    return any('\u3400' <= char <= '\u9fff' or '\U00020000' <= char <= '\U000323af'
               for char in text)


def validate_public_language(response):
    """Reject CJK in model-authored public prose; protocol/task text is untouched."""
    assessment = response.get('assessment', {})
    progress = assessment.get('task_progress', {}) if isinstance(assessment, dict) else {}
    values = [response.get('reason'), *(assessment.get(key) for key in (
        'current_subgoal', 'execution_evidence', 'expected_next_intent',
        'predicted_next_intent', 'intent_evidence')),
        progress.get('currently_attempting'), *progress.get('verified_completed', []),
        *progress.get('remaining', [])]
    waypoints = response.get('waypoints', [])
    if isinstance(waypoints, list):
        values.extend(node.get('label') for node in waypoints if isinstance(node, dict))
    if any(isinstance(value, str) and _contains_cjk(value) for value in values):
        raise ValueError('Public decision and assessment text must be written in English')


def validate_response(response, request, *, eef_step_limit=EEF_MAX_STEPS):
    reject_active_view(response)
    if response.get('request_id') != request['request_id']:
        raise ValueError('Response is stale or belongs to another observation')
    if not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('A visible observation/decision explanation is required')
    mode = response.get('mode')
    if mode not in ('student', 'edit', 'eef', 'stop'):
        raise ValueError('Expected student/edit/eef/stop')
    waypoints = response.get('waypoints', [])
    if not isinstance(waypoints, list):
        raise ValueError('waypoints must be an array; use [] for a single target')
    if waypoints and (mode != 'eef' or request.get('gate_policy') == 'gpt_only'):
        raise ValueError('Nonempty waypoints are supported only by hybrid EEF actions')
    if request.get('gate_policy') == 'failure-or-intent':
        validate_assessment(response, request)
        validate_public_language(response)
    if mode == 'stop':
        return
    n = response.get('steps')
    cap = eef_step_limit if mode == 'eef' else HYBRID_STEP_LIMITS[mode]
    if type(n) is not int or not 1 <= n <= cap:
        raise ValueError(f'Expected 1..{cap} execution steps')
    if mode == 'edit':
        edited_targets(request['student_eef_trajectory'], n, response['edit'])
    elif mode == 'eef':
        target = response['target']
        p, q = np.asarray(target['position'], float), np.asarray(target['quaternion_wxyz'], float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError('Finite EEF position/quaternion required')
        if abs(np.linalg.norm(q)-1) > 1e-4 or type(target['gripper_closed']) is not bool:
            raise ValueError('Unit quaternion and explicit boolean gripper required')
        old = request['current_eef']['quaternion_wxyz']
        delta = Rotation.from_quat(q[[1, 2, 3, 0]]) * Rotation.from_quat(np.asarray(old)[[1, 2, 3, 0]]).inv()
        if delta.magnitude() > .35+1e-9:
            raise ValueError('Recovery EEF target exceeds 0.35 rad')
