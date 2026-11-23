"""Fail closed on incomplete robot-only reachability/arrival RPC reports."""
import math

from .reachability import POSITION_TOLERANCE_M, ROTATION_TOLERANCE_RAD, limits as required_limits


def _require(value, message):
    if not value:
        raise RuntimeError('Invalid EEF verification report: '+message)


def _arms(report, *, kinematic=False, elbow_path_checked=False, curobo=False):
    arms = report.get('arms')
    _require(isinstance(arms, dict) and set(arms) == {'left', 'right'}, 'both arms required')
    passed = []
    for arm, result in arms.items():
        _require(isinstance(result, dict), arm+' result must be an object')
        good = True
        for name, tolerance in (('position_error_m', POSITION_TOLERANCE_M),
                                ('rotation_error_rad', ROTATION_TOLERANCE_RAD)):
            value = result.get(name)
            _require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
                     arm+' '+name+' must be finite and nonnegative')
            good = good and value <= tolerance
        _require(type(result.get('within_tolerance')) is bool and
                 result['within_tolerance'] == good, arm+' tolerance flag mismatch')
        if kinematic:
            for field in ('joint_limits_ok', 'joint_step_limits_ok'):
                _require(type(result.get(field)) is bool, arm+' '+field+' required')
                good = good and result[field]
            # The simulator includes elbow-path rejection in waypoint.passed.
            # A valid rejection must be returned to the planner, not mistaken
            # for a contradictory RPC result. Never weaken the path guard.
            if elbow_path_checked or 'elbow_path_ok' in result:
                _require(type(result.get('elbow_path_ok')) is bool,
                         arm+' elbow_path_ok required')
                good = good and result['elbow_path_ok']
            if curobo:
                for field in ('ik_solution_ok', 'collision_path_ok'):
                    _require(type(result.get(field)) is bool, arm+' '+field+' required')
                    good = good and result[field]
        passed.append(good)
    return all(passed)


def verify_report(report, *, nodes=None, expected_remaining_steps=None,
                  expected_limits=None, expected_solver=None):
    """Validate completeness before trusting acceptance or advancing a stage."""
    _require(isinstance(report, dict), 'object required')
    preflight = nodes is not None
    expected = 'eef_reachability_v1' if preflight else 'eef_target_check_v1'
    _require(report.get('schema') == expected, 'schema mismatch')
    _require(type(report.get('physical_steps')) is int and report['physical_steps'] == 0,
             'check must not advance physics')
    flag = 'accepted' if preflight else 'passed'
    _require(type(report.get(flag)) is bool, flag+' must be a boolean')
    limits = report.get('limits')
    _require(isinstance(limits, dict), 'limits required')
    if expected_solver is not None:
        _require(report.get('solver') == expected_solver, 'solver mismatch')
    for key, value in (required_limits() if expected_limits is None else expected_limits).items():
        _require(type(limits.get(key)) is type(value) and limits[key] == value,
                 'unexpected '+key)
    if not preflight:
        _require(report['passed'] == _arms(report), 'arrival flag mismatch')
        return report
    _require(type(expected_remaining_steps) is int and expected_remaining_steps >= 0,
             'current native remaining-step budget required')
    total = sum(node['steps'] for node in nodes)
    _require(type(report.get('total_steps')) is int and report['total_steps'] == total,
             'total step count mismatch')
    _require(type(report.get('remaining_steps')) is int and
             report['remaining_steps'] == expected_remaining_steps,
             'remaining-step budget differs from the current observation')
    budget_ok = total <= expected_remaining_steps
    _require(type(report.get('remaining_steps_sufficient')) is bool and
             report['remaining_steps_sufficient'] == budget_ok,
             'remaining-step sufficiency flag mismatch')
    checks = report.get('waypoints')
    _require(isinstance(checks, list) and len(checks) == len(nodes), 'all waypoint checks required')
    elbow_path_checked = report.get('elbow_path_checked', False)
    _require(type(elbow_path_checked) is bool, 'elbow_path_checked must be a boolean')
    for index, (node, check) in enumerate(zip(nodes, checks)):
        _require(isinstance(check, dict) and type(check.get('index')) is int and check['index'] == index and
                 check.get('label') == node['label'] and
                 type(check.get('requested_steps')) is int and
                 check['requested_steps'] == node['steps'], 'waypoint identity mismatch')
        _require(type(check.get('passed')) is bool and
                 check['passed'] == _arms(check, kinematic=True,
                     elbow_path_checked=elbow_path_checked,
                     curobo=expected_solver == 'wall_shared_curobo_v1'), 'waypoint result mismatch')
    _require(report['accepted'] == (budget_ok and all(check['passed'] for check in checks)),
             'acceptance disagrees with waypoint checks or remaining-step budget')
    return report
