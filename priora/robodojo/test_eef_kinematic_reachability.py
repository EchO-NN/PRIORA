"""CPU checks with the native X5 URDF; no Isaac, policy, API, or physics steps."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .robodojo_server.action_edit_kinematics import pose
from .robodojo_server.kinematics import DualKinematics
from .robodojo_server import reachability


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, item):
        return Tensor(self.array[item])

    def cpu(self):
        return self

    def numpy(self):
        return self.array.copy()


@pytest.fixture
def native_fk():
    explicit = os.environ.get('ROBODOJO_X5_URDF')
    urdf = (Path(explicit).expanduser() if explicit is not None else
            Path(__file__).resolve().parents[2]/'upstream/RoboDojo-current/Assets/Robots/x5/X5A.urdf')
    if not urdf.is_file():
        if explicit is not None:
            pytest.fail(f'ROBODOJO_X5_URDF does not identify an existing URDF file: {urdf}')
        pytest.skip('Native X5 URDF assets are not installed')
    manager = SimpleNamespace()
    manager.robot_list = [SimpleNamespace(type='target', arm_name=arm+'_arm',
        urdf_path=str(urdf), arm_joints_name=[f'joint{i}' for i in range(1, 7)],
        base_link='base_link', ee_link_name='link6', arm_joint_indices=list(range(6)))
        for arm in ('left', 'right')]
    manager.robot_key = [SimpleNamespace(data=SimpleNamespace(
        soft_joint_pos_limits=Tensor(np.array([[[-2., 2.]]*6])))) for _ in range(2)]
    manager.q = {arm: np.zeros(6) for arm in ('left', 'right')}
    manager.measured_override = {}
    manager.get_joint = lambda robot: {0: manager.q[robot.arm_name.split('_')[0]].copy()}
    manager.get_link_pose = lambda *args, **kwargs: {0: np.array([0., 0., 0., 1., 0., 0., 0.])}
    kin = DualKinematics(SimpleNamespace(robot_manager=manager))

    def measured(robot):
        arm = robot.arm_name.split('_')[0]
        if arm in manager.measured_override:
            return {0: manager.measured_override[arm].copy()}
        value = pose(kin.fk[arm].matrix(manager.q[arm]))
        return {0: np.r_[value['position'], value['quaternion_wxyz']]}

    manager.get_real_endpose = measured
    return kin, manager


def target_for(kin, q=None):
    q = np.zeros(6) if q is None else q
    return {arm: dict(pose(kin.fk[arm].matrix(q)), gripper_closed=False)
            for arm in ('left', 'right')}


def test_real_urdf_stages_follow_original_dls_and_do_not_mutate_measured_state(native_fk):
    kin, manager = native_fk
    initial = copy.deepcopy(manager.q)
    first = target_for(kin, np.array([.01, .01, 0., 0., 0., 0.]))
    second = target_for(kin, np.array([.01, .01, .01, 0., 0., 0.]))
    stages = [dict(label='approach', steps=12, target=first),
              dict(label='lift', steps=18, target=second)]
    report = kin.check_waypoints(stages, remaining_steps=30)
    assert report['accepted'] and report['physical_steps'] == 0
    assert report['schema'] == 'eef_reachability_v1'
    assert not report['collision_checked'] and not report['physical_tracking_guaranteed']
    for arm in ('left', 'right'):
        np.testing.assert_array_equal(manager.q[arm], initial[arm])
        expected = initial[arm].copy()
        for node, node_report in zip(stages, report['waypoints']):
            for _ in range(node['steps']):
                expected, _ = kin.fk[arm].bounded_target(
                    expected, np.array([[-2., 2.]]*6), np.eye(4), node['target'][arm])
            np.testing.assert_array_equal(node_report['arms'][arm]['predicted_joint_positions'], expected)
            assert node_report['arms'][arm]['joint_limits_ok']
            assert node_report['arms'][arm]['joint_step_limits_ok']
            assert node_report['arms'][arm]['maximum_joint_step_rad'] <= .05+1e-6


def test_early_node_failure_cannot_be_hidden_by_final_node_convergence(native_fk):
    kin, _ = native_fk
    far = target_for(kin)
    far['left']['position'][0] += .045
    nodes = [dict(label='too_fast', steps=1, target=far),
             dict(label='later', steps=29, target=copy.deepcopy(far))]
    report = kin.check_waypoints(nodes)
    assert not report['accepted'] and not report['waypoints'][0]['passed']
    assert report['waypoints'][0]['arms']['right']['within_tolerance']
    assert report['reason'] == 'not_verified_within_requested_steps_or_local_ik'
    assert report['physical_steps'] == 0


def test_rotation_residual_and_soft_joint_limits_are_checked(native_fk):
    kin, manager = native_fk
    target = target_for(kin)
    q = np.asarray(target['left']['quaternion_wxyz'])
    rotated = Rotation.from_rotvec([.3, 0, 0])*Rotation.from_quat(q[[1, 2, 3, 0]])
    target['left']['quaternion_wxyz'] = rotated.as_quat()[[3, 0, 1, 2]].tolist()
    report = kin.check_waypoints([dict(label='rotate', steps=1, target=target)])
    assert not report['accepted']
    assert report['waypoints'][0]['arms']['left']['rotation_error_rad'] > .02
    for key in manager.robot_key:
        key.data.soft_joint_pos_limits = Tensor(np.zeros((1, 6, 2)))
    report = kin.check_waypoints([dict(label='limited', steps=30, target=target)])
    assert not report['accepted']
    assert all(x['joint_limits_ok'] for x in report['waypoints'][0]['arms'].values())


def test_remaining_budget_is_separate_from_kinematic_reachability(native_fk):
    kin, _ = native_fk
    nodes = [dict(label='hold', steps=30, target=target_for(kin))]
    report = kin.check_waypoints(nodes, remaining_steps=29)
    assert not report['accepted'] and report['waypoints'][0]['passed']
    assert report['reason'] == 'insufficient_remaining_native_steps'
    assert not kin.check_waypoints(nodes, remaining_steps=0)['accepted']


@pytest.mark.parametrize('mutate', [
    lambda nodes: nodes[0].update(steps=True),
    lambda nodes: nodes[0].update(steps=0),
    lambda nodes: nodes[0].update(steps=31),
    lambda nodes: nodes[0].update(label=' '),
    lambda nodes: nodes[0]['target'].pop('right'),
    lambda nodes: nodes[0]['target']['left'].update(position=[float('nan'), 0, 0]),
    lambda nodes: nodes[0]['target']['left'].update(quaternion_wxyz=[2, 0, 0, 0]),
    lambda nodes: nodes[0]['target']['left'].update(gripper_closed=1),
])
def test_invalid_waypoint_inputs_raise_before_any_prediction(native_fk, mutate):
    kin, _ = native_fk
    nodes = [dict(label='hold', steps=1, target=target_for(kin))]
    mutate(nodes)
    kin.check = lambda: pytest.fail('Malformed schedule reached measured FK check')
    with pytest.raises(ValueError):
        kin.check_waypoints(nodes)


def test_invalid_actual_joint_state_and_remaining_budget_raise(native_fk):
    kin, manager = native_fk
    nodes = [dict(label='hold', steps=1, target=target_for(kin))]
    for remaining in (-1, True, 1.5, '1'):
        with pytest.raises(ValueError):
            kin.check_waypoints(nodes, remaining_steps=remaining)
    manager.q['left'][0] = 2.1
    with pytest.raises(ValueError, match='outside their soft limits'):
        kin.check_waypoints(nodes)


def test_arrival_uses_real_endpose_even_when_fk_would_disagree(native_fk):
    kin, manager = native_fk
    targets = target_for(kin)
    for robot in manager.robot_list:
        arm = robot.arm_name.split('_')[0]
        manager.measured_override[arm] = manager.get_real_endpose(robot)[0]
    manager.measured_override['left'][0] += .003
    for fk in kin.fk.values():
        fk.matrix = lambda q: pytest.fail('Arrival guard must not substitute predicted FK')
    report = kin.check_targets(targets)
    assert report['schema'] == 'eef_target_check_v1'
    assert not report['passed'] and report['physical_steps'] == 0
    assert report['arms']['left']['position_error_m'] == pytest.approx(.003)
    assert report['arms']['right']['within_tolerance']
    manager.measured_override['left'][0] -= .003
    manager.measured_override['left'][3:] *= -1  # Equivalent quaternion sign.
    assert kin.check_targets(targets)['passed']


def test_tolerance_boundary_uses_both_position_and_orientation():
    target = dict(position=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.])
    assert reachability.pose_errors([.002, 0, 0], [1, 0, 0, 0], target)['within_tolerance']
    assert not reachability.pose_errors([.0020001, 0, 0], [1, 0, 0, 0], target)['within_tolerance']
    q = Rotation.from_rotvec([.020001, 0, 0]).as_quat()[[3, 0, 1, 2]]
    assert not reachability.pose_errors([0, 0, 0], q, target)['within_tolerance']
