"""No physics/model calls: exercise proposal, ACK, handoff and audit semantics."""
import json

import numpy as np
import pytest

from .pi05_server.gripper_hold import GripperHold
from .robodojo_server.client import RoboDojoTools
from .test_support import Sim, Student, invoke, response


def actions(left, right=None):
    a = np.arange(len(left)*14, dtype=np.float32).reshape(-1, 14)/100
    a[:, 6] = left
    a[:, 13] = 1 if right is None else right
    return a


def test_hysteresis_independent_arms_and_no_other_joint_changes():
    hold = GripperHold('hold_closed')
    raw = actions([1, .8, .6, .7, .89, .95], [1, 1, 1, .4, .8, 1])
    result, states = hold.preview(raw)
    np.testing.assert_allclose(result[:, 6], [1, .8, 0, 0, 0, .95])
    np.testing.assert_allclose(result[:, 13], [1, 1, 1, 0, 0, 1])
    joint_cols = [i for i in range(14) if i not in (6, 13)]
    np.testing.assert_array_equal(result[:, joint_cols], raw[:, joint_cols])
    assert hold.held == [False, False]  # Preview is not execution.
    np.testing.assert_array_equal(states[-1], [False, False])


def test_unexecuted_release_does_not_clear_latch_and_cross_chunk_hold():
    hold = GripperHold('hold_closed')
    _, states = hold.preview(actions([.6, .7, 1]))
    hold.commit(states, 1, dict(mode='student'))
    assert hold.held == [True, False]
    result, states = hold.preview(actions([.8, .89, .95]))
    np.testing.assert_allclose(result[:, 6], [0, 0, .95])
    hold.commit(states, 3, dict(mode='student'))
    assert hold.held == [False, False]


def test_discarded_future_closure_and_zero_ack_do_not_latch():
    hold = GripperHold('hold_closed')
    _, states = hold.preview(actions([1, .5]))
    hold.commit(states, 0, dict(mode='student'))
    hold.commit(states, 1, dict(mode='student'))
    assert hold.held == [False, False]


def test_teacher_overrides_and_keep_follow_acknowledged_prefix():
    hold = GripperHold('hold_closed')
    _, states = hold.preview(actions([.4, 1], [.7, 1]))
    hold.commit(states, 1, dict(mode='edit', edit=dict(left=dict(gripper='keep'), right=dict(gripper='closed'))))
    assert hold.held == [True, True]
    hold.commit(states, 1, dict(mode='eef', target=dict(left=dict(gripper_closed=False), right=dict(gripper_closed=True))))
    assert hold.held == [False, True]
    result, _ = hold.preview(actions([.8], [.8]))
    np.testing.assert_allclose(result[:, [6, 13]], [[.8, 0]])


def test_continuous_mode_is_identical():
    hold = GripperHold('continuous')
    raw = actions([.4, .7, 1])
    result, states = hold.preview(raw)
    np.testing.assert_array_equal(result, raw)
    hold.commit(states, 2, dict(mode='student'))
    assert hold.held == [False, False]


@pytest.mark.parametrize('close,release', [(.9, .65), (.5, .5), (-.1, .9), (.5, 1.1), (float('nan'), .9)])
def test_invalid_thresholds(close, release):
    with pytest.raises(ValueError):
        GripperHold('hold_closed', close, release)


def test_real_tools_fk_execution_and_artifact_agree(tmp_path, monkeypatch):
    monkeypatch.setenv('ROLLOUT_PI05_GRIPPER_MODE', 'hold_closed')

    class PreviewSim(Sim):
        def request(self, op, **kwargs):
            result = super().request(op, **kwargs)
            if op == 'fk_preview':
                self.preview_actions = kwargs['actions'].copy()
                for row, action in zip(result['trajectory'], self.preview_actions):
                    for name, col in [('left', 6), ('right', 13)]:
                        row[name]['gripper_opening'] = float(action[col])
                        row[name]['gripper_closed'] = bool(action[col] < .5)
            if op == 'chunk_step':
                self.executed_actions = kwargs['actions'].copy()
            return result

    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=PreviewSim)
    invoke(tools)
    packet = invoke(tools)
    assert packet['pi05_gripper_policy']['mode'] == 'hold_closed'
    assert tools.gripper_hold.held == [False, False]
    np.testing.assert_array_equal(tools.sim.preview_actions[:, [6, 13]], 0)
    with np.load(tmp_path/'proposal_000.npz') as stored:
        np.testing.assert_allclose(stored['raw_actions'][:, [6, 13]], .3)
        np.testing.assert_allclose(stored['pi05_actions'][:, [6, 13]], .3)
        np.testing.assert_array_equal(stored['actions'][:, [6, 13]], 0)
    invoke(tools, response=response(tools.request))
    np.testing.assert_array_equal(tools.sim.executed_actions[:, [6, 13]], 0)
    assert tools.gripper_hold.held == [True, True]
    history = json.loads((tmp_path/'history.json').read_text())
    assert history[0]['pi05_gripper_policy']['held'] == [True, True]
    assert json.loads((tmp_path/'run.json').read_text())['pi05_gripper_policy']['mode'] == 'hold_closed'
