"""Hysteretic gripper postprocessing; only acknowledged actions commit state."""
import os

import numpy as np
from ..robodojo_server.eef_waypoints import target_at_step


class GripperHold:
    def __init__(self, mode='continuous', close_threshold=.65, release_threshold=.90):
        if mode not in ('continuous', 'hold_closed'):
            raise ValueError(f'Unknown pi05 gripper mode: {mode}')
        if not 0 <= close_threshold < release_threshold <= 1:
            raise ValueError('Require 0 <= close threshold < release threshold <= 1')
        self.mode = mode
        self.close_threshold = close_threshold
        self.release_threshold = release_threshold
        self.held = [False, False]

    @classmethod
    def from_env(cls):
        return cls(os.getenv('ROLLOUT_PI05_GRIPPER_MODE', 'continuous'),
                   float(os.getenv('ROLLOUT_PI05_GRIPPER_CLOSE_THRESHOLD', '.65')),
                   float(os.getenv('ROLLOUT_PI05_GRIPPER_RELEASE_THRESHOLD', '.90')))

    def metadata(self):
        return dict(mode=self.mode, close_threshold=self.close_threshold,
                    release_threshold=self.release_threshold,
                    held=list(self.held), version='acknowledged_hysteresis_v1',
                    note=('Pi05 proposals below or equal to close_threshold latch full closure (0); '
                          'values above or equal to release_threshold release the latch. '
                          'Between thresholds retain the latch. FK and actions reflect this '
                          'postprocessing. Explicit teacher open/closed commands override it. '
                          'This is command-based holding, not measured grasp or slip detection.'
                          if self.mode == 'hold_closed' else 'Unmodified continuous pi05 openings.'))

    def preview(self, actions):
        result = np.asarray(actions).copy()
        if result.ndim != 2 or result.shape[1] != 14 or not np.isfinite(result).all():
            raise ValueError('Expected finite (N,14) actions')
        if np.any((result[:, [6, 13]] < 0) | (result[:, [6, 13]] > 1)):
            raise ValueError('Expected clipped openings in [0,1]')
        held = list(self.held)
        states = []
        for row in result:
            if self.mode == 'hold_closed':
                for arm, column in enumerate((6, 13)):
                    if row[column] >= self.release_threshold:
                        held[arm] = False
                    elif row[column] <= self.close_threshold:
                        held[arm] = True
                    if held[arm]:
                        row[column] = 0
            states.append(list(held))
        return result, np.asarray(states, dtype=bool).reshape(-1, 2)

    def commit(self, preview_states, executed_count, response):
        if self.mode != 'hold_closed' or executed_count == 0:
            return
        if not 0 < executed_count <= len(preview_states):
            raise ValueError('Invalid acknowledged prefix length')
        mode = response['mode']
        for arm, name in enumerate(('left', 'right')):
            if mode == 'student' or (mode == 'edit' and response['edit'][name]['gripper'] == 'keep'):
                self.held[arm] = bool(preview_states[executed_count-1, arm])
            elif mode == 'edit':
                self.held[arm] = response['edit'][name]['gripper'] == 'closed'
            elif mode == 'eef':
                # Native termination may stop before the final planned stage.
                # Commit only the command at the last acknowledged control.
                target = (target_at_step(response, executed_count-1)
                          if response.get('waypoints') else response['target'])
                self.held[arm] = bool(target[name]['gripper_closed'])
            else:
                raise ValueError(f'Cannot commit mode {mode}')
