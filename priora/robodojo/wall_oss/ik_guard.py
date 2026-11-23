"""Continuous IK selection and an asset-derived X5 elbow exclusion.

This covers the link2/link3 mesh interference omitted by native cuRobo. It is
not a scene, inter-arm or complete self-collision planner. The angular profile
is generated offline from the original meshes, not hand-picked joint limits.
"""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

VERSION = 'wall_continuous_ik_elbow_v1'


@lru_cache(maxsize=8)
def load_profile(urdf):
    urdf = Path(urdf)
    profile = json.loads(Path(__file__).with_name('x5_elbow_profile.json').read_text())
    for relative, expected in profile['asset_sha256'].items():
        path = urdf if relative == 'X5A.urdf' else urdf.parent / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('IK elbow profile does not match robot asset: ' + str(path))
    return profile


class JointPathGuard:
    def __init__(self, urdf, names):
        self.profile = load_profile(str(urdf))
        joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
        self.names = list(names)
        self.index = self.names.index(self.profile['joint'])
        self.limits = np.asarray([[float(joints[n].find('limit').get(k))
                                  for k in ('lower', 'upper')] for n in names])

    def normalize(self, candidate, current):
        """Choose equivalent angles nearest measured joints, within URDF limits."""
        candidate, current = np.asarray(candidate, float), np.asarray(current, float)
        if candidate.shape != current.shape or candidate.shape != (len(self.names),):
            raise ValueError('IK candidate dimension mismatch')
        if not np.isfinite(candidate).all() or not np.isfinite(current).all():
            raise ValueError('Nonfinite IK candidate')
        result = candidate.copy()
        for i, (lo, hi) in enumerate(self.limits):
            k0 = math.ceil((lo-candidate[i])/(2*math.pi))
            k1 = math.floor((hi-candidate[i])/(2*math.pi))
            if k0 > k1:
                return None
            k = int(np.clip(round((current[i]-candidate[i])/(2*math.pi)), k0, k1))
            result[i] += k * 2*math.pi
        return result

    def check_path(self, current, candidate):
        current, candidate = np.asarray(current, float), np.asarray(candidate, float)
        if (current.shape != (len(self.names),) or candidate.shape != current.shape or
                not np.isfinite(np.r_[current, candidate]).all()):
            return dict(passed=False, reason='invalid_joint_command')
        if np.any(candidate < self.limits[:, 0]) or np.any(candidate > self.limits[:, 1]):
            return dict(passed=False, reason='joint_limit')
        lo, hi = sorted((current[self.index], candidate[self.index]))
        # Check the entire unwrapped interval, not only the endpoint or sparse
        # path samples. A 2-pi turn must not pass through the forbidden band.
        for left, right in self.profile['forbidden_intervals_rad']:
            k0 = math.ceil((lo-right)/(2*math.pi))
            k1 = math.floor((hi-left)/(2*math.pi))
            if k0 <= k1:
                return dict(passed=False, reason='upper_forearm_mesh_path_interference',
                            joint=self.profile['joint'], interval_rad=[float(lo), float(hi)])
        return dict(passed=True, reason='elbow_path_clear')

    def select(self, current, candidates, successes, max_step):
        """Prefer nearest valid branch; rate-limit along that same joint line."""
        current = np.asarray(current, float)
        if not np.isfinite(max_step) or max_step <= 0:
            raise ValueError('Invalid per-control joint limit')
        choices = []
        for candidate, success in zip(candidates, successes):
            if not success:
                continue
            q = self.normalize(candidate, current)
            if q is not None:
                # Nearest 2-pi representation can cross the elbow's forbidden
                # band while the farther representation stays on this branch.
                lo, hi = self.limits[self.index]
                for k in range(math.ceil((lo-q[self.index])/(2*math.pi)),
                               math.floor((hi-q[self.index])/(2*math.pi))+1):
                    equivalent = q.copy(); equivalent[self.index] += k*2*math.pi
                    if self.check_path(current, equivalent)['passed']:
                        choices.append(equivalent)
        if not choices:
            return current.copy(), dict(version=VERSION, blocked=True,
                reason='no_continuous_collision_clear_ik_solution',
                candidate_count=len(candidates), feasible_candidates=0)
        q = min(choices, key=lambda v: float(np.linalg.norm(v-current)))
        delta = q-current
        scale = min(1., max_step/max(float(np.abs(delta).max()), 1e-12))
        result = current + delta*scale
        return result, dict(version=VERSION, blocked=False,
            candidate_count=len(candidates), feasible_candidates=len(choices),
            rate_limited=scale < 1., rate_scale=scale,
            maximum_joint_step_rad=max_step,
            selected_solution_max_delta_rad=float(np.abs(delta).max()))
