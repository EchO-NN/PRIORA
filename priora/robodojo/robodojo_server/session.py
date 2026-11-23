"""One RoboDojo episode, native action stepping and success/timeout checks."""
import copy
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import uuid
import numpy as np
from PIL import Image

from ..io import write_json, require, sha256
from ..robodojo_server.decision_log import record_decision
from .kinematics import DualKinematics
from ..rgbd import main_camera_input, measured_depth, depth_cameras, depth_key, depth_range
from ..execution_limits import STUDENT_MAX_STEPS
from ..grasp_feedback import capture as capture_grasp_feedback, VERSION as GRASP_VERSION
from ..teacher_history import EpisodeImageArchive, VERSION as HISTORY_VERSION
from ..camera_intrinsics import capture as capture_intrinsics, VERSION as INTRINSICS_VERSION
from ..camera_geometry import (from_environment_config, capture as capture_geometry,
                               VERSION as GEOMETRY_VERSION)

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def register_native_evaluation(env):
    """Mirror EvalEnv.run_eval's setup, without entering its policy loop.

    Native reset clears the checks. With no registered conditions the native
    RewardManager reports success immediately, so fail closed on that state.
    These remain simulator-only predicates, never teacher planning inputs.
    """
    env.run_reward()
    if hasattr(env, 'get_score'):
        env.get_score()
    names = ('check_list', 'final_check_list', 'trigger_check_list')
    counts = {name: [len(group) for group in getattr(env.reward_manager, name)] for name in names}
    require(all(sum(counts[name][i] for name in names) > 0 for i in range(env.num_envs)),
            'Native task completion conditions are empty; refusing vacuous success')
    support_envs = []
    if getattr(env, 'interact', False) and hasattr(env, 'query_support_arm_traj'):
        support_envs = list(env.get_running_env_idx_list())
        for env_idx in support_envs:
            env.query_support_arm_traj(env_idx=env_idx)
    return dict(registered=True, source='native EvalEnv.run_eval preamble',
                condition_group_counts=counts, initial_support_arm_query_envs=support_envs)


class RoboDojoSession:
    def __init__(self, env, output, task, evaluation_identity=None):
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode_id, self.step_id = None, 0
        self.terminated = self.truncated = self.success = False
        self.poisoned, self.finished = True, False
        self.video_writer, self.video_frames = None, 0
        self.source, self.control_epoch = 'student', 0
        self.evaluation_identity = evaluation_identity
        self.main_camera_input = main_camera_input()
        self.finish_reason = None
        self.metadata = dict(task=task, instruction=task, simulator='RoboDojo',
            robot_adapter='dual_arx_x5', control_dt=1/env.obs_manager.collect_freq,
            max_episode_steps=int(env.step_lim), action_dim=14, action_horizon=50,
            frame='environment_origin', eef_link='link6', no_rollback=True,
            supports_fk_preview=True, supports_codex_decision_recording=True,
            supports_eef_waypoint_reachability=True,
            provides_camera_intrinsics=True, camera_intrinsics_version=INTRINSICS_VERSION,
            provides_camera_geometry=True, camera_geometry_version=GEOMETRY_VERSION,
            provides_measured_grasp_feedback=True, grasp_feedback_version=GRASP_VERSION,
            provides_teacher_episode_history=True, teacher_history_version=HISTORY_VERSION,
            teacher_history_hz=2, teacher_history_scope='whole_episode',
            cameras=list(CAMERAS), gripper_semantics='continuous_0_closed_1_open')
        # Our recorder owns exactly one frame per actual control ACK. Disable
        # only upstream duplicate video capture, not its native task checks.
        env._stream_vision = lambda *args, **kwargs: None
        if depth_cameras():
            self.metadata['main_camera_input'] = self.main_camera_input
            self.metadata['depth_cameras'] = list(depth_cameras())
            self.metadata['depth_visualization_range_m'] = list(depth_range())
            self.metadata['depth_sensor'] = dict(camera='cam_high', units='meters',
                field=depth_key('cam_high'), type='distance_to_image_plane', aligned_with_rgb=True)

    def _check_identity(self, episode_id, step_id):
        require(not self.poisoned and episode_id == self.episode_id and step_id == self.step_id,
                'Stale episode/tick or uncertain simulator state')

    def _observe(self, record=False):
        aliases = dict(cam_high=('cam_high', 'cam_head', 'head_camera', 'top_camera'),
                       cam_left_wrist=('cam_left_wrist', 'left_camera'),
                       cam_right_wrist=('cam_right_wrist', 'right_camera'))
        raw = self.env.get_obs()
        state = raw['state']
        states = np.concatenate([np.r_[state[f'{arm}_arm_joint_state'], state[f'{arm}_ee_joint_state']]
                                 for arm in ('left', 'right')]).astype(np.float32)
        require(states.shape == (14,) and np.isfinite(states).all(), 'Invalid native dual-arm state')
        images, depths = {}, {}
        for key, names in aliases.items():
            source = next((n for n in names if n in raw['vision']), None)
            require(source is not None, f'Missing {key}; native cameras={list(raw["vision"])}')
            images[key] = np.asarray(raw['vision'][source]['color'], np.uint8)[..., :3]
            if key in depth_cameras():
                depths[depth_key(key)] = measured_depth(raw['vision'][source], images[key].shape)
        poses = np.asarray([state[f'{arm}_ee_pose'] for arm in ('left', 'right')], np.float32)
        self.obs = dict(**images, states=states, eef_positions=poses[:, :3],
            eef_quaternions_wxyz=poses[:, 3:], instruction=raw['instruction'],
            remaining_steps=max(0, self.metadata['max_episode_steps']-self.step_id))
        self.obs.update(depths)
        self.obs['grasp_feedback_json'] = capture_grasp_feedback(self.env,
            episode_id=self.episode_id, step_id=self.step_id,
            control_dt=self.metadata['control_dt'])
        intrinsic_arrays, intrinsic_descriptions = capture_intrinsics(self.env, images)
        self.obs.update(intrinsic_arrays)
        self.metadata['camera_intrinsics'] = intrinsic_descriptions
        if self.metadata.get('provides_camera_geometry', False):
            geometry_arrays, _ = capture_geometry(
                self.camera_calibration, poses[:, :3], poses[:, 3:])
            self.obs.update(geometry_arrays)
        if record:
            self.teacher_history.capture(self.step_id, images['cam_high'])
            directory = self.episode_dir/'observations'
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f'{self.step_id:06d}.npz', **self.obs)
            if self.video_writer is None:
                import imageio.v2 as imageio
                self.video_writer = imageio.get_writer(str(self.output/'sensors.mp4'),
                    fps=1/self.metadata['control_dt'], codec='libx264', pixelformat='yuv420p',
                    macro_block_size=2, output_params=['-movflags', '+faststart'])
            frames = [np.asarray(Image.fromarray(images[k]).resize((640, 360))) for k in CAMERAS]
            self.video_writer.append_data(np.concatenate(frames, axis=1))
            self.video_frames += 1
        return self.obs

    def reset(self, seed, source, policy_version):
        require(self.episode_id is None, 'One fresh episode only; no rollback/reset')
        if self.evaluation_identity:
            require(seed == self.evaluation_identity['layout_id'], 'Reset seed differs from the frozen layout_id')
            actual_layout = self.env.seed_manager.seed_info[seed]['scene_layout']
            require(sha256(actual_layout) == self.evaluation_identity['layout_sha256'],
                    'Native layout changed before reset')
        try:
            self.env.reset(seed=[seed])
        except Exception as error:
            # Only the native named exception is classified as an invalid layout;
            # infrastructure/model errors are never silently removed from evaluation.
            write_json(self.output/'evaluation_outcome.json', dict(
                evaluation_case=self.evaluation_identity, complete=False, valid_for_success_rate=False,
                status='invalid_native_layout' if type(error).__name__ == 'UnStableError' else 'reset_error',
                error_type=type(error).__name__, error=str(error)))
            raise
        registration = register_native_evaluation(self.env)
        write_json(self.output/'native_evaluation.json', registration)
        self.episode_id = uuid.uuid4().hex
        self.episode_dir = self.output/self.episode_id
        self.episode_dir.mkdir()
        self.teacher_history = EpisodeImageArchive(self.episode_dir/'teacher_history_2hz',
            self.episode_id, self.metadata['control_dt'])
        self.poisoned = False
        self.kinematics = DualKinematics(self.env)
        write_json(self.output/'fk_validation.json', self.kinematics.check())
        self.camera_calibration = from_environment_config(self.env)
        self.metadata['camera_calibration'] = self.camera_calibration
        write_json(self.output/'camera_calibration.json', self.camera_calibration)
        self._observe(record=True)
        fingerprints = {}
        for key, value in self.obs.items():
            arr = np.asarray(value)
            fingerprints[key] = dict(shape=list(arr.shape), dtype=str(arr.dtype),
                sha256=hashlib.sha256(arr.tobytes()).hexdigest())
        write_json(self.output/'initial_observation_fingerprint.json', dict(
            evaluation_case=self.evaluation_identity, fields=fingerprints,
            scope='Recorded RGB, proprio, robot EEF and instruction; not a complete physics state',
            bitwise_physics_reproducibility_guaranteed=False))
        self.metadata['instruction'] = self.obs['instruction']
        identity = hashlib.sha256(self.obs['states'].tobytes()).hexdigest()
        result = dict(episode_id=self.episode_id, step_id=0, initial_state_hash=identity,
            instruction=self.obs['instruction'], metadata=dict(self.metadata, layout_id=seed,
            eval_seed=getattr(self.env, 'eval_seed', None), evaluation_case=self.evaluation_identity,
            initial_state_hash_scope='robot_proprio_only_not_complete_scene',
            policy_version=policy_version, native_reset_settling_not_counted_as_policy_controls=True))
        write_json(self.output/'reset.json', result)
        return result

    def chunk_step(self, actions, **kwargs):
        require(not self.finished and not self.terminated and not self.truncated, 'Episode finished')
        a = np.asarray(actions, np.float32)
        require(a.ndim == 2 and a.shape[1] == 14 and 1 <= len(a) <= STUDENT_MAX_STEPS and np.isfinite(a).all(),
                f'Expected 1..{STUDENT_MAX_STEPS} finite 14D absolute actions')
        require(np.all((a[:, [6, 13]] >= 0) & (a[:, [6, 13]] <= 1)), 'Invalid gripper opening')
        rows = []
        for action in a:
            command = {key: value for arm, offset in (('left', 0), ('right', 7))
                for key, value in ((f'{arm}_arm_joint_state', action[offset:offset+6]),
                                   (f'{arm}_ee_joint_state', action[offset+6:offset+7]))}
            before = int(self.env.take_action_cnt[0])
            native_action_started_ns = time.perf_counter_ns()
            self.env.take_action(command)
            native_action_elapsed_ns = time.perf_counter_ns() - native_action_started_ns
            require(int(self.env.take_action_cnt[0]) == before+1, 'Native action was not executed exactly once')
            self.step_id += 1
            ended = bool(self.env.end_flag[0])
            self.success = bool(ended and self.env.success[0])
            self.truncated = bool(ended and not self.success and self.step_id >= self.env.step_lim)
            self.terminated = bool(ended and not self.truncated)
            # Physical ACK is already certain even if a later camera/disk read
            # fails. Missing measurements must never erase an executed action.
            row = dict(valid=True, executed_action=action.copy(), native_action_elapsed_ns=native_action_elapsed_ns,
                obs=dict(states=np.full(14, np.nan, dtype=np.float32)),
                step_id=self.step_id, terminated=self.terminated, truncated=self.truncated, success=self.success,
                source=self.source, control_epoch=self.control_epoch, **kwargs)
            rows.append(row)
            obs = self._observe(record=True)
            row['obs'] = dict(states=obs['states'])
            row['measurement_available'] = True
            write_json(self.episode_dir/f'action_{self.step_id-1:06d}.json',
                dict(row, executed_action=action.tolist(), obs=dict(states=obs['states'].tolist())))
            if ended:
                self._write_summary('terminal')
                break
        return dict(episode_id=self.episode_id, step_id=self.step_id, steps=rows)

    def _write_summary(self, reason):
        if self.episode_id is None:
            return
        if self.video_writer is not None:
            self.video_writer.close(); self.video_writer = None
        if self.finish_reason is not None:
            reason = self.finish_reason  # Do not overwrite a budget/error outcome during server cleanup.
        complete = self.terminated or self.truncated
        invalid = 0 in getattr(self.env, 'unstable_envs', set())
        eligible = complete and not invalid and not self.poisoned
        score = None
        if eligible:
            score = 1.0 if self.success else (float(self.env.reward_manager.get_score()[0])/100
                     if hasattr(self.env, 'get_score') else 0.0)
        write_json(self.output/'evaluation_outcome.json', dict(
            evaluation_case=self.evaluation_identity, complete=complete,
            valid_for_success_rate=eligible, native_success=self.success if eligible else None,
            native_score=score, status=('invalid_native_layout' if invalid else
                'native_completed' if eligible else 'budget_censored' if reason == 'decision_budget' else 'incomplete'),
            reason=reason, native_control_steps=self.step_id,
            native_step_limit=self.metadata['max_episode_steps']))
        write_json(self.output/'summary.json', dict(episode_id=self.episode_id, step_id=self.step_id,
            success=self.success, terminated=self.terminated, truncated=self.truncated,
            complete=self.terminated or self.truncated, reason=reason, video_frames=self.video_frames,
            control_dt=self.metadata['control_dt'], no_rollback=True,
            evaluation_case=self.evaluation_identity))

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if op == 'reset':
            return self.reset(**args)
        self._check_identity(args['episode_id'], args['step_id'])
        data = {k: v for k, v in args.items() if k not in ('episode_id', 'step_id')}
        if op == 'teacher_observation':
            return self.obs  # Cached exact post-ACK observation; no hidden step/render.
        if op == 'teacher_history':
            require(data['through_step'] == self.step_id, 'History must end at current acknowledged step')
            return self.teacher_history.page(data['cursor'], self.step_id)
        if op == 'record_codex_decision':
            return record_decision(self, args)
        if op == 'begin_combination':
            write_json(self.output/'combination.json', data)
            return dict(physical_steps=0)
        if op == 'switch_control_source':
            require(data['source'] in ('student', 'gpt_eef', 'gpt_joint'), 'Invalid source')
            self.source = data['source']; self.control_epoch += 1
            write_json(self.episode_dir/f'source_{self.control_epoch:04d}.json', dict(step_id=self.step_id, **data))
            return dict(physical_steps=0, source=self.source, control_epoch=self.control_epoch)
        if op == 'fk_preview':
            return self.kinematics.preview(data['actions'])
        if op == 'check_eef_waypoints':
            return self.kinematics.check_waypoints(data['waypoints'],
                remaining_steps=max(0, self.metadata['max_episode_steps']-self.step_id))
        if op == 'check_eef_targets':
            return self.kinematics.check_targets(data['targets'])
        if op == 'eef_joint_target':
            return self.kinematics.target(data['targets'])
        if op == 'chunk_step':
            return self.chunk_step(**data)
        if op == 'finish_pilot':
            self.finished = True
            self.finish_reason = data['reason']
            result = dict(episode_id=self.episode_id, step_id=self.step_id, success=self.success,
                terminated=self.terminated, truncated=self.truncated, video_frames=self.video_frames,
                control_dt=self.metadata['control_dt'], reason=data['reason'],
                evaluation_case=self.evaluation_identity)
            self._write_summary(data['reason'])
            return result
        raise ValueError('Unsupported no-rollback operation: '+op)
