"""Episode-wide 2 Hz real images; no goal resets and no policy-side inference."""
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from .io import write_json
from .skill.image_preview import encode_jpeg

VERSION = 'gpt_episode_main_2hz_v1'
GUIDE = """Episode history:
This same GPT conversation retains your earlier public messages, tool calls and tool
results across all subtask changes and takeovers. New timestamped MAIN-camera RGB
history frames are appended at 2 Hz of simulation time, including autonomous pi0.5
execution while you were not called. They are ACTUAL past observations, not future
trajectories. Each image has its own episode, step and time label immediately before
it. Combine these with prior dialogue and the current three-camera observation;
past interpretations and issued commands are not proof of actual success. No
subtask change clears GPT episode history. Already-delivered frames remain in the
conversation and are not duplicated at every call. The full chronological image
index and public dialogue are archived locally. If native context compaction occurs,
retrieve needed evidence from the archive; do not claim unlimited raw context.
"""


def sample_step(index, control_dt):
    # Nearest real control sample; ties go forward. 25 Hz -> 0,13,25,38,50...
    return int(math.floor(index / (2 * control_dt) + .5))


class EpisodeImageArchive:
    def __init__(self, root, episode_id, control_dt):
        if not math.isfinite(control_dt) or not 0 < control_dt <= .5:
            raise ValueError('2 Hz history requires a finite control period <= 0.5 s')
        self.root, self.episode_id, self.dt = Path(root), episode_id, control_dt
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def capture(self, step, rgb):
        target = sample_step(len(self.rows), self.dt)
        if step < target:
            return
        if step != target:
            raise ValueError('Missing actual 2 Hz history sample; cannot substitute a later frame')
        payload = encode_jpeg(Image.fromarray(rgb))
        path = self.root / f'{step:08d}_cam_high.jpg'
        path.write_bytes(payload)
        row = dict(episode_id=self.episode_id, step_id=step,
                   simulation_time_s=round(step * self.dt, 9),
                   sampling_target_time_s=len(self.rows) / 2,
                   camera='cam_high', modality='rgb', evidence='actual_past_observation',
                   sha256=hashlib.sha256(payload).hexdigest(), path=str(path))
        self.rows.append(row)
        write_json(self.root/'index.json', dict(version=VERSION, hz=2,
            scope='whole_episode', episode_id=self.episode_id, images=self.rows))

    def page(self, cursor, through_step, limit=16):
        if type(cursor) is not int or not 0 <= cursor <= len(self.rows):
            raise ValueError('Invalid history cursor')
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError('History page limit must be 1..16')
        rows = [row for row in self.rows[cursor:cursor+limit] if row['step_id'] <= through_step]
        return dict(episode_id=self.episode_id, images=[dict(row,
            jpeg=Path(row['path']).read_bytes()) for row in rows],
            next_cursor=cursor+len(rows), total=sum(row['step_id'] <= through_step for row in self.rows))


class TeacherEpisodeHistory:
    """Fetch archive pages via RPC, then append each frame once per GPT thread.

    Delivery cursor commits only after the transport accepted the tool reply.
    Fetching advances no physics, calls no model, and does not affect Jev history.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_id = None
        self.rows = []
        self.delivered = 0

    def prepare(self, rollout, packet):
        if not getattr(rollout, 'episode', None):
            return packet, self.delivered
        if not rollout.meta.get('provides_teacher_episode_history'):
            return dict(packet, gpt_episode_history=dict(status='unavailable',
                reason='Simulator does not advertise the 2 Hz archive')), self.delivered
        if self.episode_id not in (None, rollout.episode):
            raise ValueError('Cannot mix different episodes in a GPT conversation')
        self.episode_id = rollout.episode
        while True:
            page = rollout._rpc('teacher_history', cursor=len(self.rows), through_step=rollout.tick)
            if page['episode_id'] != self.episode_id:
                raise ValueError('Wrong episode in image history')
            start = len(self.rows)
            if page['next_cursor'] != start + len(page['images']):
                raise ValueError('Non-contiguous image history')
            for item in page['images']:
                item = dict(item)
                payload = item.pop('jpeg')
                if item['episode_id'] != self.episode_id or item['step_id'] > rollout.tick:
                    raise ValueError('Future or foreign history image')
                if self.rows and item['step_id'] <= self.rows[-1]['step_id']:
                    raise ValueError('History images must be strictly ordered')
                if hashlib.sha256(payload).hexdigest() != item['sha256']:
                    raise ValueError('History image checksum mismatch')
                path = self.root / f"{item['step_id']:08d}_cam_high.jpg"
                path.write_bytes(payload)
                item.update(path=str(path), teacher_history=True,
                            label=f"ACTUAL HISTORY | cam_high | step={item['step_id']} | "
                                  f"t={item['simulation_time_s']:.3f}s | episode={self.episode_id}")
                self.rows.append(item)
            if len(self.rows) == page['total']:
                break
            if len(self.rows) == start:
                raise ValueError('History pagination made no progress')
        write_json(self.root/'index.json', dict(version=VERSION, episode_id=self.episode_id,
            hz=2, scope='whole_episode', images=self.rows))
        return dict(packet, images=self.rows[self.delivered:]+list(packet.get('images', [])),
            gpt_episode_history=dict(version=VERSION, scope='whole_episode', hz=2,
                cumulative_frames=len(self.rows), previously_delivered=self.delivered,
                new_frames=len(self.rows)-self.delivered, cleared_on_goal_change=False,
                index_path=str(self.root/'index.json'),
                dialogue_path=str(self.root.parent/'gpt_public_dialogue.jsonl'))), len(self.rows)

    def commit(self, count):
        if not self.delivered <= count <= len(self.rows):
            raise ValueError('Invalid delivery cursor')
        self.delivered = count
        write_json(self.root/'delivery.json', dict(episode_id=self.episode_id, delivered=count))


def append_dialogue(path, record):
    # Call only with public conversation/tool content, never private reasoning.
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
