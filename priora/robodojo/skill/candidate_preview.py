"""Reuse Jev's current fingertip overlays as additional GPT evidence."""
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import write_json
from ..realtime_monitor.current_view_selection import selected_camera_order, select_current_views
from ..realtime_monitor.digest import digest
from ..realtime_monitor.projection import CAMERAS, annotated_views, project_candidate_views
from .image_preview import prepare_image

GUIDE = """Candidate trajectory images:
candidate_trajectory_images are ADDITIONAL annotated CURRENT RGB images of an
UNEXECUTED pi0.5 candidate, separate from clean observations and ACTUAL history.
They reuse Jev's two-fingertip projection and current-view selection. The head
view is always supplied; wrists follow the same visibility/spatial-motion/grasp
exceptions as Jev. No history images are removed by this selection.
Blue paths mean commanded closure, orange paths commanded opening; left arm is
solid, right arm dashed. Orange disks mark closed-to-open commands. Arrows show
trajectory motion direction. Colors/markers do not prove contact or release.
Projection uses each camera's CURRENT pose; wrist paths can be partial as the
camera will move. A clipped endpoint does not mean motion stops there. These
are predicted robot paths, not predicted object motion or completed actions.
The label binds each image to the request, observation step and candidate prefix.
Only that prefix is drawn, even when the numerical proposal contains more steps.
After inference or goal regeneration use the newly returned images; old candidate
images in the conversation are no longer the current proposal. Full-resolution
sources remain at path; uploaded previews use the normal JPEG image settings.
"""


def _local_model_path(rollout, path):
    """Undo the existing Jev service mount mapping for this controller's files."""
    live = getattr(rollout, 'live', {})
    source, destination = live.get('shared_path_from'), live.get('shared_path_to')
    if source or destination:
        if not source or not destination:
            raise ValueError('Both shared path mapping endpoints are required')
        return Path(source).resolve() / Path(path).relative_to(Path(destination))
    return Path(path)


def _from_boundary(rollout):
    binding, temporary = rollout.boundary
    candidate = temporary['candidate_prediction']
    if (binding.observation_id != rollout.current_actual['observation_id']
            or binding.execution_watermark != rollout.tick
            or binding.control_epoch != rollout.control_epoch
            or binding.stage_id != rollout.stage['stage_id']
            or binding.stage_version != rollout.stage['version']
            or candidate['source_prediction_id'] != rollout.prediction['prediction_id']
            or candidate['status'] != 'UNEXECUTED'
            or candidate['prefix_steps'] != binding.prefix_steps
            or digest(candidate) != binding.proposal_digest
            or candidate['effective_actions_sha256'] != hashlib.sha256(
                np.asarray(rollout.actions[:binding.prefix_steps]).tobytes()).hexdigest()):
        raise ValueError('Stale Jev candidate cannot be attached to GPT')
    overlays = temporary['candidate_overlays']
    if [row['camera_id'] for row in overlays] != list(CAMERAS):
        raise ValueError('Jev candidate must contain the three ordered source views')
    for row in overlays:
        if (row['source_kind'] != 'candidate_overlay'
                or row['background_observation_id'] != binding.observation_id
                or row['proposal_digest'] != binding.proposal_digest
                or row['native_step'] != rollout.tick
                or row['simulation_time_ns'] != rollout.current_actual['simulation_time_ns']):
            raise ValueError('Jev candidate image binding mismatch')
    order = selected_camera_order(candidate.get('view_selection'))
    images = [dict(camera=row['camera_id'], path=str(_local_model_path(rollout, row['image_path'])),
                   file_sha256=row['file_sha256'], pixel_digest=row['pixel_digest'])
              for row in overlays if row['camera_id'] in order]
    return dict(source='onejev_current_candidate', observation_id=binding.observation_id,
                proposal_digest=binding.proposal_digest, prefix_steps=binding.prefix_steps,
                selected_cameras=order), images


def _render_without_monitor(rollout):
    """Use the same renderer when running GPT without an active Jev monitor."""
    needed = [key for camera in CAMERAS for key in
              (camera, camera+'_intrinsics_K', camera+'_T_environment_camera')]
    if any(key not in rollout.obs for key in needed) or not rollout.fk.get('fingertip_geometry'):
        return dict(status='unavailable', reason='Current camera calibration or fingertip geometry missing'), []
    prefix = min(getattr(rollout, 'student_max_steps', 30), len(rollout.actions), int(rollout.obs['remaining_steps']))
    trajectory = rollout.trajectory()[:prefix]
    projection = project_candidate_views(rollout.obs, trajectory, rollout.fk['fingertip_geometry'])
    order = selected_camera_order(select_current_views(projection, rollout.current_eef(), trajectory))
    views = annotated_views(rollout.obs, projection)
    directory = Path(rollout.proposal_path).parent/'teacher_candidate'
    directory.mkdir(exist_ok=True)
    images = []
    for camera in order:
        picture = Image.fromarray(views[camera]).convert('RGB')
        stream = BytesIO()
        picture.save(stream, format='PNG')
        payload = stream.getvalue()
        path = directory/(camera+'.png')
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError('Refusing to overwrite an existing candidate image')
        else:
            path.write_bytes(payload)
        images.append(dict(camera=camera, path=str(path),
            file_sha256=hashlib.sha256(payload).hexdigest(),
            pixel_digest=hashlib.sha256(picture.tobytes()).hexdigest()))
    return dict(source='jev_renderer_without_monitor', prefix_steps=prefix,
        selected_cameras=order), images


def add_candidate_images(rollout, packet, *, tool):
    # No duplicate attachments on approval, memory or execution-result calls.
    # These are never inserted into the actual-observation/history image list.
    if (tool not in ('pi05_infer', 'onejev_publish_stage')
            or rollout.phase != 'execute' or packet.get('rollout_finished')):
        return packet
    request = rollout.request
    if request['step_id'] != rollout.tick or request['episode_id'] != rollout.episode:
        raise ValueError('Stale inference request cannot supply candidate images')
    if packet.get('request_id', request['request_id']) != request['request_id']:
        raise ValueError('Candidate image request identity mismatch')
    details, images = (_from_boundary(rollout) if getattr(rollout, 'boundary', None)
                       else _render_without_monitor(rollout))
    info = dict(status='UNEXECUTED', episode_id=rollout.episode,
        request_id=request['request_id'], step_id=rollout.tick,
        simulation_time_s=rollout.tick*rollout.meta['control_dt'])
    info.update(details)
    result = dict(packet, candidate_trajectory=info, candidate_trajectory_images=images)
    # Full checksums and identities stay in the local audit, not repeated in GPT text.
    manifest = Path(rollout.proposal_path).parent/'teacher_candidate_images.json'
    write_json(manifest, dict(candidate_trajectory=info, images=images))
    return result


def prepare_candidate_image(item, max_edge):
    payload = Path(item['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != item['file_sha256']:
        raise ValueError('Candidate image changed after Jev rendered it')
    with Image.open(BytesIO(payload)) as picture:
        if hashlib.sha256(picture.convert('RGB').tobytes()).hexdigest() != item['pixel_digest']:
            raise ValueError('Candidate image pixels do not match Jev evidence')
    # Preserve the existing teacher JPEG bandwidth/size policy; never alter the PNG.
    descriptor, jpeg = prepare_image(dict(camera=item['camera'], path=item['path']), max_edge)
    preview = descriptor['codex_preview']
    descriptor['codex_preview'] = {key: preview[key] for key in ('path', 'size', 'original_size')}
    return descriptor, jpeg
