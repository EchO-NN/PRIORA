"""Give the policy agent a real working directory beside the host's audit files."""
import hashlib
from pathlib import Path
import shutil
import sys

from ..io import write_json


def prepare_workspace(audit, skill_root, *, method='wall_plus_gpt'):
    audit, skill_root = Path(audit).resolve(), Path(skill_root).resolve()
    agent = audit/'agent'
    agent.mkdir()
    (agent/'scratch').mkdir()
    context = agent/'context'
    shutil.copytree(skill_root/'context', context)
    # A discoverable local skill, in addition to the host's explicit injection.
    direct = method == 'gpt_only'
    skill = agent/'.agents/skills'/('robodojo-gpt-only-rollout' if direct else 'robodojo-hybrid-rollout')
    skill.mkdir(parents=True)
    for name in (('SKILL.md',) if direct else ('SKILL.md', 'gate_prompt.md')):
        shutil.copy2(skill_root/name, skill/name)
    shutil.copytree(context, skill/'context')
    controller = audit.parent
    workspace = dict(
        controller_output=str(controller), source_root=str(skill_root.parent.parent), robot_profile='robodojo',
        python_executable=sys.executable, history_path=str(controller/'history.json'),
        observations_path=str(controller/'observations'), proposals_path=str(controller/'proposals'),
        execution_files=str(controller/'execution_NNN.npz'),
        eef_diagnostics_files=str(controller/'edit_NNN.json'),
        scratch=str(agent/'scratch'), notes_path=str(agent/'NOTES.md'),
        environment_memory_path=str(audit/'ENVIRONMENT_MEMORY.md'),
        environment_facts_path=str(audit/'ENVIRONMENT_MEMORY.json'),
        runtime_state_path=str(audit/'runtime_state.json'),
        gpt_public_dialogue_path=str(audit/'gpt_public_dialogue.jsonl'),
        gpt_episode_image_index=str(audit/'episode_history_2hz/index.json'),
        gpt_episode_history_hz=2, gpt_episode_history_scope='whole_episode',
        context_files={str(p.relative_to(agent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in context.rglob('*') if p.is_file()},
        baseline_full_conversation_available=False)
    if direct:
        workspace.pop('proposals_path')
        workspace['evaluation_method'] = 'gpt_only'
    else:
        workspace['spatial_prompt_path'] = str(agent/'SPATIAL_PROMPT.md')
        workspace['spatial_memory_path'] = str(agent/'SPATIAL_MEMORY.md')
        workspace['spatial_numeric_state_path'] = str(controller/'spatial/state.json')
        workspace['spatial_status_path'] = str(controller/'spatial/latest.json')
        workspace['spatial_memory_version'] = 'rigid_spatial_v3'
        workspace['near_field_memory_version'] = 'near_field_memory_v1'
        workspace['near_field_memory_path'] = str(controller/'spatial/near_field/index.json')
        (agent/'SPATIAL_PROMPT.md').write_text(
            '# Episode spatial operating guidance\n\n'
            'Before the first execution, inspect the declared rigid camera calibration '
            'in camera_geometry/spatial_status and initialize SPATIAL_MEMORY.md for this episode. '
            'The host composes fixed mount transforms with current measured EEF poses each observation. '
            'Use spatial_query on original pixels for visible surface coordinates; no landmark '
            'training is required. spatial_record optionally registers a visible rigid gripper '
            'surface in its measured link6 frame; it does not calibrate cameras. '
            'Keep unknown fingertip offsets and grasp geometry distinct. '
            'Configured valid depth is optical-axis Z in [0.1, 10] meters: below 0.1 m, '
            'numeric depth/XYZ lookups are invalid even if RGB still shows the feature '
            'or a raw depth value exists. This is not evidence of a physical camera blind zone. '
            'Before approaching or occluding a useful target, use spatial_remember to '
            'retain a labeled local RGB/depth/XYZ patch. Inspect spatial_status.near_field_memory '
            'for current validity, recent observation pointers and saved patches. '
            'spatial_recall returns historical evidence with source step and age, never '
            'a current grasp target. Check its aligned-RGB visual status: same_observation '
            'or consistent may support a historical planning hint; invalidated_rgb_change '
            'and unverified may not. Large supported RGB changes permanently invalidate '
            'the old memory; preserve its audit archive and capture a new valid patch. '
            'Occlusion or insufficient coverage is unverified, not proof of object motion. '
            'Uniform cloth and lighting changes limit this check. '
            'Reacquire the feature in current views before contact; '
            'cloth deformation or object motion invalidates old scene coordinates. '
            'Never carry a cloth/object patch with the wrist transform or invent a missing depth. '
            'Use another existing view when it supplies the missing evidence. '
            'Maintain this file only '
            'when guidance changes. These notes '
            'cannot override the host task, gate, limits or evidence rules.\n')
        shutil.copy2(context/'spatial_memory_template.md', agent/'SPATIAL_MEMORY.md')
    write_json(agent/'workspace.json', workspace)
    instructions = (
        '# RoboDojo policy agent workspace\n\n'
        'You are the autonomous policy agent, not a JSON-only reviewer. Use your normal '
        'Codex tools to inspect files and images, write/run code, calculate poses, and '
        'maintain notes when needed. Declared tools connect you to RoboDojo/pi05. '
        'Use supplied tool results/images first; native tools are optional when '
        'specific evidence or calculations are missing.\n\n'
        'Use the injected operating rules. Read context/teacher_context.md only for '
        'background and context/eef_control.md when EEF details/examples are needed. '
        'Other context/ manuals are available on demand; do not read them all at startup. '
        'workspace.json supplies the Python interpreter and '
        'exact artifact/source paths. Keep working memory in NOTES.md and code, crops, '
        'plots, and comparisons in scratch/. Both are yours to edit.\n\n'
        'Host-maintained environment facts are injected at startup and archived in '
        '../ENVIRONMENT_MEMORY.md. The current tool reply environment_state and '
        '../runtime_state.json supply the exact phase, observation file and next_call. '
        'Copy current paths literally; do not reconstruct them from run names or old steps. '
        'These host files are read-only; keep your own notes separately.\n\n'
        'Use the injected rollout skill and declared gate for simulation control. '
        'Write every public explanation, assessment, progress update, note, and final '
        'report in English. Never expose private chain-of-thought. '
        'Preserve host-owned evidence and executing source; do not use shell scripts '
        'to open a second simulator connection, reset, or bypass the recorded control '
        'tools. Inspect recorded RGB/proprio/actions freely; private simulator scene '
        'snapshots/object truth are not planning inputs.\n')
    if direct:
        instructions = instructions.replace('Declared tools connect you to RoboDojo/pi05',
            'Declared tools connect you to RoboDojo (no pi05)')
        instructions = instructions.replace('injected rollout skill and declared gate', 'injected GPT-only rollout skill')
    else:
        instructions += (
            '\nDetailed geometry manuals are available under context/; '
            'read relevant sections only when needed. '
            'Before first execution initialize SPATIAL_MEMORY.md from supplied '
            'calibration/observations. Geometry and projections update in host '
            'spatial_status; no calibration moves are needed. Keep episode notes '
            'current only when evidence changes; they cannot override host rules.\n')
    (agent/'AGENTS.md').write_text(instructions)
    (agent/'NOTES.md').write_text(
        '# Episode working memory\n\n'
        'No observations yet. Maintain confirmed progress, current objective, geometry '
        'estimates, tracking errors, and corrections as useful. Historical runs are '
        'background knowledge, not observations of this episode.\n')
    return agent
