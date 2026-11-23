"""WALL proposals keep their native 16D EEF format through observation and ACK."""
import copy
import json
import hashlib
import os
import shutil
from pathlib import Path
import time
import uuid

import numpy as np

from ...cards import load_card
from ..io import InputError, write_json
from ..robodojo_server.client import RoboDojoTools, ARMS, validate_dual_response
from ..robodojo_server.protocol import RPCClient
from ..pi05_server.gripper_hold import GripperHold
from ..skill.run import CodexPolicy


class WallGripperHold(GripperHold):
    def metadata(self):
        result=super().metadata()
        result['note']=result['note'].replace('Pi05','WALL').replace('pi05','WALL')
        return result


class WallPolicyClient:
    def __init__(self, port=19660):
        self.rpc = RPCClient('127.0.0.1', port, timeout=600)
        self.metadata = self.rpc.request('metadata')
        if self.metadata.get('backend') != 'WALL-OSS/native-EEF':
            raise RuntimeError('Wrong student service')
        self.session_id=None;self.index=0;self.last_frame=-1

    def infer(self, observation, output):
        r=self.rollout
        if self.session_id is None:
            self.session_id=r.episode
            self.rpc.request('reset',session_id=self.session_id)
        state={}
        for i,arm in enumerate(ARMS):
            state[arm+'_arm_joint_state']=observation['states'][7*i:7*i+6]
            state[arm+'_ee_joint_state']=observation['states'][7*i+6:7*i+7]
            state[arm+'_ee_pose']=np.r_[observation['eef_positions'][i],observation['eef_quaternions_wxyz'][i]]
        frames=r._rpc('wall_history',after=self.last_frame)['frames']
        obs=dict(vision={key:dict(color=observation[key]) for key in r.camera_keys},
                 state=state,instruction=observation['instruction'])
        result=self.rpc.request('infer',session_id=self.session_id,tick=r.tick,task=r.task,
                                observation=obs,frames=frames)
        if result['received_prompt'] != observation['instruction']:
            raise RuntimeError('WALL received a different subtask')
        self.last_frame=r.tick
        raw=np.asarray(result['actions'],np.float32); actions=raw.copy()
        actions[:,[7,15]]=np.clip(actions[:,[7,15]],0,1)
        np.savez_compressed(output,raw_actions=raw,actions=actions,
                            **{k:observation[k] for k in (*r.camera_keys,'states')})
        self.index+=1
        prediction=dict(prediction_id=uuid.uuid4().hex,inference_index=self.index-1,
            inference_seconds=result['seconds'],policy_instruction=observation['instruction'],
            frame_transport='controller_rpc',
            policy_server_received_prompt=result['received_prompt'],
            policy_instruction_context=observation.get('policy_instruction_context'),
            wall_memory=result['memory'],student_backend=self.metadata['backend'])
        write_json(Path(output).with_name('wall_memory.json'),dict(
            session_id=self.session_id,step_id=r.tick,**result['memory']))
        return actions,prediction

    def close(self):
        try:
            if self.session_id is not None and self.rpc.sock is not None:
                self.rpc.request('close',session_id=self.session_id)
        finally:self.rpc.close()


class WallTools(RoboDojoTools):
    student_max_steps=16

    def verify_eef_report(self, report, **kwargs):
        from .curobo_kinematics import VERSION, limits
        from ..robodojo_server.waypoint_guard import verify_report
        if self.meta.get('gpt_eef_solver') == VERSION:
            kwargs.update(expected_limits=limits(self.meta['control_dt']),expected_solver=VERSION)
        verified=verify_report(report, **kwargs)
        if kwargs.get('nodes') is not None and self.meta.get('motion_smoothing'):
            from . import motion_smoothing as smooth
            from ..robodojo_server.reachability import checked_targets
            # The simulator validates/canonicalizes targets before hashing them.
            # Match that representation, including integer-to-float conversion.
            nodes=[dict(n,target=checked_targets(n['target'])) for n in kwargs['nodes']]
            detail=report.get('motion_smoothing') or {}
            if (detail.get('version') != smooth.VERSION or
                    detail.get('joint_alpha') != smooth.JOINT_ALPHA or
                    detail.get('nodes_sha256') != smooth.digest(nodes)):
                raise RuntimeError('EEF smoothing metadata differs between client and simulator')
            schedule=smooth.verified_schedule(nodes,detail)
            self._verified_smooth_eef=(self.tick,smooth.digest(kwargs['nodes']),schedule)
            # The archived RPC contains the exact controls. Keep the model's
            # reachability result compact for both accepted and rejected plans.
            verified=dict(verified,motion_smoothing={k:v for k,v in detail.items() if k!='eef_schedule'})
        return verified

    def correction_targets(self,response):
        if response['mode']=='eef' and self.meta.get('motion_smoothing'):
            from .motion_smoothing import digest
            from ..robodojo_server.eef_waypoints import validated_waypoints
            nodes=validated_waypoints(response,self.request)
            verified=getattr(self,'_verified_smooth_eef',None)
            if verified is None or verified[:2] != (self.tick,digest(nodes)):
                raise RuntimeError('No matching smoothed EEF preflight for this step')
            return copy.deepcopy(verified[2])
        return super().correction_targets(response)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        previous=self.gripper_hold
        self.gripper_hold=WallGripperHold(previous.mode,previous.close_threshold,previous.release_threshold)
        self.student.rollout=self
        self.subtask=None;self.subtask_history=[]
        self.require_whole_task_trial=os.environ.get('WALL_REQUIRE_WHOLE_TASK_TRIAL')=='1'
        self.whole_task_trial_completed=False
        self.service_handlers={'wall_set_subtask':self.set_subtask}

    def _rpc(self,op,**kwargs):
        if op in ('wall_eef_step','chunk_step'):
            reserve=int(os.environ.get('WALL_MIN_FREE_BYTES',str(int(1.5*2**30))))
            waited=False
            while True:
                native=self.sim.request('wall_storage',episode_id=self.episode,step_id=self.tick)
                local=shutil.disk_usage(self.output).free
                if min(local,native['free_bytes']) >= reserve:break
                waited=True
                write_json(self.output/'storage_wait.json',dict(state='waiting',step_id=self.tick,
                    controller_free_bytes=local,simulator_free_bytes=native['free_bytes'],reserve_bytes=reserve))
                time.sleep(5)
            if waited:
                write_json(self.output/'storage_wait.json',dict(state='resumed',step_id=self.tick))
        return super()._rpc(op,**kwargs)

    def _infer_proposal(self,directory):
        packet=super()._infer_proposal(directory)
        if self.subtask is None or self.subtask==self.obs['instruction']:
            self.whole_task_trial_completed=True
        self.request['execution_step_limits']=dict(self.request['execution_step_limits'],student=16)
        self.request['student_backend']='WALL-OSS/native-EEF'
        packet.update(execution_step_limits=self.request['execution_step_limits'],
            student_backend=self.request['student_backend'])
        write_json(self.output/f'request_{len(self.history):03d}.json',self.request)
        write_json(directory/'proposal.json',packet)
        return packet

    @staticmethod
    def validate(response,request):
        validate_dual_response(response,request)
        if response['mode']=='student' and response['steps']>16:
            raise ValueError('WALL student prefix must be 1..16 steps; GPT EEF still supports 1..30')

    def set_subtask(self,goal):
        if self.phase != 'infer':
            raise InputError('Set a subtask after start/execute and BEFORE pi05_infer; follow next_call')
        if not isinstance(goal,str) or not 1<=len(goal.strip())<=800:
            raise InputError('Provide one concise achievable subtask, 1..800 characters')
        goal=goal.strip()
        if (self.require_whole_task_trial and not self.whole_task_trial_completed
                and goal!=self.obs['instruction']):
            raise InputError('This experiment first tests the complete native task. '
                'Call pi05_infer without changing the goal; review the full-task proposal '
                'and prefer action correction. Decompose only after clear semantic intent errors.')
        if goal != self.subtask:
            self.subtask=goal
            self.subtask_history.append(dict(step_id=self.tick,goal=goal))
            write_json(self.output/'subtasks.json',self.subtask_history)
        return dict(step_id=self.tick,policy_instruction=self.subtask,
            native_task_instruction=self.obs['instruction'],physical_steps=0,
            next_call=self.next_call(),history_preserved=True,
            policy_instruction_context=self._policy_observation()['policy_instruction_context'])

    def _policy_observation(self):
        native=self.obs['instruction']
        instruction=self.subtask or native
        return dict(self.obs,instruction=instruction,
            policy_instruction_context=dict(
                source='native_whole_task' if instruction==native else 'gpt_subtask',
                native_goal=native))

    def preview_student_grippers(self,actions):
        if actions.shape != (32,16):raise ValueError('Expected WALL H32x16 proposal')
        proxy=np.zeros((len(actions),14),np.float32)
        proxy[:,[6,13]]=actions[:,[7,15]]
        proxy,states=self.gripper_hold.preview(proxy)
        result=actions.copy();result[:,[7,15]]=proxy[:,[6,13]]
        return result,states

    def preview_student_trajectory(self):
        geometry=self._rpc('wall_robot_geometry')
        trajectory=[dict(index=i,**{arm:dict(position=row[offset:offset+3].tolist(),
            quaternion_wxyz=row[offset+3:offset+7].tolist(),
            gripper_opening=float(row[offset+7]),gripper_closed=bool(row[offset+7]<.5))
            for arm,offset in (('left',0),('right',8))}) for i,row in enumerate(self.actions)]
        return dict(geometry,trajectory=trajectory,frame='environment_origin',
                    interpretation='native_WALL_EEF_targets_not_object_future')

    def student_diagnostics(self):
        return dict(action_format='H32x16 absolute EEF xyz+wxyz+opening per arm',
            native_action_horizon=32,maximum_executable_prefix=16,
            gripper_opening_range=[float(self.actions[:,[7,15]].min()),float(self.actions[:,[7,15]].max())],
            note='Pose targets are proposed commands, not predicted object motion or grasp success.')

    def execute_student(self,steps):
        return self._rpc('wall_eef_step',actions=self.actions[:steps],
                         student_prediction_id=self.prediction['prediction_id'])

    def gripper_history(self,executed):
        columns=(7,15) if executed.ndim==2 and executed.shape[1]==16 else (6,13)
        return [{arm:bool(row[column]<.5) for arm,column in zip(ARMS,columns)} for row in executed]


WALL_GUIDE='''
WALL backend and task decomposition:
The student is WALL-OSS 1.0 Lite RoboDojo, not pi0.5. The legacy tool name
pi05_infer calls WALL and returns its native H32 absolute dual-arm EEF proposal,
not joint angles. Each arm is xyz meters, unit quaternion wxyz, then opening.
The student mode executes 1..16 controls with RoboDojo native EEF IK. GPT edit
and EEF corrections use the same native cuRobo solver and 5/30 control budgets.
No Jev or other automatic reviewer is running. You review each proposal.
When a separate demonstration or opponent arm is performing its move, keep both
controlled arms still in their waiting poses. Resume only after the other arm
has completed its move, including any return motion, and come to rest. A brief
pause to grasp or release is part of its move. This also applies to reaching
for or picking up the next object away from the shared work area.
When the observation reports an enabled IK motion_guard, it keeps continuous
joint solutions and screens upper-arm/forearm interference. A blocked action
holds both arms and grippers, stops the remaining prefix and returns fresh
feedback; use that feedback to choose a reachable correction. Rate-limited
EEF targets may not be reached in one control. This is not a full collision
planner and does not certify object contact or successful grasping.
After robodojo_start, pi05_infer can run directly: without wall_set_subtask,
WALL receives the COMPLETE native task instruction, including its constraints.
For a single-object task, prioritize this whole-task input before decomposing.
Judge scope from the task and scene, not the count of all visible objects:
containers, destinations, supports and background clutter alone do not make
it a multi-object task. Do not invent a new scope-classification tool call.
Your internal task_progress/subgoals need not replace WALL's instruction.

Prefer action correction while preserving the whole-task goal. A missed grasp,
slip, inaccurate placement, awkward pose or rejected IK is an execution/geometry
problem, not by itself evidence that WALL cannot follow the complete task.
Apply the existing outcome/intent gate; normal intermediate progress and
uncertainty alone do not justify either takeover or task decomposition.
If observed behavior or the candidate provides clear evidence of a SEMANTIC
intent error (wrong target object, wrong destination, wrong required operation
or violated prerequisite/order), treat it as evidence that WALL is not handling
the whole-task instruction in the current state. Then GPT may decompose it into
concise subtask goals using wall_set_subtask, and continue supervising/correcting.
This is a recovery decision, not proof of the base model's general incapability.
For genuinely multi-object tasks, a coherent object or object-category subtask
may be selected when needed; do not force every task into microscopic phases.

Match every selected goal to the native whole-task requirements. KEEP the
existing instruction by default, whether whole-task or subtask. Do not prioritize
changing or rewording it when reviewing a proposal. If its target object/category,
destination and required outcome are unchanged, reuse the exact existing goal.
A routine intermediate approach/grasp/transport phase or GPT corrective action
does not by itself require changing the instruction. Once decomposition is
needed, during a failed attempt or retry reassess the observed scene: if an object dropped, moved, or otherwise
changed state so that the previous instruction no longer fits, you MAY revise
the subtask to match the recovery needed. Do not automatically rewrite every
failure, and do not force an obsolete instruction to stay unchanged. Call
wall_set_subtask for such a state-grounded retry, after completion to select
the next goal, or when the intended semantic goal truly changes;
call it BEFORE the next inference. The exact goal string
is passed to WALL and echoed as policy_instruction. Native success criteria
and task budgets are unchanged. Stage changes preserve GPT whole-episode
dialogue and 2 Hz image history. WALL maintains independent per-episode memory.
'''


CUROBO_GUIDE = """Per control step (25 Hz, 0.04 s), WALL and GPT EEF/edit share native cuRobo
32-seed IK, world/base conversion, nearest continuous joint-branch selection
and joint limits. Each commanded joint increment is at most 0.16 rad
(5 rad/s actuator limit times 0.04 s times the native 80% interpolation window).
There is no artificial 2 cm Cartesian DLS increment cap. Native self/table
collision constraints are sampled at 9 positions along each commanded joint
segment, with an additional elbow mesh exclusion. Other objects and inter-arm
collisions are not certified. Preflight uses this same solver without stepping
physics. Actual execution recomputes from measured joints after every ACK.
Allocate enough controls to reach the target; a preflight pass is not proof of physical
arrival or grasp. A blocked correction holds both arms and returns feedback.
When motion_smoothing is enabled, each existing GPT EEF target uses eased
Cartesian position and shortest-arc orientation interpolation over exactly its
requested controls: gentle start and slowdown near the target. No new stage,
waiting control or extra step is added. Final arm joint commands are filtered
across acknowledged WALL/GPT actions and rechecked by the same collision guard.
Gripper commands keep their original timing. If the allocated steps cannot
reach the target after smoothing, preflight rejects rather than jumping there.

Avoid unintended collisions along the WHOLE motion, not only at its endpoint.
Use RGBD and grasp feedback to consider both arms, fingers and the full extent
of a held object, including a board's far end. Preserve already placed objects
and structures. Clear obstacles before lateral travel or rotation, leave visible
clearance, and use slower/shorter movement near contact. Only intentional task
contact is acceptable. If load rises unexpectedly or objects/supports shift,
stop pushing and reassess before continuing. A cuRobo pass does NOT certify
clearance from scene objects, the other arm or the object being carried.
"""


def curobo_instructions(text):
    import re
    text = re.sub(r'Per control step \(25 Hz, 0\.04 s\), EEF/edit uses bounded local IK:.*?numerical preflight verifies the allocation\.',
                  CUROBO_GUIDE.strip(), text, flags=re.S)
    text = re.sub(r'- EEF corrections use local damped least-squares IK,.*?inspect after-execution RGB and measured pose\.',
                  '- '+CUROBO_GUIDE.strip(), text, flags=re.S)
    return text.replace('bounded robot-only IK checks every node',
                        'shared cuRobo IK checks every node').replace(
                        'Actual motion uses bounded local IK.', 'Actual motion uses shared cuRobo IK.')


class WallCodexPolicy(CodexPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=self.workspace/'SKILL.md'
        text=curobo_instructions(path.read_text())
        path.write_text(text if WALL_GUIDE in text else text+'\n'+WALL_GUIDE)

    def customize_prompt(self,prompt):
        prompt=curobo_instructions(prompt)
        # Update the per-episode manuals as well, before the GPT thread starts.
        # Generic pi0.5/DLS workspaces and running episodes remain untouched.
        for name in ('SKILL.md','eef_control.md'):
            for path in self.agent_workspace.rglob(name):
                text=curobo_instructions(path.read_text())
                if name=='SKILL.md' and WALL_GUIDE not in text:text+='\n'+WALL_GUIDE
                path.write_text(text)
        import sys
        card_path = Path(os.environ['PRIORA_TASK_CARD'])
        card = load_card(card_path, os.environ['PRIORA_TASK'])
        template = Path(__file__).with_name('teacher.md').read_text()
        base = template.replace('{EPISODE_CONTROLLER}', str(self.workspace.parent))
        base = base.replace('{EPISODE_PYTHON}', sys.executable)
        if self.whole_trial_required():
            base += '\nFor this layout, test the complete task on WALL before setting a subtask.\n'
        (self.workspace/'official_card.json').write_text(json.dumps(card, ensure_ascii=False, indent=2))
        return base + '\n\n' + card['text'] + '\n'

    @staticmethod
    def whole_trial_required():
        return os.getenv('WALL_REQUIRE_WHOLE_TASK_TRIAL') == '1'

    def customize_tool_specs(self,specs):
        specs=copy.deepcopy(specs)
        next(s for s in specs if s['name']=='pi05_infer')['description']=(
            'Infer a fresh WALL H32 absolute EEF proposal with the complete native task by default, or the explicitly selected subtask. '
            'No initial wall_set_subtask call is required. '
            'The tool retains its legacy name. No simulation step is executed.')
        execute=next(s for s in specs if s['name']=='robodojo_execute')
        execute['inputSchema']['properties']['response']['properties']['steps']['description']=(
            'WALL student 1..16, GPT eef 1..30, GPT edit 1..5. Native termination may stop earlier.')
        specs.append(dict(type='function',name='wall_set_subtask',description='Optionally select a concise WALL subtask. Single-object tasks start with the complete native task via pi05_infer; prefer action correction and decompose only after clear semantic intent errors. Multiple-object tasks may use coherent object/category goals when needed. Keep the current goal by default; once decomposed, change for the next completed-stage goal or state-grounded recovery. Do not rewrite on ordinary execution failure alone. Use after start/execute, before inference. No motion.',
            inputSchema=dict(type='object',properties=dict(goal=dict(type='string')),required=['goal'],additionalProperties=False)))
        return specs
