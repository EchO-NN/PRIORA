"""One GPU weight copy; separately owned frame memory and RNG for every episode."""
import argparse
import copy
import json
import os
from pathlib import Path
import socketserver
import threading
import time

import numpy as np

from ..robodojo_server.protocol import VERSION, receive_packet, send_packet

CHECKPOINT = os.environ.get('PRIORA_WALL_CHECKPOINT', '')
SHA256 = '771b82f375eedcd55a24f5a5327a2dab6b84daf142be025f7faa1fd1146a8940'
REVISION = '39fa65b9c9e95b0316db0fc9dbe41211e3a0d4cc'
METADATA = dict(backend='WALL-OSS/native-EEF', config='wall_oss_lite_robodojo',
    checkpoint=CHECKPOINT, checkpoint_sha256=SHA256, source_revision=REVISION,
    action_horizon=32, action_dim=16, action_type='absolute_eef_wxyz',
    student_max_steps=16, correction_max_steps=30, memory_max_advance=30)


class Engine:
    def __init__(self):
        import torch
        from wall_x.serving.robodojo.policy import RoboDojoPolicyConfig
        from wall_x.serving.robodojo.memory_policy import KeyframeMemoryRoboDojoPolicy
        from ..io import sha256
        if not CHECKPOINT or sha256(Path(CHECKPOINT)/'model.safetensors') != SHA256:
            raise ValueError('Set PRIORA_WALL_CHECKPOINT to the pinned WALL RoboDojo checkpoint')
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        # replan_steps controls protocol advance validation. WALL execution stays
        # at <=16; allow the existing GPT correction budget of <=30 between reads.
        config = RoboDojoPolicyConfig(checkpoint_path=CHECKPOINT, norm_key='ex_normal',
            model_device='cuda:0', replan_steps=30, action_chunk_length=32, image_resize='serving')
        self.policy = KeyframeMemoryRoboDojoPolicy(config)
        model = self.policy.model.model
        assert model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        self.sessions = {}
        self.lock = threading.Lock()

    def request(self, op, args):
        import torch
        if op == 'metadata':
            return dict(METADATA)
        key = args['session_id']
        if not isinstance(key, str) or not key.isalnum():
            raise ValueError('Expected episode UUID context key')
        with self.lock:
            if op == 'reset':
                if key in self.sessions:
                    raise ValueError('Do not reuse a WALL episode session')
                self.sessions[key] = dict(state=self.policy.reset(), frames={}, tick=-1, prior=None)
                return dict(reset=True, session_id=key)
            if op == 'close':
                self.sessions.pop(key, None)
                return dict(closed=True)
            if op != 'infer':
                raise ValueError('Unknown WALL operation')
            s = self.sessions[key]; tick = args['tick']; obs = args['observation']
            if type(tick) is not int or tick < s['tick']:
                raise ValueError('WALL physical time cannot go backwards')
            # Reinference at the same physical observation must not insert that
            # frame twice into memory or advance RNG/history as if it executed.
            state = copy.deepcopy(s['prior'] if tick == s['tick'] else s['state'])
            for frame in args['frames']:
                s['frames'][frame['sim_frame']] = frame['face_view']
            oldest = max(0, tick-self.policy.memory_config.ring_span)
            frame = lambda i: dict(sim_frame=i, face_view=s['frames'][i])
            previous = state['prev_sim_frame']
            current = dict(obs, task_id=args['task'], sim_frame=tick,
                kfmem_history=[frame(i) for i in range((previous or 0)+1, tick+1)],
                kfmem_client_state_v1={**state, 'context_key':key,
                    'ring':[frame(i) for i in range(oldest, tick+1)],
                    'mem':[frame(i) for i in state['mem_frame_ids']]})
            begin=time.monotonic()
            with torch.inference_mode():
                self.policy.update_obs(current)
                assert self.policy._pending[1] == obs['instruction']
                result = self.policy.get_action()
            torch.cuda.synchronize()
            actions=np.asarray([np.r_[a['left_ee_pose'],a['left_ee_joint_state'],
                a['right_ee_pose'],a['right_ee_joint_state']] for a in result['actions']],np.float32)
            if actions.shape != (32,16) or not np.isfinite(actions).all():
                raise ValueError('Invalid native WALL proposal')
            s.update(prior=state, state=result['kfmem_state'], tick=tick)
            needed = set(range(oldest,tick+1)) | set(state['mem_frame_ids']) | set(s['state']['mem_frame_ids'])
            s['frames'] = {i:v for i,v in s['frames'].items() if i in needed}
            memory = {k:v for k,v in s['state'].items() if k != 'rng_state'}
            inference_seconds = time.monotonic()-begin
            # The 24 GiB peer shares its GPU with two simulators. Release only
            # unused allocator blocks; all model weights remain GPU resident.
            if os.environ.get('WALL_RELEASE_IDLE_CACHE') == '1':
                torch.cuda.empty_cache()
            return dict(actions=actions, seconds=time.monotonic()-begin,
                received_prompt=obs['instruction'], memory=memory, session_id=key,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                inference_seconds_without_cache_release=inference_seconds)


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=19660);a=p.parse_args()
    engine=Engine()
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(86400)
            try:
                while True:
                    request=receive_packet(self.request)
                    reply=dict(version=VERSION,request_id=request['request_id'])
                    try:
                        if request.get('version') != VERSION:raise ValueError('Invalid protocol')
                        reply.update(ok=True,result=engine.request(request['op'],request.get('args',{})))
                    except Exception as error:
                        import traceback;traceback.print_exc()
                        reply.update(ok=False,error=f'{type(error).__name__}: {error}')
                    send_packet(self.request,reply)
            except (EOFError,ConnectionError,TimeoutError):pass
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address=True
        daemon_threads=True
    with Server(('127.0.0.1',a.port),Handler) as server:
        print(json.dumps(dict(event='ready',port=a.port,metadata=METADATA)),flush=True)
        server.serve_forever()


if __name__ == '__main__':main()
