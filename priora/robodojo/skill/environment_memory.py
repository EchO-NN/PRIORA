"""Host-owned execution facts and a fresh, read-only rollout state reference.

This ledger never invokes a robot service, changes arguments, or repairs actions.
The host's sandbox permissions make the audit directory read-only to the policy.
"""
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from ..io import write_json


VERSION = 'environment_memory_v1'
COMMANDS = ('cat', 'ls', 'pwd', 'rg', 'find', 'head', 'tail', 'sed', 'jq', 'python3')
NEXT_CALL_FIELDS = ('tool', 'task', 'observation_path', 'proposal_path', 'output_dir', 'response')


def _public_error(value):
    """Bound diagnostics and avoid copying common credential forms into memory."""
    if isinstance(value, dict):
        value = value.get('message', value.get('code', 'Tool returned an error'))
    text = str(value)
    text = re.sub(r'(?i)\bBearer\s+[^\s,;]+', 'Bearer [redacted]', text)
    text = re.sub(r'\b(?:sk-|hf_)[A-Za-z0-9_-]+', '[redacted]', text)
    text = re.sub(r'(?i)(api[_-]?key|access[_-]?token|password|secret)([\s\"\x27]*[:=][\s\"\x27]*)[^\s,;\"\x27]+',
                  r'\1\2[redacted]', text)
    text = re.sub(r'(https?://)[^/\s:@]+:[^/\s@]+@', r'\1[redacted]@', text)
    return text[:320]


class EnvironmentMemory:
    def __init__(self, agent_workspace, method='wall_plus_gpt'):
        self.agent = Path(agent_workspace).resolve()
        self.audit = self.agent.parent
        self.controller = self.audit.parent
        self.method = method
        self.path = self.audit/'ENVIRONMENT_MEMORY.json'
        self.markdown_path = self.audit/'ENVIRONMENT_MEMORY.md'
        self.runtime_path = self.audit/'runtime_state.json'

    def initialize(self):
        """Probe a small explicit allowlist once; return facts for prompt injection."""
        self.audit.mkdir(parents=True, exist_ok=True)
        system = platform.system()
        bash = '/bin/bash' if os.access('/bin/bash', os.X_OK) else None
        facts = dict(version=VERSION, evaluation_method=self.method,
            platform_system=system, os_name=os.name,
            command_syntax='POSIX' if os.name == 'posix' else 'platform native',
            preferred_shell=bash,
            # Preserve a virtualenv interpreter symlink: resolving it can select
            # the base environment instead of the verified active interpreter.
            python_executable=os.path.abspath(sys.executable),
            available_commands={name: shutil.which(name) for name in COMMANDS},
            agent_cwd=str(self.agent), controller_output=str(self.controller),
            workspace_path=str(self.agent/'workspace.json'),
            environment_memory_path=str(self.path),
            runtime_state_path=str(self.runtime_path),
            availability_scope='Host probes only; sandbox permissions may further restrict access.')
        shell_note = (
            'This is Linux/POSIX. Use POSIX commands; do not use PowerShell Get-Content, '
            'cmd.exe, drive letters, or Windows paths.' if system == 'Linux' else
            f'Host platform is {system}; use the declared command syntax and available tools.')
        guidance = (
            '# Verified execution environment memory\n\n' + shell_note + '\n'
            + 'Python: ' + facts['python_executable'] + '\n'
            'workspace.json in the agent cwd contains exact artifact paths; '
            '../ENVIRONMENT_MEMORY.json contains verified commands and host facts. '
            'Reuse these facts rather than repeating environment discovery. '
            'Use the newest environment_state and exact next_call paths/IDs; '
            'observation_path is the JSON file, not its directory. '
            'next_call.response is a placeholder: supply your own assessment/decision. '
            'Read-only queries do not change the control phase. Correct rejected '
            'arguments without resetting or replaying acknowledged motion. '
            'Keep notes in the writable agent workspace; host evidence is read-only.\n')
        # Preserve stable files when initialized again for the same workspace.
        if not self.path.exists() or self.path.read_text() != json.dumps(
                facts, indent=2, ensure_ascii=False, allow_nan=False):
            write_json(self.path, facts)
        if not self.markdown_path.exists() or self.markdown_path.read_text() != guidance:
            self.markdown_path.write_text(guidance)
        return guidance

    @staticmethod
    def _same_request(request, episode, tick):
        return (isinstance(request, dict) and request.get('request_id') is not None
                and request.get('episode_id', episode) == episode
                and request.get('step_id', tick) == tick)

    def refresh(self, rollout, *, tool=None, packet=None):
        """Publish only current authoritative identifiers, not large observations."""
        phase = getattr(rollout, 'phase', 'start')
        episode = getattr(rollout, 'episode', getattr(rollout, 'episode_id', None))
        tick = getattr(rollout, 'tick', None)
        observation = getattr(rollout, 'observation_path', None)
        expected = rollout.next_call() if phase != 'done' else None
        next_call = ({key: expected[key] for key in NEXT_CALL_FIELDS if key in expected}
                     if isinstance(expected, dict) else None)
        state = dict(version=VERSION, evaluation_method=self.method,
            episode_id=episode, phase=phase, step_id=tick,
            observation_path=str(observation) if observation is not None else None,
            observation_current=bool(episode is not None and observation is not None
                                     and phase not in ('start', 'done')),
            next_call=next_call,
            model_authored_response_required=bool(phase in ('execute', 'act')))
        packet = packet if isinstance(packet, dict) else {}
        if phase in ('execute', 'act'):
            request = getattr(rollout, 'request', None)
            if self._same_request(request, episode, tick):
                state['request_id'] = request['request_id']
            elif (tool in (('pi05_infer',) if phase == 'execute' else
                           ('robodojo_start', 'robodojo_act'))
                  and 'error' not in packet and packet.get('success') is not False
                  and packet.get('episode_id') == episode and packet.get('step_id') == tick
                  and self._same_request(packet, episode, tick)):
                state['request_id'] = packet['request_id']
            if phase == 'execute' and next_call and 'proposal_path' in next_call:
                state['proposal_path'] = next_call['proposal_path']
        if tool is not None:
            failed = (bool(packet.get('error')) or bool(packet.get('error_type'))
                      or packet.get('success') is False)
            result = dict(tool=str(tool), success=not failed)
            if failed:
                if packet.get('error_type') or packet.get('error_code'):
                    result['error_code'] = _public_error(packet.get('error_type') or packet['error_code'])
                result['error_message'] = _public_error(packet.get('error', packet.get('message',
                                                              'Tool returned an error')))
            state['tool_result'] = result
        self.audit.mkdir(parents=True, exist_ok=True)
        write_json(self.runtime_path, state)
        return state
