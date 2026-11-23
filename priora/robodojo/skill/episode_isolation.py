"""Expose only the active controller and a disposable Codex home to the agent."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sandbox(argv, workspace, env):
    state = Path(os.environ['ROLLOUT_CODEX_STATE_DIR']).resolve(strict=True)
    controller = Path(os.environ['WALL_PRIVATE_CONTROLLER']).resolve(strict=True)
    agent = controller / 'codex_workspace' / 'agent'
    if workspace.resolve().parent != controller:
        raise ValueError('The workspace must belong to this episode')
    private = state / 'codex_home'
    private.mkdir(mode=0o700, exist_ok=True)
    if not os.environ.get('PRIORA_BASE_URL'):
        source = Path(os.getenv('CODEX_HOME', str(Path.home()/'.codex'))) / 'auth.json'
        with source.open('rb') as src, (private/'auth.json').open('xb') as dst:
            os.fchmod(dst.fileno(), 0o600)
            shutil.copyfileobj(src, dst)
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise RuntimeError('Install bubblewrap to isolate each robot episode')
    args = [bwrap, '--die-with-parent', '--new-session', '--unshare-pid', '--unshare-ipc',
            '--unshare-uts', '--ro-bind', '/usr', '/usr', '--ro-bind', '/etc', '/etc',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/run']
    for name in ('bin','sbin','lib','lib64'):
        location = Path('/')/name
        if location.is_symlink():
            args += ['--symlink', os.readlink(location), str(location)]
        elif location.is_dir():
            args += ['--ro-bind', str(location), str(location)]
    resolver = Path('/etc/resolv.conf').resolve()
    if not resolver.is_relative_to('/etc'):
        args += ['--ro-bind', str(resolver), str(resolver)]
    prefix = Path(sys.prefix).resolve()
    python_path = str(Path(sys.executable).parent)
    for runtime in dict.fromkeys((prefix, Path(sys.base_prefix).resolve())):
        if runtime.is_relative_to('/usr'):
            continue
        if runtime in (Path('/'), Path.home()) or controller.is_relative_to(runtime):
            raise ValueError('Use a dedicated Python environment for the controller')
        args += ['--ro-bind', str(runtime), str(runtime)]
    tool = Path(shutil.which(argv[0]) or argv[0]).resolve(strict=True)
    if not tool.is_relative_to('/usr') and not tool.is_relative_to(prefix):
        args += ['--ro-bind', str(tool), str(tool)]
    argv = [str(tool), *argv[1:]]
    args += ['--bind', str(state), str(state), '--dir', '/home/agent',
             '--ro-bind', str(controller), str(controller), '--bind', str(agent), str(agent),
             '--chdir', str(agent)]
    clean = {k:v for k,v in env.items() if k.lower() in
             ('http_proxy','https_proxy','no_proxy','all_proxy','lang','lc_all')}
    if os.environ.get('PRIORA_BASE_URL'):
        clean['PRIORA_API_KEY'] = os.environ['PRIORA_API_KEY']
    clean.update(HOME='/home/agent', USER='agent', LOGNAME='agent', SHELL='/bin/bash',
                 PATH=python_path+':/usr/local/bin:/usr/bin:/bin', CODEX_HOME=str(private), TMPDIR='/tmp')
    with tempfile.TemporaryDirectory(prefix='priora-host-only-') as host_canary:
        probe = ['sh', '-ec', 'test -d "$1" && test ! -e "$2"', 'sh', str(agent), host_canary]
        subprocess.run([*args, '--', *probe], env=clean, check=True, timeout=15)
    (workspace/'isolation.json').write_text(json.dumps(dict(private_codex_home=True, controller=str(controller))))
    return [*args,'--',*argv], clean
