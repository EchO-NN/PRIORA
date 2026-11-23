"""Verify archived native-timeout evidence for the replay renderer."""
import json
from pathlib import Path

from .io import sha256

SCHEMA = 'robodojo.rpc_idle_timeout_failure.v1'
REASON = 'simulator_rpc_idle_timeout_900s'
SIDECAR = 'evaluation_adjudication.json'


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def verified_adjudication(archive):
    archive = Path(archive)
    path = archive/SIDECAR
    if not path.exists():
        return None
    row = read(path)
    if (row.get('schema') != SCHEMA or row.get('reason') != REASON
            or row.get('evaluation_success') is not False or row.get('retry') is not False
            or row.get('native_complete') is not False or row.get('native_success') is not None
            or row.get('native_score') is not None or not row.get('original_sha256')):
        raise ValueError('Invalid timeout adjudication')
    for relative, digest in row['original_sha256'].items():
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or sha256(archive/p) != digest:
            raise ValueError('Adjudicated evidence changed: '+relative)
    return row
