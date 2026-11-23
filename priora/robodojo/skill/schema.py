from ..execution_limits import HYBRID_STEP_LIMITS, DIRECT_EEF_MAX_STEPS


def _obj(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def response_schema(request_id=None):
    number = {"type": "number"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    vector4 = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    string = {"type": "string"}
    progress = _obj(dict(
        verified_completed={"type": "array", "items": string},
        currently_attempting=string,
        remaining={"type": "array", "items": string},
    ))
    assessment = _obj(dict(
        task_progress=progress,
        current_subgoal=string,
        execution_status={"type": "string", "enum": [
            "not_started", "progressing", "failed", "uncertain", "recovered"]},
        execution_evidence=string,
        expected_next_intent=string,
        predicted_next_intent=string,
        intent_status={"type": "string", "enum": ["aligned", "misaligned", "uncertain"]},
        intent_evidence=string,
    ))
    identifier = string if request_id is None else {"type": "string", "enum": [request_id]}
    schema = _obj(dict(
        request_id=identifier,
        mode={"type": "string", "enum": ["student", "edit", "eef", "stop"]},
        steps={"type": "integer", "minimum": 1, "maximum": max(HYBRID_STEP_LIMITS.values()),
               "description": "Choose the execution length: student/eef 1..30, edit 1..5. Native termination may stop earlier."},
        reason=string,
        edit=_obj(dict(delta_position=vector3, delta_rotation_vector=vector3,
                       gripper={"type": "string", "enum": ["keep", "open", "closed"]})),
        target=_obj(dict(position=vector3, quaternion_wxyz=vector4,
                         gripper_closed={"type": "boolean"})),
        assessment=assessment,
    ))
    for field in ('edit', 'target'):
        arm = schema['properties'][field]
        schema['properties'][field] = _obj(dict(left=arm, right=arm))
    schema['properties']['waypoints'] = {
        'type': 'array',
        'maxItems': HYBRID_STEP_LIMITS['eef'],
        'description': (
            'EEF nodes in execution order; [] means one target (also use [] in other modes). '
            'Node steps sum to steps; target equals the last node. Follow the action '
            'contract for bounds, preflight/arrival checks and immediate gripper timing.'),
        'items': _obj(dict(
            label={'type': 'string', 'minLength': 1, 'maxLength': 80,
                   'description': 'Short English stage label, e.g. approach, close, lift.'},
            steps={'type': 'integer', 'minimum': 1, 'maximum': HYBRID_STEP_LIMITS['eef']},
            target=schema['properties']['target'],
        )),
    }
    schema['required'].append('waypoints')
    return schema


def direct_response_schema():
    target = response_schema()['properties']['target']
    return _obj(dict(request_id={'type': 'string'},
        mode={'type': 'string', 'enum': ['eef']},
        steps={'type': 'integer', 'minimum': 1, 'maximum': DIRECT_EEF_MAX_STEPS}, reason={'type': 'string'},
        target=target))
