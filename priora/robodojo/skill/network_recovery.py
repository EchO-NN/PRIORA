"""Same-thread wakeup eligibility. No simulator restart or action replay."""
import json

NETWORK_CONTINUE_DELAYS = (10, 30, 60)
# Repeat the final capacity delay until the provider becomes available. This
# schedule is not an attempt budget; account/auth failures are excluded below.
CAPACITY_CONTINUE_DELAYS = (5,)
NETWORK_RECOVERY_VERSION = 'same_thread_capacity_wait_v2'
ACTIVE_CAPACITY_GRACE_WINDOWS = 2
ACTIVE_TURN_GRACE_WINDOWS = ACTIVE_CAPACITY_GRACE_WINDOWS


def _matching_turn(response, thread_id, turn_id, state):
    """Reject incomplete, stale or contradictory snapshots without guessing."""
    if not isinstance(response, dict) or not isinstance(response.get('thread'), dict):
        return None
    thread = response['thread']
    status, turns = thread.get('status'), thread.get('turns')
    if (thread.get('id') != thread_id or not isinstance(status, dict)
            or status.get('type') != state or not isinstance(turns, list) or not turns
            or not isinstance(turns[-1], dict) or turns[-1].get('id') != turn_id):
        return None
    return turns[-1]


def active_turn(response, thread_id, turn_id):
    """Read-only evidence of the same live turn, including silent model work."""
    turn = _matching_turn(response, thread_id, turn_id, 'active')
    return turn is not None and turn.get('status') == 'inProgress'


def active_capacity_turn(response, thread_id, turn_id):
    """Compatibility name for the former capacity-only waiting contract."""
    return active_turn(response, thread_id, turn_id)


def closed_terminal_turn(response, thread_id, turn_id):
    """Recover a lost completion notification, never an active/interrupted turn."""
    turn = _matching_turn(response, thread_id, turn_id, 'idle')
    return turn if turn is not None and turn.get('status') in ('completed', 'failed') else None


def is_capacity_error(error):
    text = json.dumps(error, ensure_ascii=False).lower()
    if any(s in text for s in ('quota', 'usage limit', 'insufficient', 'billing',
                               'credit balance', 'unauthorized', 'authentication', 'invalid api key')):
        return False
    return any(s in text for s in ('temporary capacity limit', 'serveroverloaded',
                                   'selected model is at capacity', 'ratelimitexceeded',
                                   '429 too many requests', 'rate limit exceeded',
                                   '"httpstatuscode": 429'))


def is_network_error(error):
    text = json.dumps(error, ensure_ascii=False).lower()
    # Account/config/safety failures must not become automatic model calls.
    if any(s in text for s in ('usage limit', 'quota', 'unauthorized', 'authentication',
                               'invalid api key', 'permission denied', 'safety',
                               'five rejected', 'exceeds 5 cm')):
        return False
    return is_capacity_error(error) or any(s in text for s in ('stream disconnected', 'responsestreamdisconnected',
        'error decoding response body', 'network error', 'connection reset',
        'connection closed', 'connection timed out', 'workspace routing discovery timed out',
        'tls close_notify', 'upstream service is temporarily unavailable',
        '上游服务暂时不可用'))


def closed_network_turn(response, thread_id, turn_id):
    """Timeout probe is read-only. Never wake an active or interrupted turn."""
    turn = closed_terminal_turn(response, thread_id, turn_id)
    return turn if turn is not None and turn.get('status') == 'failed' and is_network_error(turn.get('error')) else None
