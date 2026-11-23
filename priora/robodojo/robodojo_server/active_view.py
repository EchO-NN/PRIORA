"""Reject the removed camera-repositioning protocol, including legacy requests.

The action schema no longer exposes this feature. Retain a negative guard so
an old prompt cannot smuggle the former exception through runtime validation.
"""


def reject_active_view(response):
    if isinstance(response, dict) and response.get('active_view') is not None:
        raise ValueError('active_view has been removed; camera-only arm movements are not allowed')
