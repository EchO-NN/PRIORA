"""Account quota snapshots from the same app-server that owns the episode."""
import time


class Quota:
    def __init__(self, transport, provider, floor):
        self.transport = transport
        self.floor = int(floor) if provider == 'openai' and floor is not None else None
        if self.floor is not None and not 0 <= self.floor < 100:
            raise ValueError('PRIORA_QUOTA_FLOOR must be in [0, 99]')
        self.remaining = None
        self.checked = None

    def update(self, result):
        bucket = (result.get('rateLimitsByLimitId') or {}).get('codex')
        if not bucket:
            bucket = result.get('rateLimits') or {}
            if bucket.get('limitId') not in (None, 'codex'):
                return
        used = [bucket[k]['usedPercent'] for k in ('primary', 'secondary')
                if isinstance((bucket.get(k) or {}).get('usedPercent'), (float, int))]
        if used:
            self.remaining = 100 - max(used)
            self.checked = time.monotonic()

    def exhausted(self):
        if self.floor is None:
            return False
        if self.checked is None or time.monotonic() - self.checked >= 30:
            self.update(self.transport.request('account/rateLimits/read', {}, 30))
            if self.remaining is None:
                raise RuntimeError('Codex did not supply the selected account quota')
        return self.remaining <= self.floor

    def wait(self, keepalive):
        while self.exhausted():
            if keepalive:
                keepalive()
            time.sleep(30)
