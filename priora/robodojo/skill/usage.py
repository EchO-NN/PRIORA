from ..io import write_json


def write_usage(policy, usage, token_limit, model, effort):
    provider = getattr(policy, 'provider', None)
    from ..settings import PROVIDER
    provider = provider or PROVIDER
    segments = getattr(policy, '_usage_segments', {})
    baseline = getattr(policy, '_usage_baselines', {}).get(policy.thread_id, {})
    adjusted = dict(usage, total={key:value-baseline.get(key, 0)
        for key,value in usage.get('total', {}).items()})
    if any(value < 0 for value in adjusted['total'].values()):
        raise RuntimeError('Fork usage counter is below its inherited baseline; preserve raw receipts')
    segments[policy.thread_id] = dict(provider=provider, usage=adjusted,
        inherited_usage_baseline=baseline, native_total=usage.get('total', {}))
    policy._usage_segments = segments
    totals = {}
    for segment in segments.values():
        for key, value in segment['usage'].get('total', {}).items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0)+value
    combined = dict(usage, total=totals)
    write_json(policy.workspace.parent/'token_usage.json', dict(model=model, effort=effort,
        provider=provider, usage=combined, total_tokens_include_cached_input=True,
        configured_limit=token_limit, segments=segments))
    return totals
