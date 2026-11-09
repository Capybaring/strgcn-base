"""Iterate raw training observations without expanding overlapping tasks."""

def raw_observations(packet):
    if packet.get('kind') == 'tpatch_records':
        for _, _, values, mask in packet['records']:
            yield values, mask
    else:
        for a, _, b, _ in packet['windows']:
            yield packet['values'][a:b], packet['mask'][a:b]
