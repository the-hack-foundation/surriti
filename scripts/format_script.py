import sys, json
data = json.load(sys.stdin)
edges = [e for e in data.get('links',[]) if e.get('table')=='relates_to']
nodes = {n['uuid']: n for n in data.get('nodes',[])}
print(f'EDGES: {len(edges)}    ENTITY NODES: {sum(1 for n in nodes.values() if n.get("table")=="entity")}')
print()
print('=== ALL FACT EDGES ===')
for e in sorted(edges, key=lambda x: (nodes.get(x.get('source'),{}).get('name', '?'), x.get('canonical_name') or x.get('name') or '')):
    src = nodes.get(e.get('source'), {}).get('name', '?')
    tgt = nodes.get(e.get('target'), {}).get('name', '?')
    pred = e.get('canonical_name') or e.get('name') or '?'
    status = e.get('status') or 'active'
    inv = e.get('invalid_at') or ''
    fact = (e.get('fact') or '')[:120]
    flags = []
    if inv: flags.append('INVALID')
    if e.get('singleton'): flags.append('singleton')
    if e.get('is_belief'): flags.append('belief')
    sup = e.get('supersedes') or []
    if sup: flags.append(f'supersedes={len(sup)}')
    flag = (' [' + ','.join(flags) + ']') if flags else ''
    print(f'  {src!s:30s} -[{pred}]-> {tgt!s:25s} {status}{flag}')
    if fact:
        print(f'      "{fact}"')
print()
print('=== TRAIT NODES ===')
for n in nodes.values():
    if 'trait' in (n.get('labels') or []):
        print(f'  - {n.get("name")}  summary={(n.get("summary") or "")[:80]!r}')
print()
print('=== ENTITY NAMES ===')
for n in sorted(nodes.values(), key=lambda x: x.get('name','')):
    if n.get('table') == 'entity':
        print(f'  - {n.get("name")}  labels={n.get("labels")}')
