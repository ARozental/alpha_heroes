import json,sys,collections
f=sys.argv[1]
types=collections.Counter(); bytes_=collections.Counter()
blk=collections.Counter(); blkb=collections.Counter()
models=collections.Counter(); compact=0; ts0=ts1=None; userprompts=0; userchars=0; asstext=0
sidechain=0
for line in open(f,errors='replace'):
    try: d=json.loads(line)
    except: continue
    t=d.get('type'); types[t]+=1; bytes_[t]+=len(line)
    if d.get('isSidechain'): sidechain+=1
    if d.get('isCompactSummary'): compact+=1
    ts=d.get('timestamp')
    if ts: ts0=ts0 or ts; ts1=ts
    m=d.get('message')
    if isinstance(m,dict):
        if m.get('model'): models[m['model']]+=1
        c=m.get('content')
        if isinstance(c,str):
            blk[t+':str']+=1; blkb[t+':str']+=len(c)
            if t=='user': userprompts+=1; userchars+=len(c)
        elif isinstance(c,list):
            for b in c:
                if not isinstance(b,dict): continue
                k=t+':'+str(b.get('type'))
                blk[k]+=1; blkb[k]+=len(json.dumps(b))
                if t=='user' and b.get('type')=='text': userprompts+=1; userchars+=len(b.get('text',''))
                if t=='assistant' and b.get('type')=='text': asstext+=len(b.get('text',''))
print(f, ts0, ts1)
print('types', dict(types))
print('MB by type', {k: round(v/1e6,1) for k,v in bytes_.items()})
print('blocks', dict(blk))
print('MB by block', {k: round(v/1e6,2) for k,v in blkb.items()})
print('models', dict(models)); print('compact summaries', compact, 'sidechain', sidechain)
print('user text blocks', userprompts, 'chars', userchars, 'assistant text chars', asstext)
