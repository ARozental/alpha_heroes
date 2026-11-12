import re,sys,os
src,prefix,target=sys.argv[1],sys.argv[2],int(sys.argv[3])
txt=open(src).read()
header,body=txt.split('\n',1)
ents=re.split(r'(?=\n\[\d{4}-\d\d-\d\d \d\d:\d\d\] )',body)
chunks=[];cur=[];size=0
for e in ents:
    if size>=target and (e.startswith('\n[') and '] USER:' in e[:40] or size>=target+25000):
        chunks.append(cur);cur=[];size=0
    cur.append(e);size+=len(e)
if cur:
    if chunks and size<target*0.2: chunks[-1].extend(cur)
    else: chunks.append(cur)
for i,c in enumerate(chunks,1):
    t=''.join(c)
    ts=re.findall(r'\n\[(\d{4}-\d\d-\d\d \d\d:\d\d)\]',t)
    fn=f'{prefix}_part{i}of{len(chunks)}.md'
    with open(fn,'w') as f:
        f.write(f'{header} — part {i} of {len(chunks)}, {ts[0]} → {ts[-1]}\n{t}')
    print(fn,len(t),ts[0],'→',ts[-1])
