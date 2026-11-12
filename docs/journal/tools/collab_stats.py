import json,sys,collections,re
from datetime import datetime
from zoneinfo import ZoneInfo
TZ=ZoneInfo('America/Chicago')
P='/Users/alon/.claude_personal/projects/-Users-alon-projects-alpha-heroes/'
sessions={'cd':P+'a62ff69f-8277-4be4-a845-fb7b74052196.jsonl','side':P+'fd6b4705-c93e-4733-95c8-5df67a89337c.jsonl',
 'b1':P+'68ab2bd0-4fe9-4b5a-b8b6-e8f7c73a56ac.jsonl','e5':P+'31529395-f677-4318-b857-085c1c3696a2.jsonl',
 'fork':P+'df787a6f-7188-4dd6-b740-aa3ff9bea8e6.jsonl','sub':P+'68ab2bd0-4fe9-4b5a-b8b6-e8f7c73a56ac/subagents/agent-af5e648493de284d1.jsonl',
 'home':'/Users/alon/.claude/projects/-Users-alon/b608c016-3781-4592-8e77-6274f92fd065.jsonl'}
MIN={'fork':'2026-11-09T17:52'}
def loc(ts): return datetime.fromisoformat(ts.replace('Z','+00:00')).astimezone(TZ)
out={}
daily=collections.defaultdict(lambda: collections.Counter())
for name,f in sessions.items():
    seen=set(); usage=collections.Counter(); models=collections.Counter(); tools=collections.Counter()
    prompts=0; interrupts=0; compacts=0; peer_in=0; peer_out=0; first=last=None; notif=0
    user_ts=[]; asst_ts=[]; model_usage=collections.defaultdict(collections.Counter)
    last_user_prompt=None
    for line in open(f,errors='replace'):
        try: d=json.loads(line)
        except: continue
        ts=d.get('timestamp')
        if not ts: continue
        if name in MIN and ts<MIN[name]: continue
        t=d.get('type'); m=d.get('message')
        if not isinstance(m,dict): continue
        lt=loc(ts); day=lt.strftime('%m-%d')
        first=first or lt; last=lt
        if t=='assistant':
            mid=m.get('id'); mod=m.get('model')
            if mod=='<synthetic>': continue
            if mid and mid not in seen:
                seen.add(mid); u=m.get('usage') or {}
                for k in ('input_tokens','cache_creation_input_tokens','cache_read_input_tokens','output_tokens'):
                    usage[k]+=u.get(k,0) or 0; model_usage[mod][k]+=u.get(k,0) or 0
                models[mod]+=1; daily[day][name+'_turns']+=1; asst_ts.append(lt)
            for b in m.get('content') or []:
                if isinstance(b,dict) and b.get('type')=='tool_use':
                    tools[b.get('name')]+=1; daily[day]['tool_calls']+=1
                    if b.get('name')=='SendMessage': peer_out+=1; daily[day]['peer_msgs']+=1
        elif t=='user':
            if d.get('isCompactSummary'): compacts+=1; continue
            c=m.get('content'); texts=[c] if isinstance(c,str) else [b.get('text','') for b in (c or []) if isinstance(b,dict) and b.get('type')=='text']
            for s in texts:
                s=(s or '').strip()
                if not s: continue
                if 'cross-session-message' in s[:300]: peer_in+=1; continue
                if s.startswith('<task-notification'): notif+=1; continue
                if d.get('isMeta') or s.startswith('<') : continue
                if s.startswith('[Request interrupted'): interrupts+=1; continue
                if s==last_user_prompt: continue
                last_user_prompt=s
                prompts+=1; daily[day]['prompts']+=1; daily[day][name+'_prompts']+=1; user_ts.append(lt)
    # autonomous stretches: gaps between consecutive user prompts with AI turns in between
    user_ts.sort(); gaps=[]
    for a,b in zip(user_ts,user_ts[1:]):
        h=(b-a).total_seconds()/3600
        if h>=3:
            n=sum(1 for x in asst_ts if a<x<b)
            if n>50: gaps.append((a.strftime('%m-%d %H:%M'),round(h,1),n))
    out[name]=dict(first=first.strftime('%Y-%m-%d %H:%M') if first else None,last=last.strftime('%Y-%m-%d %H:%M') if last else None,prompts=prompts,
        assistant_api_turns=len(seen),interrupts=interrupts,compactions=compacts,peer_in=peer_in,peer_out=peer_out,notifications=notif,
        models=dict(models),usage=dict(usage),tools=dict(tools.most_common()),long_autonomous_stretches=sorted(gaps,key=lambda g:-g[1])[:12],
        model_usage={k:dict(v) for k,v in model_usage.items()})
json.dump(out,open('collab_stats.json','w'),indent=1,default=str)
for k,v in out.items():
    print('=====',k); 
    for kk,vv in v.items(): print(' ',kk,':',vv)
print('\nDAILY (local)')
keys=['prompts','cd_prompts','b1_prompts','e5_prompts','fork_prompts','cd_turns','b1_turns','e5_turns','tool_calls','peer_msgs']
print('day   '+' '.join(f'{k[:10]:>10}' for k in keys))
for day in sorted(daily):
    print(day+' '+' '.join(f'{daily[day][k]:>10}' for k in keys))
