import json,sys,re,os,collections
from datetime import datetime
from zoneinfo import ZoneInfo
TZ=ZoneInfo('America/Chicago')
MIN_TS=sys.argv[4] if len(sys.argv)>4 else ''
def loc(ts):
    if not ts: return ''
    try: return datetime.fromisoformat(ts.replace('Z','+00:00')).astimezone(TZ).strftime('%Y-%m-%d %H:%M')
    except Exception: return ts[:16]
src=sys.argv[1]; out=sys.argv[2]; label=sys.argv[3]
def short(s,n):
    s=s.strip()
    return s if len(s)<=n else s[:n]+f' …[+{len(s)-n} chars]'
entries=[]  # (ts, kind, text)
last_user=None; model=None; pending_tools=collections.Counter()
compacts=[]
def flush_tools(ts):
    global pending_tools
    if pending_tools:
        entries.append((ts,'TOOLS',', '.join(f'{k}×{v}' if v>1 else k for k,v in pending_tools.items())))
        pending_tools=collections.Counter()
for line in open(src,errors='replace'):
    try: d=json.loads(line)
    except: continue
    t=d.get('type'); rawts=d.get('timestamp') or ''
    if MIN_TS and rawts and rawts<MIN_TS: continue
    ts=loc(rawts)
    m=d.get('message')
    if not isinstance(m,dict): continue
    c=m.get('content')
    if t=='user':
        if d.get('isCompactSummary'):
            flush_tools(ts)
            txt=c if isinstance(c,str) else ' '.join(b.get('text','') for b in c if isinstance(b,dict))
            compacts.append((ts,txt)); entries.append((ts,'COMPACTION','(context was compacted here; summary stored separately)'))
            continue
        blocks=[c] if isinstance(c,str) else [b for b in (c or []) if isinstance(b,dict)]
        for b in blocks:
            if isinstance(b,str): s=b
            elif b.get('type')=='text': s=b.get('text','')
            elif b.get('type')=='tool_result':
                rc=b.get('content'); rs=rc if isinstance(rc,str) else ' '.join(x.get('text','') for x in (rc or []) if isinstance(x,dict))
                if "doesn't want to proceed" in rs or 'user rejected' in rs.lower() or 'The user denied' in rs:
                    flush_tools(ts); entries.append((ts,'USER-REJECTED-TOOL',short(rs,1500)))
                continue
            else: continue
            s=s.strip()
            if not s: continue
            if s.startswith('<local-command-caveat') : continue
            if s.startswith('<command-name>'):
                cm=re.search(r'<command-name>(.*?)</command-name>.*?<command-args>(.*?)</command-args>',s,re.S)
                flush_tools(ts); entries.append((ts,'SLASH',(cm.group(1)+' '+cm.group(2)).strip() if cm else short(s,200))); continue
            if s.startswith('<local-command-stdout>'):
                entries.append((ts,'SLASH-OUT',short(re.sub(r'</?local-command-stdout>','',s),300))); continue
            if s.startswith('<task-notification>'):
                summ=re.search(r'<summary>(.*?)</summary>',s,re.S); ev=re.search(r'<event>(.*?)</event>',s,re.S)
                st=re.search(r'<status>(.*?)</status>',s,re.S)
                txt=' '.join(x.group(1).strip() for x in (summ,st,ev) if x)
                flush_tools(ts); entries.append((ts,'NOTIFY',short(re.sub(r'\s+',' ',txt or s),160))); continue
            if 'cross-session-message' in s[:300]:
                fm=re.search(r'from-name="([^"]*)"',s)
                body=re.sub(r'^.*?<cross-session-message[^>]*>','',s,flags=re.S); body=re.sub(r'</cross-session-message>.*$','',body,flags=re.S)
                flush_tools(ts); entries.append((ts,f'PEER-MSG from {fm.group(1) if fm else "?"}',short(body,1000))); continue
            if d.get('isMeta'):
                # other meta: system reminders etc
                if len(s)<400 and not s.startswith('<system-reminder>'):
                    entries.append((ts,'META',short(s,400)))
                continue
            if s.startswith('<system-reminder>'): continue
            if s==last_user: continue
            last_user=s
            flush_tools(ts); entries.append((ts,'USER',short(s,12000)))
    elif t=='assistant':
        mod=m.get('model')
        if mod and mod!='<synthetic>' and mod!=model:
            flush_tools(ts); entries.append((ts,'MODEL',mod)); model=mod
        for b in (c or []):
            if not isinstance(b,dict): continue
            bt=b.get('type')
            if bt=='text':
                s=b.get('text','').strip()
                if s: flush_tools(ts); entries.append((ts,'AI',s))
            elif bt=='tool_use':
                nm=b.get('name'); inp=b.get('input') or {}
                if nm=='SendMessage':
                    flush_tools(ts); entries.append((ts,f'AI→SendMessage to {inp.get("to")}',short(str(inp.get('message') or inp.get('content') or ''),4000)))
                elif nm in ('Agent','Task'):
                    flush_tools(ts); entries.append((ts,f'AI→Subagent ({inp.get("subagent_type","")}: {inp.get("description","")})',short(str(inp.get('prompt','')),2500)))
                elif nm=='AskUserQuestion':
                    flush_tools(ts); entries.append((ts,'AI→AskUserQuestion',short(json.dumps(inp.get('questions'),ensure_ascii=False),2000)))
                elif nm in ('Write',) and str(inp.get('file_path','')).endswith('.md'):
                    pending_tools[f'Write {os.path.basename(str(inp.get("file_path")))}']+=1
                elif nm=='ExitPlanMode':
                    flush_tools(ts); entries.append((ts,'AI→Plan',short(str(inp.get('plan','')),6000)))
                else:
                    pending_tools[nm]+=1
flush_tools('')
with open(out,'w') as f:
    f.write(f'# Digest of session {label} (times America/Chicago)\n\n')
    for ts,k,txt in entries:
        f.write(f'\n[{ts}] {k}: {txt}\n')
with open(out.replace('.md','.compactions.md'),'w') as f:
    for ts,txt in compacts: f.write(f'\n\n######## COMPACTION SUMMARY at {ts}\n{txt}\n')
print(label, 'entries',len(entries),'digest chars',os.path.getsize(out),'compactions',len(compacts),os.path.getsize(out.replace('.md','.compactions.md')))
c=collections.Counter(k.split(' ')[0] for _,k,_ in entries); print(dict(c))
