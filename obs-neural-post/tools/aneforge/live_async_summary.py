"""Summarize bounded runtime queues and host-stage overlap from live evidence."""
import argparse
import json
import statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot',type=Path)
a=p.parse_args();d=json.loads(a.snapshot.read_text())
rows=[r for r in d.get('recentFrames',[]) if r.get('publicationState','published')=='published']
nr=[(r['nrStartedAt'],r['completedAt'],r['sequence']) for r in rows if 'nrStartedAt' in r]
mfg=[(r['mfgStartedAt'],r['publishedAt'],r['sequence']) for r in rows if 'mfgStartedAt' in r]
overlaps=[max(0,min(b,y)-max(a,x))*1000 for a,b,seq in nr for x,y,other in mfg if other!=seq]
queue=[r['mfgQueueMS'] for r in rows if 'mfgQueueMS' in r]
summary={k:d.get(k) for k in ['execution','captureRunning','captureError','completed','failed','error',
 'processedFPS','processingMedianMS','generatedFrames','mfgFailures','mfgError','mfgSuperseded','playbackQueueDepth']}
summary.update(sampledPublishedRows=len(rows),generatedInRows=sum(r.get('generated',0) for r in rows),
 nrMFGHostOverlapMS=sum(overlaps),overlappingHostIntervals=sum(v>0 for v in overlaps),
 mfgQueueMedianMS=statistics.median(queue) if queue else None,
 mfgQueueMaxMS=max(queue) if queue else None,
 scope='Host-stage intervals only; not proof of simultaneous GPU kernel execution',
 settings=d.get('settings',{}),nativePreview=d.get('nativePreview',{}))
print(json.dumps(summary,indent=2))
