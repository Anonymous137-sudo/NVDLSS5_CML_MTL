"""Bound an experimental native Core ML probe; preserve its last reached stage."""
import argparse
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('executable', type=Path)
p.add_argument('model', type=Path)
p.add_argument('features', type=Path)
p.add_argument('report', type=Path)
p.add_argument('--seconds', type=float, default=300)
p.add_argument('--pre-block', action='store_true')
a = p.parse_args()
if a.seconds <= 0:
    p.error('seconds must be positive')
cmd = [str(a.executable.resolve()), str(a.model.resolve()), str(a.features.resolve()), str(a.report.resolve())]
if a.pre_block:
    cmd.append('--pre-block')
started = time.monotonic()
with a.report.with_suffix('.runtime.log').open('w') as log:
    try:
        result = subprocess.run(cmd, stdout=log, stderr=log, timeout=a.seconds)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        report = json.loads(a.report.read_text()) if a.report.exists() else {}
        report['lastStageBeforeTimeout'] = report.get('stage', 'not reported')
        report['stage'] = 'timedOut'
        report['timeoutSeconds'] = a.seconds
        report['elapsedSeconds'] = time.monotonic() - started
        report['outcome'] = 'Diagnostic deadline reached. No inference rate or ANE utilization is established by compilation.'
        a.report.write_text(json.dumps(report, indent=2))
        raise SystemExit(124)
