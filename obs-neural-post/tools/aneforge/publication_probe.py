"""Exhaustive on-device E4M3 numerical parity over all finite FP16 inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--aneforge', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--publication', choices=['compact','magic'], default='compact')
a = p.parse_args()
os.environ['ANEFORGE_DYLIB'] = str(a.library.resolve())
sys.path.insert(0, str(a.aneforge.resolve()))
import aneforge as af
from nr_ops import e4m3, e4m3_magic

a.output.mkdir(parents=True, exist_ok=True)
x = af.input((1, 64, 1, 1024))
operation=e4m3_magic if a.publication=='magic' else e4m3
model = af.compile(operation(x), opt=0, build_dir=str(a.output/('e4m3-'+a.publication)))
try:
    source = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(1,64,1,1024)
    finite = np.isfinite(source)
    source = np.where(finite, source, 0).astype(np.float16)
    actual = model(source).astype(np.float16)
    magnitude = np.minimum(abs(source.astype(np.float32)), 448)
    exponent = np.floor(np.log2(np.maximum(magnitude, 2**-6)))
    step = np.where(magnitude < 2**-6, 2**-9, 2.**(exponent-3))
    reference = (np.sign(source)*np.rint(magnitude/step)*step).astype(np.float16)
    indices=np.flatnonzero(actual.reshape(-1)!=reference.reshape(-1))
    cases=[dict(input=float(source.reshape(-1)[i]),actual=float(actual.reshape(-1)[i]),expected=float(reference.reshape(-1)[i])) for i in indices[:32]]
    report = dict(firstMismatches=cases,publication=a.publication,scope='All finite FP16 inputs; NaN/Inf excluded, signed-zero differences reported separately',
        finiteCases=int(finite.sum()), mismatches=int(np.count_nonzero(actual!=reference)),
        maxAbsError=float(np.max(abs(actual-reference))),
        signedZeroDifferences=int(np.count_nonzero((actual==0)&(reference==0)&
                                                 (np.signbit(actual)!=np.signbit(reference)))))
    (a.output/('e4m3-'+a.publication+'-parity.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if report['mismatches']:
        raise RuntimeError('ANE E4M3 publication differs from reference')
finally:
    model.release()
