"""Export the candidate contract; untrained probes stay under build/, not models/."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from model import AppearanceStudent,CONTRACT,WIDTH,HEIGHT

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--width',type=int,default=24,choices=(24,48,64,96))
    p.add_argument('--probe',action='store_true',help='Export reproducible nonidentity untrained fields for placement/parity diagnostics only')
    a=p.parse_args()
    if not a.checkpoint and not a.probe: p.error('Provide trained --checkpoint or explicitly select --probe')
    if a.probe and a.checkpoint: p.error('Probe and trained checkpoint are mutually exclusive')
    if a.probe and 'build' not in a.output.resolve().parts: p.error('Diagnostic probe must remain in build/')
    training = {}
    if a.checkpoint and a.checkpoint.with_suffix('.training.json').exists():
        training=json.loads(a.checkpoint.with_suffix('.training.json').read_text())
        if training.get('contract') != CONTRACT: p.error('Checkpoint decoder contract differs from current model; use its matching exporter')
        if (training.get('diagnostic_only') or training.get('pilot_only')) and 'build' not in a.output.resolve().parts: p.error('Diagnostic trained fixture must remain in build/')
    torch.manual_seed(73); model=AppearanceStudent(width=training.get('width',a.width)).eval()
    if a.checkpoint: model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    else:
        # Nonzero output prevents constant folding from making an empty benchmark.
        torch.nn.init.normal_(model.fields.weight,std=0.015)
        torch.nn.init.normal_(model.fields.bias,std=0.01)
    import coremltools as ct
    traced=torch.jit.trace(model,torch.zeros(1,3,HEIGHT,WIDTH))
    package=ct.convert(traced,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS14,
        inputs=[ct.ImageType(name='rgb',shape=(1,3,HEIGHT,WIDTH),scale=1/255,color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name='appearance_fields',dtype=np.float16)],compute_precision=ct.precision.FLOAT16)
    metadata={'contract':CONTRACT,'training_status':'untrained-probe' if a.probe else ('diagnostic-fixture' if training.get('diagnostic_only') else ('pilot-teacher-distilled' if training.get('pilot_only') else 'checkpoint-unvalidated')),
        'input':{'name':'rgb','shape':[1,3,HEIGHT,WIDTH],'color':'RGB SDR sRGB','range':[0,1],'coverage':'entire 16:9 frame'},
        'output':{'name':'appearance_fields','shape':[1,16,HEIGHT//2,WIDTH//2],'dtype':'float16',
                  'channels':{'affine_rgb_rows':[0,12],'detail_rgb_gain':[12,15],'confidence_logit':[15,16]}},
        'width':model.encoder.out_channels,'parameters':sum(x.numel() for x in model.parameters()),'model_bytes_fp16':sum(x.numel()*2 for x in model.parameters()),
        'decoder':'compose() in model.py; tanh affine scale 0.125, detail gain tanh x2 times source minus 3x3 box, sigmoid confidence; bilinear align_corners=False',
        'temporal_state':'none; runtime must associate fields with source sequence/epoch; no stale-field reuse',
        'deployment_ready':False,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None}
    package.user_defined_metadata['neural_post_contract']=CONTRACT
    package.user_defined_metadata['training_status']=metadata['training_status']
    a.output.parent.mkdir(parents=True,exist_ok=True);package.save(str(a.output))
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    # A real-image prediction, compared to the exact same preprocessed bytes.
    from PIL import Image
    source=Path('build/gta5-neural-motion-180/pair-1826-input.png')
    if source.exists(): image=Image.open(source).convert('RGB').resize((WIDTH,HEIGHT),Image.Resampling.BOX)
    else: image=Image.fromarray(np.random.default_rng(73).integers(0,256,(HEIGHT,WIDTH,3),dtype=np.uint8))
    tensor=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()[None]/255
    with torch.no_grad(): expected=model(tensor).numpy()
    prediction=package.predict({'rgb':image})['appearance_fields']
    error=np.abs(prediction.astype(np.float32)-expected)
    parity={'input':'GTA V matched captured frame' if source.exists() else 'seeded diagnostic RGB',
            'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),'shape':list(prediction.shape),
            'finite':bool(np.isfinite(prediction).all()),'output_mean':float(prediction.mean()),
            'note':'Conversion parity only; training_status='+metadata['training_status']+'. Not appearance quality or ANE placement proof.'}
    a.output.with_suffix('.parity.json').write_text(json.dumps(parity,indent=2));print(json.dumps(parity))
    if not parity['finite'] or parity['max_abs_error']>0.005: raise RuntimeError('Export parity gate failed')
if __name__=='__main__':main()
