"""Build the isolated ANEForge/Metal interoperability extension."""
from pathlib import Path
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--aneforge', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mlx-project', type=Path, help='Also build the native MLX layout probe using existing runtime libraries')
args = parser.parse_args()
source = args.aneforge.resolve() / 'aneforge/_lib/ane_e5rt_dispatch.mm'
if not source.is_file():
    parser.error('ANEForge dispatch source not found')
args.output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    'xcrun', 'clang++', '-target', 'arm64-apple-macos15.0', '-std=c++17', '-O2', '-Wall', '-Wextra',
    '-dynamiclib', '-Wl,-install_name,@rpath/' + args.output.name, '-fPIC', '-fobjc-arc', '-framework', 'Foundation',
    '-framework', 'Metal', '-framework', 'IOSurface',
    f'-DANEFORGE_DISPATCH_SOURCE="{source}"', '-I', str(source.parent),
    str(Path(__file__).with_name('SurfaceBridge.mm')),
    str(Path(__file__).with_name('HybridFFN.mm')), '-o', str(args.output),
], check=True)
print(args.output)

if args.mlx_project:
    root = args.mlx_project.resolve()
    products = root / '.build/release'
    runtime = Path(__file__).resolve().parents[2] / 'build/nr'
    if not (runtime/'mlx.metallib').is_file():
        parser.error('The existing runtime directory needs its matching mlx.metallib')
    subprocess.run([
        'xcrun', 'swiftc', '-O', '-target', 'arm64-apple-macos15.0',
        '-I', str(products),
        '-I', str(root/'.build/checkouts/mlx-swift/Source/Cmlx/include'),
        '-I', str(root/'.build/checkouts/swift-numerics/Sources/_NumericsShims/include'),
        '-import-objc-header', str(Path(__file__).with_name('ANEForgeBridge.h')),
        str(Path(__file__).with_name('HybridLayoutProbe.swift')),
        str(args.output.resolve()), '-L', str(runtime), '-lnvgnx_dlss_mlx',
        '-Xlinker', '-rpath', '-Xlinker', str(args.output.resolve().parent),
        '-Xlinker', '-rpath', '-Xlinker', str(runtime),
        '-framework', 'Metal', '-framework', 'IOSurface', '-framework', 'Foundation',
        '-o', str(args.output.resolve().parent/'hybrid-layout-probe'),
    ], check=True)
