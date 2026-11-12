"""Fetch pinned source dependencies and apply the saved project patches.

Existing checkouts are verified; this never resets or overwrites local work.
"""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent.parent
for name,url,revision,patch in [
 ('MLX-DLSS','https://github.com/iamwavecut/MLX-DLSS.git','MLX-DLSS-REVISION','mlx-dlss.patch'),
 ('rife-metal','https://github.com/cinemore/rife-metal.git','RIFE-METAL-REVISION','rife-metal.patch')]:
    target=root/name;wanted=(root/'dependencies'/revision).read_text().strip()
    if not target.exists():
        subprocess.run(['git','clone',url,str(target)],check=True)
        subprocess.run(['git','checkout','--detach',wanted],cwd=target,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
    if actual!=wanted:raise SystemExit(f'{name}: expected {wanted}, found {actual}; existing checkout left untouched')
    patch_path=str(root/'dependencies'/patch)
    applied=subprocess.run(['git','apply','--reverse','--check',patch_path],cwd=target,capture_output=True)
    if applied.returncode:
        subprocess.run(['git','apply','--check',patch_path],cwd=target,check=True)
        subprocess.run(['git','apply',patch_path],cwd=target,check=True)
    print(name,actual,'project patch verified')
