import json
from pathlib import Path

root = Path("/home/ec2-user/research/scripts/png_out")
manifest = {}

for sample in sorted(p for p in root.iterdir() if p.is_dir()):
    for sub in ("dim", "normal"):
        d = sample / sub
        if not d.is_dir():
            continue
        imgs = sorted(
            str(p) for p in d.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        )
        if imgs:
            manifest[f"{sample.name}/{sub}"] = imgs

with open("png_out_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

counts = {len(v) for v in manifest.values()}
print(f"{len(manifest)} entries, images per folder: {counts}")