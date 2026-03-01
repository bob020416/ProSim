"""
Diagnose checkpoint vs model key mismatches.
Shows: which keys are missing from the model (skipped) and which model keys
are missing from the checkpoint (randomly initialized).
"""
import sys, os
sys.path.insert(0, os.getcwd())

import torch
from prosim.config.default import get_config
from prosim.core.registry import registry

config = get_config(
    "prosim_demo/cfg/no_text.yaml,prosim_demo/cfg/uncond_random10.yaml",
    cluster="local"
)

ckpt_path = "checkpoints/prosim_demo_model.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt_keys = set(ckpt["state_dict"].keys())

model_cls = registry.get_model(config.MODEL.TYPE)
model = model_cls(config)
model_keys = set(model.state_dict().keys())

in_ckpt_not_model = sorted(ckpt_keys - model_keys)
in_model_not_ckpt = sorted(model_keys - ckpt_keys)

print(f"\n=== Keys in CHECKPOINT but NOT in MODEL (skipped, {len(in_ckpt_not_model)} total) ===")
# Group by top-level module
from collections import Counter
skipped_modules = Counter(k.split('.')[0] + '.' + k.split('.')[1] if k.count('.') >= 1 else k
                          for k in in_ckpt_not_model)
for mod, cnt in skipped_modules.most_common():
    print(f"  {mod}: {cnt} keys")

print(f"\n=== Keys in MODEL but NOT in CHECKPOINT (not loaded!, {len(in_model_not_ckpt)} total) ===")
if in_model_not_ckpt:
    missing_modules = Counter(k.split('.')[0] + '.' + k.split('.')[1] if k.count('.') >= 1 else k
                              for k in in_model_not_ckpt)
    for mod, cnt in missing_modules.most_common():
        print(f"  {mod}: {cnt} keys")
else:
    print("  None — all model weights are present in checkpoint ✓")

print(f"\n=== Summary ===")
print(f"  Checkpoint keys:  {len(ckpt_keys)}")
print(f"  Model keys:       {len(model_keys)}")
print(f"  Successfully loaded: {len(ckpt_keys & model_keys)}")
print(f"  Skipped (ckpt only): {len(in_ckpt_not_model)}")
print(f"  Missing (model only): {len(in_model_not_ckpt)}")
