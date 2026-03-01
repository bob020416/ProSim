"""
Rebuild trajdata's scenes_list.dill from individual cached scene_metadata files,
using the correct WaymoSceneRecord namedtuple format that trajdata expects.
This lets trajdata skip tfrecord loading entirely and go straight to cache.
"""
import os
import dill
from trajdata.dataset_specific.scene_records import WaymoSceneRecord

cache_root = "/home/msc-auto/wjchang/ProSim/trajdata_cache/waymo_val"
scenes_list_path = os.path.join(cache_root, "scenes_list.dill")

scene_dirs = sorted([
    d for d in os.listdir(cache_root)
    if os.path.isdir(os.path.join(cache_root, d)) and d.startswith("scene_")
])

print(f"Found {len(scene_dirs)} cached scene directories")

all_scenes_list = []
for scene_dir in scene_dirs:
    meta_path = os.path.join(cache_root, scene_dir, "scene_metadata_dt0.10.dill")
    if not os.path.exists(meta_path):
        print(f"  WARNING: no metadata for {scene_dir}, skipping")
        continue
    with open(meta_path, "rb") as f:
        meta = dill.load(f)

    record = WaymoSceneRecord(
        name=meta.name,
        length=str(meta.length_timesteps),
        data_idx=meta.raw_data_idx,
    )
    all_scenes_list.append(record)
    print(f"  {record}")

print(f"\nWriting {len(all_scenes_list)} WaymoSceneRecords to scenes_list.dill ...")
with open(scenes_list_path, "wb") as f:
    dill.dump(all_scenes_list, f)

# Verify
with open(scenes_list_path, "rb") as f:
    verify = dill.load(f)
print(f"Done — verified {len(verify)} records in {scenes_list_path}")
print(f"First: {verify[0]}")
