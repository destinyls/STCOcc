import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/yanglei/STCOcc')

from mmdet3d.datasets.nuscenes_ego_pose_loader import nuScenesDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train, val, test

print("Testing the fix for IndexError...")

# Load the NuScenes dataset
nusc = NuScenes('v1.0-trainval', '/home/yanglei/STCOcc/data/nuscenes/')

# Debug: Print some information about the scenes
print(f"Total scenes in dataset: {len(nusc.scene)}")
print(f"First few scene names: {[scene['name'] for scene in nusc.scene[:3]]}")
print(f"Number of scenes in val split: {len(val)}")
print(f"First few val scene names: {val[:3]}")

# Check if any scene is in the val split
scenes_in_val = [scene for scene in nusc.scene if scene['name'] in val]
print(f"Number of scenes from dataset in val split: {len(scenes_in_val)}")

# Load the nuScenesDataset with debug info
print("\nCreating nuScenesDataset instance...")
nusdata = nuScenesDataset(nusc, 'val')

print(f"Number of sample tokens: {len(nusdata.sample_tokens)}")
print(f"Number of valid indices: {len(nusdata.valid_index)}")

# Try with train split
try:
    print("\nTrying with 'train' split...")
    nusdata_train = nuScenesDataset(nusc, 'train')
    print(f"Train split - Number of sample tokens: {len(nusdata_train.sample_tokens)}")
    print(f"Train split - Number of valid indices: {len(nusdata_train.valid_index)}")
    
    if nusdata_train.sample_tokens:
        sample_token = nusdata_train.sample_tokens[0]
        print(f"Testing with train sample token: {sample_token}")
        ref_sample_token, output_origin_tensor = nusdata_train.get_data_by_sample_token(sample_token)
        print(f"✓ Success! Got data for sample token: {ref_sample_token}")
        print(f"Output origin tensor shape: {output_origin_tensor.shape}")
    else:
        print("No sample tokens in train split either.")
except Exception as e:
    print(f"Error with train split: {e}")

print("\nTest completed.")
