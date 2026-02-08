# Add this debug code to see what paths are being searched
import os
import glob

data_root = 'data/tartanair'  # Your path here
camera = 'lcam_front'

# What the loader searches for
voxel_pattern = os.path.join(data_root, '*', '*', 'P*', f'voxel_label_{camera}', '*_voxel_label.npy')
print(f"Looking for: {voxel_pattern}")
print(f"Found: {glob.glob(voxel_pattern)[:5]}")

# Alternative pattern (flat structure)
voxel_pattern2 = os.path.join(data_root, 'P*', f'voxel_label_{camera}', '*_voxel_label.npy')
print(f"Alt pattern: {voxel_pattern2}")
print(f"Found: {glob.glob(voxel_pattern2)[:5]}")
