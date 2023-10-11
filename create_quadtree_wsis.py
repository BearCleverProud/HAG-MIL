

import h5py
import os
import numpy as np
from wsi_core.wsi_utils import save_hdf5

def generate_surroundings(coord, step):
    x, y = coord.tolist()
    return [coord.tolist(), [x, y+step], [x+step, y], [x+step, y+step]]

def generate_patches(level, patch_size):
    if level == 2:
        root = f'camelyon_patches_{patch_size}_level_{level}/patches/'
    else:
        root = f'camelyon_patches_{patch_size}_level_{level}_corresponding/patches/'
    new_root = f'camelyon_patches_{patch_size}_level_{level-1}_corresponding/patches/'
    print(root)
    filenames = os.listdir(root)
    for filename in filenames:
        with h5py.File(os.path.join(root, filename), "r") as f:
            attr = {}

            attrs = f['coords'].attrs
            attr['patch_level'] = level - 1
            attr['patch_size'] = patch_size
            attr['level_dim'] = (attrs['level_dim'][0] * 2, attrs['level_dim'][1] * 2)
            coordinates = []
            if level == 2:
                step = patch_size * 2
            elif level == 1:
                step = patch_size

            for coord in f['coords']:
                new_coord = generate_surroundings(coord, step=step)
                coordinates.extend(new_coord)
            
            coordinates = np.array(coordinates)
            print(f'{filename}: {coordinates.shape}, original: {f["coords"].shape}, times 4 {(f["coords"].shape[0]*4, 2)}')
            attr_dict = { 'coords' : attr}
            asset_dict = {'coords' : coordinates}
            save_hdf5(os.path.join(new_root, filename), asset_dict, attr_dict, mode='w')

generate_patches(2, 256)
generate_patches(1, 256)