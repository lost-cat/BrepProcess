import json
import os

import h5py
import tqdm

from source.cadlib.fusion360_json_convert_util import from_fusion360_dict_CADSequence

json_path = 'data/fusion360/train_test.json'
step_dir = 'data/fusion360/reconstruction'
vec_dir = 'data/fusion360/vec16'
if not os.path.exists(vec_dir):
    os.makedirs(vec_dir)


def process_one(name):
    file_path = os.path.join(step_dir, name + '.json')
    vec_path = os.path.join(vec_dir, name + '.h5')

    if not os.path.exists(file_path):
        print(f'{name} not found')

    with open(file_path, 'r') as f:
        all_stat = json.load(f)
        cad_seq_json = from_fusion360_dict_CADSequence(all_stat)
        cad_seq_json.normalize()
        cad_seq_json.numericalize()
        vec_json = cad_seq_json.to_vector(max_len_loop=25)
        if vec_json is not None:
            with h5py.File(vec_path, 'w') as f_vec:
                f_vec.create_dataset('vec', data=vec_json, dtype=int)


if __name__ == '__main__':
    with open(json_path, 'r') as f:
        train_test_split = json.load(f)

    pbar_train = tqdm.tqdm(train_test_split['train'])
    pbar_test = tqdm.tqdm(train_test_split['test'])

    for name in pbar_train:
        process_one(name)

    for name in pbar_test:
        process_one(name)
