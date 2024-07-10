import argparse
import json
import os

import tqdm

from source.process import process_one

data_dir = ''

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Process Fusion360 data')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='data directory')
    args = arg_parser.parse_args()
    data_dir = args.data_dir
    json_file = os.path.join(data_dir, 'train_test.json')
    save_dir = os.path.join(data_dir, 'dgl')
    step_dir = os.path.join(data_dir, 'reconstruction')

    # process_one('27839_4a077326_0010', save_dir, step_dir)
    with open(json_file, 'r') as f:
        data_ids = json.load(f)
        test_data_ids = data_ids['test']
        train_data_ids = data_ids['train']

    pbar_test = tqdm.tqdm(test_data_ids)
    pbar_train = tqdm.tqdm(train_data_ids)
    for data_id in pbar_test:
        pbar_test.set_description('processing test' + data_id)
        process_one(data_id, save_dir, step_dir)
    for data_id in pbar_train:
        pbar_train.set_description('processing train' + data_id)
        process_one(data_id, save_dir, step_dir)

    pass
