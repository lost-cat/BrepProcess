import argparse
import json
import os

from source.process import process_one

data_dir = ''

INVALID_IDS = []

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Process Fusion360 data')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='data directory')
    args = arg_parser.parse_args()
    data_dir = args.data_dir
    json_file = os.path.join(data_dir, 'train_test.json')
    save_dir = os.path.join(data_dir, 'dgl')
    step_dir = os.path.join(data_dir, 'reconstruction')

    with open(json_file, 'r') as f:
        data_ids = json.load(f)
        test_data_ids = data_ids['test']
        train_data_ids = data_ids['train']

    for data_id in test_data_ids:
        process_one(data_id, save_dir, step_dir)
        pass
    for data_id in train_data_ids:
        process_one(data_id, save_dir, step_dir)

    pass
