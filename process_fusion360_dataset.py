import argparse
import json
import os

import dgl
import tqdm

from source.util import extract_dgl_graph_from_step


def process_one(step_path, save_path):
    graph, success = extract_dgl_graph_from_step(step_path)
    if not success:
        return False
    dgl.data.save_graphs(save_path, [graph])
    return True
    pass


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Process Fusion360 data')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='data directory')
    args = arg_parser.parse_args()
    data_dir = args.data_dir
    json_file = os.path.join(data_dir, 'train_test_new.json')
    save_dir = os.path.join(data_dir, 'dgl')
    step_dir = os.path.join(data_dir, 'step_generated')
    invalid_ids = []
    # process_one('27839_4a077326_0010', save_dir, step_dir)
    with open(json_file, 'r') as f:
        data_ids = json.load(f)
        test_data_ids = data_ids['test']
        train_data_ids = data_ids['train']

    pbar_test = tqdm.tqdm(test_data_ids)
    pbar_train = tqdm.tqdm(train_data_ids)
    for data_id in pbar_test:
        pbar_test.set_description('processing test' + data_id)
        step_path = os.path.join(step_dir, data_id + '.step')
        save_path = os.path.join(save_dir, data_id + '.bin')
        success = process_one(step_path, save_path)
        if not success:
            invalid_ids.append(data_id)
    for data_id in pbar_train:
        pbar_train.set_description('processing train' + data_id)
        step_path = os.path.join(step_dir, data_id + '.step')
        save_path = os.path.join(save_dir, data_id + '.bin')
        success = process_one(step_path, save_path)
        if not success:
            invalid_ids.append(data_id)

    for invalid_id in invalid_ids:
        print(invalid_id)
