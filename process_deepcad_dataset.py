import json
import os.path

import dgl.data
import tqdm

from source.util import extract_dgl_graph_from_step

INVALID_IDS = []


def process_one(data_id, save_dir, step_dir) -> bool:
    """
     Processes a single data item by extracting a DGL graph from a STEP file and saving it.

     Args:
         data_id (str): The identifier of the data item to process.
         save_dir (str): The directory where the processed graph will be saved. Defaults to None.
         step_dir (str): The directory where the STEP file is located. Defaults to None.

     Returns:
         bool: True if the processing was successful, False otherwise.
     """

    save_path = os.path.join(save_dir, data_id + '.bin')
    step_path = os.path.join(step_dir, data_id + '.step')

    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    graph, success = extract_dgl_graph_from_step(step_path)
    if not success:
        return False
    dgl.data.save_graphs(save_path, [graph])
    return True


DATA_DIR = '../../data'
SAVE_DIR = os.path.join(DATA_DIR, 'dgl')
STEP_DIR = os.path.join(DATA_DIR, 'step')
RECORD_FILE = os.path.join(DATA_DIR, 'balanced_train_val_test_split.json')

new_train_data_ids = []
new_validation_data_ids = []
new_test_data_ids = []


def process_all():
    global new_train_data_ids, new_validation_data_ids, new_test_data_ids

    with open(RECORD_FILE, 'r') as f:
        all_data = json.load(f)
    pbar = tqdm.tqdm(all_data['train'])
    for x in pbar:
        pbar.set_description('processing train' + x)
        process_one(x, SAVE_DIR, STEP_DIR)
        pbar.set_postfix({'invalid': len(INVALID_IDS)})

    pbar = tqdm.tqdm(all_data['validation'])
    for x in pbar:
        pbar.set_description('processing validation' + x)
        process_one(x, SAVE_DIR, STEP_DIR)
        pbar.set_postfix({'invalid': len(INVALID_IDS)})

    pbar = tqdm.tqdm(all_data['test'])
    for x in pbar:
        pbar.set_description('processing test' + x)
        process_one(x, SAVE_DIR, STEP_DIR)
        pbar.set_postfix({'invalid': len(INVALID_IDS)})

    invalid_id_file = os.path.join(DATA_DIR, 'invalid_ids.json')
    new_train_val_test_split_file = os.path.join(DATA_DIR, 'new_train_val_test_split.json')
    # write INVALID_IDS to invalid_id_file
    with open(invalid_id_file, 'w') as f:
        json.dump(INVALID_IDS, f, indent=2)
    with open(new_train_val_test_split_file, 'w') as f:
        json.dump({'train': new_train_data_ids, 'validation': new_validation_data_ids, 'test': new_test_data_ids}, f,
                  indent=2)



if __name__ == '__main__':
    # test_data_ids = ['0000/00000007']
    #
    # for data_id in test_data_ids:
    #     process_one(data_id, 'none')
    #
    process_all()
