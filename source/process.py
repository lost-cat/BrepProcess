import json
import os.path

import dgl.data
import occwl.io
import tqdm

from source.util import get_path_by_data_id, read_step, check_data, convert_to_dgl_graph, get_face_edge_info

INVALID_IDS = []


def process_one(data_id, save_dir=None, step_dir=None):
    """
    This function processes a single data_id. It first checks if the data_id is in the list of INVALID_IDS.
    If it is, the function prints a message and returns. If not, it proceeds to process the data_id.

    The function constructs the save_path and step_path using the data_id. It then reads the step_path file
    and retrieves occ_face and edge information.

    If the directory for the save_path does not exist, it creates it. Finally, it writes the occ_face and edge
    information to a h5 file at the save_path.

    Args:
        data_id (str): The data_id to be processed.

    Returns:
        None
    """
    # Check if data_id is in the list of invalid ids
    # if data_id in INVALID_IDS:
    #     print('skip {} in invalid ids'.format(data_id))
    #     return

    # Construct the data_id, save_path and step_path
    save_path = os.path.join(save_dir, data_id + '.bin')
    step_path = os.path.join(step_dir, data_id + '.step')

    step_path_new = os.path.join(step_dir, data_id + '_0001.step')
    if os.path.exists(step_path_new):
        step_path = step_path_new

    # Read the step_path file and retrieve occ_face and edge information\
    try:
        shape = read_step(step_path, normalized=True)
        face_infos, edge_infos = get_face_edge_info(shape)
    except Exception as e:
        print('invalid id', data_id, e)
        INVALID_IDS.append(data_id)
        return

        # Create the directory for save_path if it does not exist
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    # Check if the data is valid
    face_list = list(face_infos.values())
    edge_list = list(edge_infos.values())
    is_valid = check_data(face_list, edge_list)
    if not is_valid:
        print('invalid data', data_id)
        INVALID_IDS.append(data_id)
        return

    dgl_graph = convert_to_dgl_graph(face_list, edge_list)
    dgl.data.save_graphs(save_path, [dgl_graph])


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


def process_brep2seq_data():
    filelist = '../../data/brep2seq/test.txt'
    save_dir = '../../data/brep2seq/dgl'

    step_dir = '../../data/brep2seq/step_path'
    with open(filelist, 'r') as f:
        data_ids = [x.strip() for x in f.readlines()]
        for data_id in data_ids:
            process_one(data_id, save_dir, step_dir)


if __name__ == '__main__':
    # test_data_ids = ['0000/00000007']
    #
    # for data_id in test_data_ids:
    #     process_one(data_id, 'none')
    #
    process_all()
