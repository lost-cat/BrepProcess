import json
import os.path

import tqdm

from source.dataProcess.util import get_path_by_data_id, read_step, write_h5file, check_data

INVALID_IDS = []


def process_one(data_id, phase):
    """
    This function processes a single data_id. It first checks if the data_id is in the list of INVALID_IDS.
    If it is, the function prints a message and returns. If not, it proceeds to process the data_id.

    The function constructs the save_path and step_path using the data_id. It then reads the step file
    and retrieves face and edge information.

    If the directory for the save_path does not exist, it creates it. Finally, it writes the face and edge
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
    save_path = get_path_by_data_id(data_id, SAVE_DIR, '.h5')
    step_path = get_path_by_data_id(data_id, STEP_DIR, '.step')

    # Read the step file and retrieve face and edge information\
    try:
        face_infos, edge_infos = read_step(step_path)
    except Exception:
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
    # Write the face and edge information to a h5 file at the save_path
    write_h5file(save_path, face_list, edge_list)
    if phase == 'train':
        new_train_data_ids.append(data_id)
    elif phase == 'validation':
        new_validation_data_ids.append(data_id)
    elif phase == 'test':
        new_test_data_ids.append(data_id)


DATA_DIR = '../../data'
SAVE_DIR = os.path.join(DATA_DIR, 'h5file')
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
        process_one(x, 'train')
        pbar.set_postfix({'invalid': len(INVALID_IDS)})
    pbar = tqdm.tqdm(all_data['validation'])
    for x in pbar:
        pbar.set_description('processing validation' + x)
        process_one(x, 'validation')
        pbar.set_postfix({'invalid': len(INVALID_IDS)})
    pbar = tqdm.tqdm(all_data['test'])
    for x in pbar:
        pbar.set_description('processing test' + x)
        process_one(x, 'test')
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
    # test_data_ids = ['0000/00000001']
    #
    # for data_id in test_data_ids:
    #     process_one(data_id, 'none')

    process_all()
