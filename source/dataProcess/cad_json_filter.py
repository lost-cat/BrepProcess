import json
import hashlib
import os


def get_file_hash(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


DATA_DIR = '../../data'
RECORD_FILE = os.path.join(DATA_DIR, 'train_val_test_split.json')

new_train_data_ids = []
new_validation_data_ids = []
new_test_data_ids = []

with open(RECORD_FILE, 'r') as f:
    all_data = json.load(f)
seen_hashes = set()
for x in all_data['train']:
    json_path = os.path.join(DATA_DIR, 'cad_json', x + '.json')
    hash_ = get_file_hash(json_path)
    if hash_ in seen_hashes:
        print('duplicate file:', x)
    else:
        seen_hashes.add(hash_)
        new_train_data_ids.append(x)

for x in all_data['validation']:
    json_path = os.path.join(DATA_DIR, 'cad_json', x + '.json')
    hash_ = get_file_hash(json_path)
    if hash_ in seen_hashes:
        print('duplicate file:', x)
    else:
        seen_hashes.add(hash_)
        new_validation_data_ids.append(x)

for x in all_data['test']:
    json_path = os.path.join(DATA_DIR, 'cad_json', x + '.json')
    hash_ = get_file_hash(json_path)
    if hash_ in seen_hashes:
        print('duplicate file:', x)
    else:
        seen_hashes.add(hash_)
        new_test_data_ids.append(x)
filtered_train_val_test_split_file = os.path.join(DATA_DIR, 'filtered_train_val_test_split.json')
with open(filtered_train_val_test_split_file, 'w') as f:
    json.dump({'train': new_train_data_ids, 'validation': new_validation_data_ids, 'test': new_test_data_ids}, f,
              indent=2)
