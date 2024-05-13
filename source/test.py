import json

import occwl.io

with open('../data/balanced_train_val_test_split.json', 'r') as f:
    data = json.load(f)

with open('../data/invalid_ids.json', 'r') as f:
    invalid_ids = json.load(f)

new_train = []
new_val = []
new_test = []
for x in data['train']:
    if x in invalid_ids:
        continue
    new_train.append(x)

for x in data['validation']:
    if x in invalid_ids:
        continue
    new_val.append(x)

for x in data['test']:
    if x in invalid_ids:
        continue
    new_test.append(x)

with open('../data/new_split.json', 'w') as f:
    json.dump({'train': new_train, 'validation': new_val, 'test': new_test}, f, indent=2)

