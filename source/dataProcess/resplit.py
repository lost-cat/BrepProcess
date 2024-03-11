import json

from sklearn.model_selection import train_test_split

# Load the data
with open('../../data/filtered_train_val_test_split.json', 'r') as f:
    data = json.load(f)

# Calculate the ratios
total = len(data['train']) + len(data['validation']) + len(data['test'])
train_ratio = 7 / (7 + 1.5 + 1.5)
temp_ratio = 1 - train_ratio
val_ratio = 1.5 / (1.5 + 1.5)

# Combine all data
all_data = data['train'] + data['validation'] + data['test']

# Split the data
train_data, temp_data = train_test_split(all_data, train_size=train_ratio)
val_data, test_data = train_test_split(temp_data, train_size=val_ratio)

with open('../../data/balanced_train_val_test_split.json', 'w') as f:
    json.dump({'train': train_data, 'validation': val_data, 'test': test_data}, f, indent=2)
print('Train set length:', len(train_data))
print('Validation set length:', len(val_data))
print('Test set length:', len(test_data))
