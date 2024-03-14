# import json
#
# with open('../../data/balanced_train_val_test_split.json', 'r') as f:
#     balanced_data = json.load(f)
#
# print('balanced train length', len(balanced_data['train']))
# print('balanced val length', len(balanced_data['validation']))
# print('balanced test length', len(balanced_data['test']))
# print('total length', len(balanced_data['train']) + len(balanced_data['validation']) + len(balanced_data['test']))
#
#
# for x in balanced_data['validation']:
#     if x in balanced_data['train']:
#         print(x)
#
# with open('../../data/train_val_test_split.json', 'r') as f:
#     data = json.load(f)
# print('original train length', len(data['train']))
# print('original val length', len(data['validation']))
# print('original test length', len(data['test']))
#
# print('total length', len(data['train']) + len(data['validation']) + len(data['test']))
#
# # please split the data into train,val,test in 7:1.5:1.5
from occwl.graph import face_adjacency
