import torch

from source.dataProcess.process import process_one

test_data_ids = ['0066/00665514', '0033/00331864', '0046/00469549', '0023/00237704']

if __name__ == '__main__':
    for data_id in test_data_ids:
        process_one(data_id,'none')
