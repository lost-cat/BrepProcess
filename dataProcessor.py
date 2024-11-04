import json
import os

import dgl
import h5py
import tqdm
from sympy import false
from sympy.conftest import process_split

from source.cadlib.visualize import create_CAD
from source.util import extract_dgl_graph_from_step


def ensure_dir_exists(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

class DataProcessor:

    def __init__(self, data_dir,save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir

    def process(self):
        raise NotImplementedError()


class DeepCadDataProcessor(DataProcessor):

    def __init__(self, data_dir,save_dir):
        super().__init__(data_dir,save_dir=save_dir)
        self.split_path = os.path.join(data_dir,'train_val_test_split.json')
        self.raw_path = os.path.join(data_dir, 'cad_json')

        self.save_graph_dir = os.path.join(save_dir, 'dgl')
        self.save_seq_dir = os.path.join(save_dir, 'seq')
        self.invalid_ids = []

    def process(self):
        with open(self.split_path, 'r') as f:
            split = json.load(f)
            self.process_split(split)

        #print invalid ids
        print('Invalid IDs:', self.invalid_ids)
        pass

    def process_split(self,split):
        for phase in split:
            pbar = tqdm.tqdm(split[phase])
            for data_id in pbar:
                status = self.process_one(data_id)
                if not status:
                    self.invalid_ids.append(data_id)
                pbar.set_postfix({'invalid': len(self.invalid_ids)})


    def process_one(self,data_id):
        json_path = os.path.join(self.raw_path, data_id + '.json')

        save_graph_path = os.path.join(self.save_graph_dir, data_id + '.bin')
        save_seq_path = os.path.join(self.save_seq_dir, data_id + '.h5')

        truck_dir = os.path.dirname(save_graph_path)
        ensure_dir_exists(truck_dir)
        truck_dir = os.path.dirname(save_seq_path)
        ensure_dir_exists(truck_dir)


        with open(json_path, 'r') as f:
            data = json.load(f)
            try:
                from source.cadlib.extrude import CADSequence
                cad_seq = CADSequence.from_dict(data)
                cad_seq.normalize()
                shape = create_CAD(cad_seq)
            except Exception:
                print('create cad failed', data_id)
                return false
            try:
                from source.cadlib.extrude import CADSequence
                cad_seq = CADSequence.from_dict(data)
                cad_seq.normalize()
                cad_seq.numericalize()
                seq = cad_seq.to_vector()
            except Exception:
                print('create cad failed', data_id)
                return false

        with h5py.File(save_seq_path, 'w') as f_vec:
            f_vec.create_dataset('vec', data=seq, dtype=int)

        graph, success = extract_dgl_graph_from_step(None,shape)
        if not success:
            return False

        dgl.data.save_graphs(save_graph_path, [graph])

        return True

