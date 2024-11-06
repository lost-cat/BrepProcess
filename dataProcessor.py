import json
import os

import dgl
import h5py
import occwl.io
from occwl.shape import Shape
import tqdm
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

from source.cadlib.visualize import create_CAD
from source.util import extract_dgl_graph_from_step


def ensure_dir_exists(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def write_step_file(shape, save_path):
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step_path.schema", "AP203")
    step_writer.Transfer(shape, STEPControl_AsIs)
    status = step_writer.Write(save_path)
    if status != IFSelect_RetDone:
        raise ValueError('write step failed')


class DataProcessor:

    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir

    def process(self):
        raise NotImplementedError()


class DeepCadDataProcessor(DataProcessor):

    def __init__(self, data_dir, save_dir):
        super().__init__(data_dir, save_dir=save_dir)
        self.save_step = True
        self.save_vec = True
        self.split_path = os.path.join(data_dir, 'train_val_test_split.json')
        self.raw_path = os.path.join(data_dir, 'cad_json')

        self.save_graph_dir = os.path.join(save_dir, 'dgl')
        self.save_seq_dir = os.path.join(save_dir, 'seq')
        self.save_step_dir = os.path.join(save_dir, 'brep')
        self.invalid_ids = ['0035/00350811', '0042/00420819', '0017/00178001', '0016/00163962',
                            '0083/00833781', '0056/00560203', '0076/00765224']
        self.new_split = {}

    def process(self, save_step=True, save_vec=True):
        self.save_step = save_step
        self.save_vec = save_vec
        for id in self.invalid_ids:

            self.process_one(id)
        with open(self.split_path, 'r') as f:
            split = json.load(f)
            self.process_split(split)

        new_split_path = os.path.join(self.save_dir, 'new_split.json')
        with open(new_split_path, 'w') as f:
            json.dump(self.new_split, f, indent=2)
        # print invalid ids
        print('Invalid IDs:', self.invalid_ids)
        pass

    def process_split(self, split):
        for phase in split:
            if phase == 'train':

                pbar = tqdm.tqdm(split[phase])
            else:
                pbar = tqdm.tqdm(split[phase])
            valid_ids = []
            for data_id in pbar:
                pbar.set_postfix({'invalid': len(self.invalid_ids), 'data_id': data_id})
                if data_id in self.invalid_ids:
                    continue
                # status = self.process_one(data_id)
                # if not status:
                #     self.invalid_ids.append(data_id)
                if self.is_data_id_available(data_id):
                    valid_ids.append(data_id)

            self.new_split[phase] = valid_ids

    def process_one(self, data_id):
        json_path = os.path.join(self.raw_path, data_id + '.json')

        save_graph_path = os.path.join(self.save_graph_dir, data_id + '.bin')
        save_step_path = os.path.join(self.save_step_dir, data_id + '.step')
        save_seq_path = os.path.join(self.save_seq_dir, data_id + '.h5')

        truck_dir = os.path.dirname(save_graph_path)
        ensure_dir_exists(truck_dir)
        truck_dir = os.path.dirname(save_seq_path)
        ensure_dir_exists(truck_dir)
        truck_dir = os.path.dirname(save_step_path)
        if not os.path.exists(truck_dir):
            os.makedirs(truck_dir)
        with open(json_path, 'r') as f:
            data = json.load(f)
        try:
            from source.cadlib.extrude import CADSequence
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            shape = create_CAD(cad_seq)
        except Exception:
            print('create cad failed', data_id)
            return False

        if self.save_step:
            try:
                occwl_shape = Shape.occwl_shape(shape)
                flag = self.write_step_file(occwl_shape, data_id)
                if not flag:
                    return False
            except Exception:
                print("save_shape_failed", shape)
        graph, success = extract_dgl_graph_from_step(save_step_path)
        if not success:
            return False

        dgl.data.save_graphs(save_graph_path, [graph])
        try:
            from source.cadlib.extrude import CADSequence
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            cad_seq.numericalize()
            seq = cad_seq.to_vector()
        except Exception:
            print('create cad failed', data_id)
            return False

        with h5py.File(save_seq_path, 'w') as f_vec:
            f_vec.create_dataset('vec', data=seq, dtype=int)

        return True

    def write_step_file(self, shape, data_id):
        save_step_path = os.path.join(self.save_step_dir, data_id + '.step')

        try:
            return occwl.io.save_step([shape], save_step_path)
        except Exception as e:
            print('create step_file failed', data_id, e)
            return False

    def is_data_id_available(self, data_id) -> bool:
        save_graph_path = os.path.join(self.save_graph_dir, data_id + '.bin')
        save_step_path = os.path.join(self.save_step_dir, data_id + '.step')
        save_seq_path = os.path.join(self.save_seq_dir, data_id + '.h5')

        if (os.path.exists(save_seq_path) and os.path.exists(save_graph_path)
                and os.path.exists(save_step_path)):
            return True
        else:
            return False
