import json
import os.path

import tqdm
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.STEPControl import STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs
from joblib import Parallel, delayed

from source.dataProcess.cadlib.visualize import create_CAD

DATA_DIR = '../../data'
SAVE_DIR = os.path.join(DATA_DIR, 'step')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

RECORD_FILE = os.path.join(DATA_DIR, 'train_val_test_split.json')

INVALID_IDS = []


def write_step_file(shape, save_path):
    step_writer = STEPControl_Writer()
    # dd = step_writer.WS().TransferWriter().FinderProcess()
    # print(dd)
    Interface_Static_SetCVal("write.step.schema", "AP203")
    step_writer.Transfer(shape, STEPControl_AsIs)
    status = step_writer.Write(save_path)
    if status != IFSelect_RetDone:
        raise ValueError('write step failed')


def process_one(data_id):
    if data_id in INVALID_IDS:
        print('skip {} in invalid ids'.format(data_id))
        return
    print('processing', data_id)

    # processing data
    save_path = os.path.join(SAVE_DIR, data_id + '.step')
    json_path = os.path.join(RAW_DIR, data_id + '.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    try:
        from source.dataProcess.cadlib.extrude import CADSequence
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception:
        print('create cad failed', data_id)
        return None

    try:
        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox)
        if bbox.IsVoid():
            raise ValueError('box check failed')

        truck_dir = os.path.dirname(save_path)
        if not os.path.exists(truck_dir):
            os.makedirs(truck_dir)

        write_step_file(shape, save_path)
    except Exception as e:
        print('create step failed', data_id)
        return None


if __name__ == '__main__':
    with open(RECORD_FILE, 'r') as f:
        all_data = json.load(f)
    for x in tqdm.tqdm(all_data['train'], postfix='train'):
        process_one(x)
    for x in tqdm.tqdm(all_data['validation']):
        process_one(x)

    Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in tqdm.tqdm(all_data['test']))
