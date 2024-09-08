import glob
import os.path

import h5py
import numpy as np
import occwl.io
import tqdm
from OCC.Display.SimpleGui import init_display
from occwl.shape import Shape

from source.cadlib.extrude import CADSequence
# from source.cadlib.fusion360_json_convert_util import from_fusion360_dict_CADSequence
#
# fusion360_json_dir = 'data/fusion360/reconstruction'
# fusion360_json_paths = glob.glob(os.path.join(fusion360_json_dir, '*.json'))
# path = fusion360_json_paths[1]
# with open(path, 'r') as f:
#     data = json.load(f)
#     cad_seq = f                             rom_fusion360_dict_CADSequence(data)
#     cad_seq.normalize()
#     cad_seq.numericalize()
#     vec = cad_seq.to_vector(max_n_ext=20)
#
# print(path)
#
#
# vec_path ="data/fusion360_vec/100155_57ec5fc6_0000.h5"
# with open(vec_path, 'rb') as f:
#     file = h5py.File(f)
#     cad_vec = file['vec']
#     cad_vec = np.array(cad_vec)
#
# print(path)
from source.cadlib.visualize import create_CAD

if __name__ == '__main__':
    step_dir = 'data/fusion360/reconstruction'
    vec16_dir = 'data/fusion360/vec16'
    save_dir = 'data/fusion360/step_generated'

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    paths = glob.glob(os.path.join(vec16_dir, '*.h5'))
    pbar = tqdm.tqdm(paths)
    for path in pbar:
        name = os.path.basename(path).split('.')[0]
        print(name)
        step_path = os.path.join(step_dir, name + '.step')
        json_path = os.path.join(step_dir, name + '.json')
        save_path = os.path.join(save_dir, name + '.step')
        # with open(json_path, 'r') as f:
        #     all_stat = json.load(f)
        #     cad_seq_json = from_fusion360_dict_CADSequence(all_stat)
        #     # cad_seq_json.seq[-1].operation = 0
        #     cad_seq_json.normalize()
        #     cad_seq_json.numericalize()
        #     # json_shape = create_CAD(cad_seq_json)
        #     vec_json = cad_seq_json.to_vector(max_len_loop=25)
        #     json_shape = vec2CADsolid(vec_json)
        #     #
        #     occwl_json_shape = Shape.occwl_shape(json_shape)
        #     occwl_json_shape = occwl_json_shape.scale_to_unit_box()
        #     json_shape_1 = occwl_json_shape.topods_shape()

        with open(path, 'rb') as f:
            file = h5py.File(f)
            cad_vec = np.array(file['vec'])
            cad_seq = CADSequence.from_vector(cad_vec, True)
            try:
                vec_shape = create_CAD(cad_seq)
                occwl_vec_shape = Shape.occwl_shape(vec_shape)
                occwl_vec_shape = occwl_vec_shape.scale_to_unit_box()
                # occwl_vec_shape.translate(np.array([-2, 0, 0]))
                # vec_shape_1 = occwl_vec_shape.topods_shape()
                success = occwl.io.save_step([occwl_vec_shape], save_path)
                # if success:
                #     print(f'{name} saved')
            except Exception as e:
                print(e)
                continue

        # occwl_shape = occwl.io.load_single_compound_from_step(step_path)
        # occwl_shape = occwl_shape.scale_to_unit_box()
        # occwl_shape.translate(np.array([2, 0, 0]))
        # step_shape = occwl_shape.topods_shape()
        # #
        # display, start_display, add_menu, add_function_to_menu = init_display()
        #
        # # display.DisplayShape(json_shape_1,color='BLUE')
        # display.DisplayShape(vec_shape_1, )
        # display.DisplayShape(step_shape)
        # start_display()
        #
        # # 按键后才继续运行
