import json

import numpy as np
from OCC.Display.SimpleGui import init_display
from occwl.shape import Shape

from source.cadlib.visualize import create_CAD, CADsolid2pc, write_ply, vec2CADsolid

if __name__ == '__main__':

    path = '/media/huangxinyu/disk/数据/data/cad_json/0000/00000061.json'
    with open(path, 'rb') as f:
        data = json.load(f)
        try:
            from source.cadlib.extrude import CADSequence
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            shape = create_CAD(cad_seq)
        except Exception:
            print('create cad failed')

        try:
            from source.cadlib.extrude import CADSequence

            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            cad_seq.numericalize()
            seq = cad_seq.to_vector()
        except Exception:
            print('create cad failed')

        try:
            vec_shape = vec2CADsolid(seq)

            occwl_vec_shape = Shape.occwl_shape(vec_shape)
            occwl_vec_shape = occwl_vec_shape.scale_to_unit_box()
            occwl_vec_shape.translate(np.array([-2, 0, 0]))
            vec_shape = occwl_vec_shape.topods_shape()
            # vec_shape_1 = occwl_vec_shape.topods_shape()
            # success = occwl.io.save_step([occwl_vec_shape], save_path)
            # pc = CADsolid2pc(occwl_vec_shape.topods_shape(), 8096)
            # write_ply(pc, pc_path)
            # if success:
            #     print(f'{name} saved')
        except Exception as e:
            print(e)

        # occwl_shape = occwl.io.load_single_compound_from_step(step_path)
        # occwl_shape = occwl_shape.scale_to_unit_box()
        # occwl_shape.translate(np.array([2, 0, 0]))
        # step_shape = occwl_shape.topods_shape()
        #
        display, start_display, add_menu, add_function_to_menu = init_display()

        # display.DisplayShape(json_shape_1,color='BLUE')
        display.DisplayShape(shape, color='BLUE')
        display.DisplayShape(vec_shape)
        start_display()

        # 按键后才继续运行
