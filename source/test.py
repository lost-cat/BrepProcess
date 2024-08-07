from OCC.Core.STEPControl import STEPControl_Reader
# from OCC.Display.SimpleGui import init_display

from source.util import normalize, get_face_edge_info, check_data, convert_to_dgl_graph, read_step

step_path = '../data/fusion360/reconstruction/20203_7e31e92a_0000_0001.step'
# step_reader = STEPControl_Reader()
# step_reader.ReadFile(step_path)
# step_reader.TransferRoot()
# shape = step_reader.OneShape()
# shape_normalized = normalize(shape)

compound_scaled = read_step(step_path, normalized=True)
face_infos, edge_infos = get_face_edge_info(compound_scaled)

face_list = list(face_infos.values())
edge_list = list(edge_infos.values())
is_valid = check_data(face_list, edge_list)
if not is_valid:
    print('invalid data')

dgl_graph = convert_to_dgl_graph(face_list, edge_list)
# dgl.data.save_graphs(save_path, [dgl_graph])

# display, start_display, add_menu, add_function_to_menu = init_display()
#
# display.DisplayShape(shape_normalized)
# start_display()
