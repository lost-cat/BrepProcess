import json
import os.path

import h5py
import numpy as np
import torch
import tqdm
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer
from joblib import Parallel, delayed
from torch_geometric.utils import to_undirected, add_self_loops
import torch_geometric.data as data
from typing import List

from source.dataProcess.dataStructure import FaceInfo, EdgeInfo


def get_face_infos(faces):
    face_infos = {}
    for face in faces:
        face_info = FaceInfo(face, len(face_infos))
        face_infos[face_info.hash] = face_info

    return face_infos


def get_edge_infos(topo, face_infos, occ_faces):
    edge_infos = {}
    edges = topo.edges()
    face_adj = np.zeros((len(occ_faces), len(occ_faces)))
    for edge in edges:
        faces = list(topo.faces_from_edge(edge))
        if len(faces) != 2:
            continue

        edge_info = EdgeInfo(edge, len(edge_infos))
        edge_info.faces = faces
        for face in faces:
            edge_info.face_hashes.append(hash(face))
            edge_info.face_tags.append(occ_faces.index(face))
        face_adj[edge_info.face_tags[0], edge_info.face_tags[1]] = 1
        face_adj[edge_info.face_tags[1], edge_info.face_tags[0]] = 1
        edge_infos[edge_info.hash] = edge_info

    return edge_infos, face_adj
    pass


def write_h5file(h5_path, face_infos: List[FaceInfo], edge_infos: List[EdgeInfo]):
    file = h5py.File(h5_path, 'w')

    file.create_dataset("face_normals", data=np.array([face_info.face_normal for face_info in face_infos]))
    file.create_dataset('face_centroids', data=np.array([face_info.centroid for face_info in face_infos]))
    file.create_dataset('face_areas', data=np.array([face_info.face_area for face_info in face_infos]))
    file.create_dataset('face_types', data=np.array([face_info.face_type for face_info in face_infos]))

    file.create_dataset('edge_links', data=np.array([edge_info.face_tags for edge_info in edge_infos]))
    file.create_dataset('edge_types', data=np.array([edge_info.edge_type for edge_info in edge_infos]))
    # group.create_dataset('edge_convexity', data=np.array([edge_info.convexity for edge_info in edge_infos]))
    file.close()
    pass


def load_h5file(data_id):
    data_id = os.path.join(data_id.split('/')[0], data_id.split('/')[1])
    h5_path = os.path.join(SAVE_DIR, data_id + '.h5')
    file = h5py.File(h5_path, 'r')
    face_normals = np.array(file.get("face_normals"))
    face_centroids = np.array(file.get("face_centroids"))
    face_areas = np.array(file.get("face_areas")).reshape(-1, 1)
    face_types = np.array(file.get("face_types")).reshape(-1, 1)
    edge_links = np.array(file.get("edge_links"))

    attrs = np.concatenate((face_normals, face_centroids, face_areas, face_types), axis=1)
    return attrs, edge_links
    pass


def read_step(filepath):
    if not os.path.exists(filepath):
        raise Exception(filepath, "not exists")
    reader = STEPControl_Reader()
    reader.ReadFile(filepath)
    reader.TransferRoot()
    shape = reader.OneShape()

    # transfer_reader = reader.WS().TransferReader()
    topo_explorer = TopologyExplorer(shape)
    faces = list(topo_explorer.faces())
    face_infos = get_face_infos(faces)
    edge_infos, face_adj = get_edge_infos(topo_explorer, face_infos, faces)

    return face_infos, edge_infos
    # for (i, item) in face_infos.items():
    #     print(item.hash, "-----", item.face_type, item.face_normal)
    #
    # print("============")
    #
    # for i, item in edge_infos.items():
    #     print(item.hash, "-----", item.edge_type, "face hash:", face_infos[item.face_hashes[0]].hash,
    #           face_infos[item.face_hashes[0]].face_normal)
    #
    # print(face_adj)
    # print(len(faces))
    # edges = list(topo_explorer.edges())
    # print(len(edges))
    # for edge in edges:
    #     neighbor_faces = list(topo_explorer.faces_from_edge(edge))
    #     # print(len(neighbor_faces))
    #     # print(hash(edge))
    #     if len(neighbor_faces) == 1:
    #         related_edges = topo_explorer.edges_from_face(neighbor_faces[0])
    #         print(BRepAdaptor_Surface(neighbor_faces[0], True).GetType())
    #         for related_edge in related_edges:
    #             print(BRepAdaptor_Curve(related_edge).GetType())

    pass


INVALID_IDS = ['0074/00745817', '0086/00866760']


def process_one(data_id):
    """
    This function processes a single data_id. It first checks if the data_id is in the list of INVALID_IDS.
    If it is, the function prints a message and returns. If not, it proceeds to process the data_id.

    The function constructs the save_path and step_path using the data_id. It then reads the step file
    and retrieves face and edge information.

    If the directory for the save_path does not exist, it creates it. Finally, it writes the face and edge
    information to a h5 file at the save_path.

    Args:
        data_id (str): The data_id to be processed.

    Returns:
        None
    """
    # Check if data_id is in the list of invalid ids
    if data_id in INVALID_IDS:
        print('skip {} in invalid ids'.format(data_id))
        return

    # Construct the data_id, save_path and step_path
    data_id = os.path.join(data_id.split('/')[0], data_id.split('/')[1])
    save_path = os.path.join(SAVE_DIR, data_id + '.h5')
    step_path = os.path.join(STEP_DIR, data_id + '.step')

    # Read the step file and retrieve face and edge information\
    try:
        face_infos, edge_infos = read_step(step_path)
    except Exception:
        print(Exception)
        return

        # Create the directory for save_path if it does not exist
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    # Write the face and edge information to a h5 file at the save_path
    write_h5file(save_path, list(face_infos.values()), list(edge_infos.values()))


DATA_DIR = '../../data'
SAVE_DIR = os.path.join(DATA_DIR, 'h5file')
STEP_DIR = os.path.join(DATA_DIR, 'step')
RECORD_FILE = os.path.join(DATA_DIR, 'train_val_test_split.json')

if __name__ == '__main__':
    # attrs, edge_links = load_h5file('0000/00000172')
    # edge_links = to_undirected(torch.tensor(edge_links, dtype=torch.long).t().contiguous())
    # edge_links = add_self_loops(edge_links, num_nodes=attrs.shape[0])
    # data = data.Data(x=torch.tensor(attrs, dtype=torch.float32),
    #                  edge_index=edge_links[0])
    # print(data)
    # process_one('0074/00745817')
    # process_one('0000/00000251')

    with open(RECORD_FILE, 'r') as f:
        all_data = json.load(f)
    pbar = tqdm.tqdm(all_data['train'])
    for x in pbar:
        pbar.set_description('processing train' + x)
        process_one(x)
    pbar = tqdm.tqdm(all_data['validation'])
    for x in pbar:
        pbar.set_description('processing validation' + x)
        process_one(x)
    pbar = tqdm.tqdm(all_data['test'])
    for x in pbar:
        pbar.set_description('processing test' + x)
        process_one(x)
