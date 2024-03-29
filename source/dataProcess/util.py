import math
import os
from typing import List

import dgl
import h5py
import numpy as np
import torch
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer

from source.dataProcess.dataStructure import EdgeInfo, FaceInfo


def get_path_by_data_id(data_id, prefix='none', ext='.step'):
    data_id = os.path.join(data_id.split('/')[0], data_id.split('/')[1])
    return os.path.join(prefix, data_id + ext)


# todo  change to occwl
def read_step(filepath):
    if not os.path.exists(filepath):
        print('file not exists', filepath)
        raise Exception()
    reader = STEPControl_Reader()
    reader.ReadFile(filepath)
    reader.TransferRoot()
    shape = reader.OneShape()
    topo_explorer = TopologyExplorer(shape)
    faces = list(topo_explorer.faces())
    face_infos = get_face_infos(faces)
    edge_infos, face_adj = get_edge_infos(topo_explorer, face_infos, faces)

    return face_infos, edge_infos


IDENTITY = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def convert_to_dgl_graph(face_infos: List[FaceInfo], edge_infos: List[EdgeInfo]):
    src = [e.face_tags[0] for e in edge_infos]
    dst = [e.face_tags[1] for e in edge_infos]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(face_infos))

    face_normals = np.array([face_info.face_normal for face_info in face_infos])
    face_centroids = np.array([face_info.centroid for face_info in face_infos])
    face_areas = np.array([face_info.face_area for face_info in face_infos]).reshape(-1, 1)
    faces_types = []
    for face_info in face_infos:
        faces_types.append(IDENTITY[face_info.face_type])

    face_attrs = np.concatenate((face_normals, face_centroids, face_areas, np.stack(faces_types, axis=0)), axis=1)
    dgl_graph.ndata['x'] = torch.from_numpy(face_attrs)
    uv_face_attrs = np.array([face_info.uv_face_attr for face_info in face_infos])
    dgl_graph.ndata['uv_attrs'] = torch.from_numpy(uv_face_attrs)

    edge_types = []
    for edge_info in edge_infos:
        edge_types.append(IDENTITY[edge_info.edge_type])
    edge_length = [e.length for e in edge_infos]
    edge_radius = [e.radius for e in edge_infos]
    edge_start_points = [e.start_point for e in edge_infos]
    edge_end_points = [e.end_point for e in edge_infos]

    edge_attrs = np.concatenate((np.stack(edge_types, axis=0), np.array(edge_length).reshape(-1, 1),
                                 np.array(edge_radius).reshape(-1, 1), np.array(edge_start_points),
                                 np.array(edge_end_points)), axis=1)
    dgl_graph.edata['x'] = torch.from_numpy(edge_attrs)

    uv_edge_attrs = np.array([edge_info.edge_uv_attr for edge_info in edge_infos])
    dgl_graph.edata['uv_attrs'] = torch.from_numpy(uv_edge_attrs)
    return dgl_graph


def write_h5file(h5_path, face_infos: List[FaceInfo], edge_infos: List[EdgeInfo]):
    file = h5py.File(h5_path, 'w')

    # Create datasets for face information
    file.create_dataset("face_normals", data=np.array([face_info.face_normal for face_info in face_infos]))
    file.create_dataset('face_centroids', data=np.array([face_info.centroid for face_info in face_infos]))
    file.create_dataset('face_areas', data=np.array([face_info.face_area for face_info in face_infos]))
    faces_types = []
    for face_info in face_infos:
        faces_types.append(IDENTITY[face_info.face_type])

    file.create_dataset('face_types', data=np.stack(faces_types, axis=0))
    # Create datasets for edge information
    file.create_dataset('edge_links', data=np.array([edge_info.face_tags for edge_info in edge_infos]))
    # use one-hot encoding
    edge_types = []
    for edge_info in edge_infos:
        edge_types.append(IDENTITY[edge_info.edge_type])
    file.create_dataset('edge_types', data=np.stack(edge_types, axis=0))
    length_or_radius = []
    for edge_info in edge_infos:
        if edge_info.edge_type == 0:
            length_or_radius.append(edge_info.length)
        elif edge_info.edge_type == 1:
            length_or_radius.append(edge_info.radius)
        else:
            length_or_radius.append(-1)

    file.create_dataset('length_or_radius', data=np.array(length_or_radius).reshape(-1, 1))
    file.create_dataset('edge_start_points', data=np.array([edge_info.start_point for edge_info in edge_infos]))
    file.create_dataset('edge_end_points', data=np.array([edge_info.end_point for edge_info in edge_infos]))

    # Close the h5 file
    file.close()


def load_h5file(data_id, data_dir):
    h5_path = get_path_by_data_id(data_id, data_dir, '.h5')
    file = h5py.File(h5_path, 'r')
    face_normals = np.array(file.get("face_normals"))
    face_centroids = np.array(file.get("face_centroids"))
    face_areas = np.array(file.get("face_areas")).reshape(-1, 1)
    face_types = np.array(file.get("face_types")).reshape(-1, 3)
    edge_links = np.array(file.get("edge_links"))
    edge_types = np.array(file.get("edge_types")).reshape(-1, 3)
    length_or_radius = np.array(file.get("length_or_radius")).reshape(-1, 1)
    edge_start_points = np.array(file.get("edge_start_points"))
    edge_end_points = np.array(file.get("edge_end_points"))

    attrs = np.concatenate((face_normals, face_centroids, face_areas, face_types), axis=1)

    edge_attrs = np.concatenate((edge_types, length_or_radius, edge_start_points, edge_end_points), axis=1)
    return attrs, edge_links, edge_attrs
    pass


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


def is_invalid(x, should_be_positive=False):
    if type(x) is float or int:
        if should_be_positive:
            return math.isnan(x) or math.isinf(x) or x <= 0
        else:
            return math.isnan(x) or math.isinf(x)
    elif type(x) is np.ndarray:
        if should_be_positive:
            return np.isnan(x).any() or np.isinf(x).any() or (x <= 0).any()
        else:
            return np.isnan(x).any() or np.isinf(x).any()


def check_data(face_list, edge_list) -> bool:
    if len(face_list) == 0 or len(edge_list) == 0:
        return False
    for face in face_list:
        for f in face.face_normal:
            if is_invalid(f):
                return False
        for f in face.centroid:
            if is_invalid(f):
                return False
        if is_invalid(face.face_area, True):
            return False
        for point in face.points:
            if is_invalid(point):
                print('face point invalid')
                return False
        for tangent in face.points_tangent:
            if is_invalid(tangent):
                print('face tangent invalid')
                return False
    for edge in edge_list:
        if edge.edge_type == 0:
            if is_invalid(edge.length, True):
                return False
        if edge.edge_type == 1:
            if is_invalid(edge.radius, True):
                return False
        for point in edge.start_point:
            if is_invalid(point):
                return False
        for point in edge.end_point:
            if is_invalid(point):
                return False
        for point in edge.points:
            if is_invalid(point):
                print('edge point invalid')
                return False
        for tangent in edge.points_tangent:
            if is_invalid(tangent):
                print('edge tangent invalid')
                return False
    return True
