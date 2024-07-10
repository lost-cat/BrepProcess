import math
import os
from typing import List

import dgl
import h5py
import numpy as np
import occwl.io
import torch
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Trsf, gp_Pnt, gp_Vec
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.graph import face_adjacency

from source.dataStructure import EdgeInfo, FaceInfo


def get_path_by_data_id(data_id, prefix='none', ext='.step_path'):
    data_id = os.path.join(data_id.split('/')[0], data_id.split('/')[1])
    return os.path.join(prefix, data_id + ext)


def normalize(shape, center=True):
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    # Calculate the scale factor
    scale_x = 2.0 / (xmax - xmin)
    scale_y = 2.0 / (ymax - ymin)
    scale_z = 2.0 / (zmax - zmin)
    scale_factor = min(scale_x, scale_y, scale_z)
    # Scale the shape
    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor)
    scaled_shape = BRepBuilderAPI_Transform(shape, scale_trsf).Shape()
    if center:
        bbox = Bnd_Box()
        brepbndlib_Add(scaled_shape, bbox)
        new_xmin, new_ymin, new_zmin, new_xmax, new_ymax, new_zmax = bbox.Get()
        translate_x = -0.5 * (new_xmax + new_xmin)
        translate_y = -0.5 * (new_ymax + new_ymin)
        translate_z = -0.5 * (new_zmax + new_zmin)
        # Apply the translation
        translate_trsf = gp_Trsf()
        translate_trsf.SetTranslation(gp_Vec(translate_x, translate_y, translate_z))
        scaled_shape = BRepBuilderAPI_Transform(scaled_shape, translate_trsf).Shape()
    return scaled_shape


def read_step(filepath, normalized=False):
    if not os.path.exists(filepath):
        print('file not exists', filepath)
        raise Exception()
    compound = occwl.io.load_single_compound_from_step(filepath)
    if normalized:
        compound_scaled = compound.scale_to_box(1.0)
        return compound_scaled
    return compound


def get_face_edge_info(compound: Compound):
    mapper = EntityMapper(compound)
    occwl_faces = compound.faces()
    occwl_edges = compound.edges()
    face_infos = get_face_infos(occwl_faces, mapper)
    edge_infos = get_edge_infos(compound, mapper)
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

    # Create datasets for occ_face information
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


def get_face_infos(occ_faces, mapper):
    face_infos = {}
    for occ_face in occ_faces:
        face_info = FaceInfo(occ_face, mapper.face_index(occ_face))
        face_infos[face_info.hash] = face_info

    return face_infos


def get_edge_infos(compound: Compound, mapper):
    edge_infos = {}
    graph = occwl.graph.face_adjacency(compound, True)

    for u, v, data in graph.edges:
        edge = data['edge']
        edge_info = EdgeInfo(edge, mapper.edge_index(edge))
        edge_info.face_tags.extend([u, v])

    return edge_infos


def is_invalid(x, should_be_positive=False):
    if isinstance(x, float) or isinstance(x, int):
        if should_be_positive:
            return math.isnan(x) or math.isinf(x) or x <= 0
        else:
            return math.isnan(x) or math.isinf(x)
    elif isinstance(x, np.ndarray):

        if should_be_positive:
            return np.isnan(x).any() or np.isinf(x).any() or (x <= 0).any()
        else:
            return np.isnan(x).any() or np.isinf(x).any()
    else:
        raise Exception('unsupport data type', type(x))


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
        for uv_attr in face.uv_face_attr:
            if is_invalid(uv_attr):
                print('uv invalid')
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
        for uv_edge in edge.edge_uv_attr:
            if is_invalid(uv_edge):
                print('edge point invalid')
                return False

    return True
