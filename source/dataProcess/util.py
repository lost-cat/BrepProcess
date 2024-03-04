import math
import os
from typing import List

import h5py
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer

from source.dataProcess.dataStructure import EdgeInfo, FaceInfo


def get_path_by_data_id(data_id, prefix='none', ext='.step'):
    data_id = os.path.join(data_id.split('/')[0], data_id.split('/')[1])
    return os.path.join(prefix, data_id + ext)


def read_step(filepath):
    """
    This function reads a STEP file and extracts face and edge information.
    The function returns the face information and edge information.

    Args:
        filepath (str): The path to the STEP file to be read.

    Returns:
        face_infos (dict): A dictionary where the keys are hashes of the faces and the values are FaceInfo objects.
        edge_infos (dict): A dictionary where the keys are hashes of the edges and the values are EdgeInfo objects.
    """
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


def write_h5file(h5_path, face_infos: List[FaceInfo], edge_infos: List[EdgeInfo]):
    """
    This function writes face and edge information to a h5 file.

    It  creates datasets for face normals,face centroids, face areas, and face types using the face information. It
    also creates datasets for edge links and edge types using the edge information.

    Args:
        h5_path (str): The path to the h5 file to be written.
        face_infos (list[FaceInfo]): A list of FaceInfo objects containing face information.
        edge_infos (list[EdgeInfo]): A list of EdgeInfo objects containing edge information.

    Returns:
        None
    """

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
        else:
            length_or_radius.append(edge_info.radius)

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
    """
    This function retrieves edge information from a topology explorer object and a list of faces.

    It iterates over the edges in the topology explorer object. For each edge, it retrieves the faces associated with
    the edge. If the edge is not associated with exactly two faces, it skips the edge.

    It creates an EdgeInfo object for the edge and appends the hashes and indices of the associated faces to the
    EdgeInfo object. It also updates a face adjacency matrix to indicate the adjacency of the faces associated with
    the edge.

    It adds the EdgeInfo object to a dictionary where the keys are the hashes of the edges and the values are the
    EdgeInfo objects.

    Args:
        topo (TopologyExplorer): The topology explorer object to retrieve edges from.
        face_infos (dict): A dictionary where the keys are hashes of the faces and the values are FaceInfo objects.
        occ_faces (list): A list of faces.

    Returns:
        edge_infos (dict): A dictionary where the keys are hashes of the edges and the values are EdgeInfo objects.
        face_adj (numpy.ndarray): A 2D numpy array representing the face adjacency matrix.
    """
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


def check_data(face_list, edge_list) -> bool:
    if len(face_list) == 0 or len(edge_list) == 0:
        return False
    for face in face_list:
        for f in face.face_normal:
            if math.isnan(f) or math.isinf(f):
                return False
        for f in face.centroid:
            if math.isnan(f) or math.isinf(f):
                return False
        if math.isnan(face.face_area) or math.isinf(face.face_area) or face.face_area <= 0:
            return False
    return True
