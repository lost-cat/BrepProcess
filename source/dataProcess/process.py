import os.path

import h5py
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer

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
        if len(faces) == 1:
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


def write_h5file(h5_path, step_name, face_infos: list[FaceInfo], edge_infos: list[EdgeInfo]):
    file = h5py.File(h5_path, 'w')
    group = file.create_group(step_name)

    group.create_dataset("face_normals", data=np.array([face_info.face_normal for face_info in face_infos]))
    group.create_dataset('face_centroids', data=np.array([face_info.centroid for face_info in face_infos]))
    group.create_dataset('face_areas', data=np.array([face_info.face_area for face_info in face_infos]))
    group.create_dataset('face_types', data=np.array([face_info.face_type for face_info in face_infos]))

    group.create_dataset('edge_links', data=np.array([edge_info.face_tags for edge_info in edge_infos]))
    group.create_dataset('edge_types', data=np.array([edge_info.edge_type for edge_info in edge_infos]))
    # group.create_dataset('edge_convexity', data=np.array([edge_info.convexity for edge_info in edge_infos]))
    file.close()
    pass


def read_h5file(h5_path, step_name):
    file = h5py.File(h5_path, 'r')
    group = file.get(step_name)
    face_normals = np.array(group.get("face_normals"))
    face_centroids = np.array(group.get("face_centroids"))
    face_areas = np.array(group.get("face_areas")).reshape(-1, 1)
    face_types = np.array(group.get("face_types")).reshape(-1, 1)
    edge_links = np.array(group.get("edge_links"))

    attri = np.concatenate((face_normals, face_centroids, face_areas, face_types), axis=1)
    return attri, edge_links
    pass


def read_step(filepath):
    if os.path.exists(filepath) is None:
        print(filepath, "not exists")

    reader = STEPControl_Reader()
    reader.ReadFile(filepath)
    reader.TransferRoot()
    shape = reader.OneShape()

    transfer_reader = reader.WS().TransferReader()
    topo_explorer = TopologyExplorer(shape)
    faces = list(topo_explorer.faces())
    face_infos = get_face_infos(faces)
    for (i, item) in face_infos.items():
        print(item.hash, "-----", item.face_type, item.face_normal)

    print("============")
    edge_infos, face_adj = get_edge_infos(topo_explorer, face_infos, faces)

    for i, item in edge_infos.items():
        print(item.hash, "-----", item.edge_type, "face hash:", face_infos[item.face_hashes[0]].hash,
              face_infos[item.face_hashes[0]].face_normal)

    print(face_adj)
    filename = filepath.split('/')[-1].split('.')[0]
    write_h5file("../../data/h5Path/" + filename + ".h5", filename, list(face_infos.values()),
                 list(edge_infos.values()))
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


if __name__ == '__main__':
    path = "../../data/step/0000/00000007.step"
    read_step(path)


# attrs, edge_links = read_h5file("../../data/h5Path/" + 'test' + ".h5",
#                                 'test')
# edge_links = to_undirected(torch.tensor(edge_links, dtype=torch.long).t().contiguous())
# edge_links = add_self_loops(edge_links, num_nodes=attrs.shape[0])
# data = data.Data(x=torch.tensor(attrs, dtype=torch.float32),
#                  edge_index=edge_links[0])
