from enum import IntEnum

import numpy as np
import occwl.face
from OCC.Core.GeomAbs import GeomAbs_Line
from occwl.edge import Edge
from occwl.uvgrid import uvgrid, ugrid


class FaceType(IntEnum):
    PLANE = 0
    CYLINDER = 1
    OTHER = 2


class EdgeType(IntEnum):
    LINE = 0
    CIRCLE = 1
    OTHER = 2


def get_face_type(face: occwl.face.Face) -> FaceType:
    face_type = face.surface_type()
    if face_type == "plane":
        return FaceType.PLANE
    elif face_type == "cylinder":
        return FaceType.CYLINDER
    else:
        print('face_type:', face_type)
        return FaceType.OTHER


def get_surface_area_and_centroid(face: occwl.face.Face):
    face_area = face.area()
    centroid = face.point(face.uv_bounds().center())
    return face_area, centroid


def get_face_normal(face: occwl.face.Face):
    uv_bound = face.uv_bounds()
    return face.normal(uv_bound.center())


def get_face_uv_grid(occwl_face, num_u=10, num_v=10):
    points = uvgrid(
        occwl_face, method="point", num_u=num_u, num_v=num_v, reverse_order_with_face=True
    )
    normals = uvgrid(
        occwl_face, method="normal", num_u=num_u, num_v=num_v, reverse_order_with_face=True
    )
    visibility_status = uvgrid(
        occwl_face, method="visibility_status", num_u=num_u, num_v=num_v, reverse_order_with_face=True
    )
    mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
    uv_face_attr = np.concatenate((points, normals, mask), axis=-1)
    return uv_face_attr


class FaceInfo:
    face_normal: np.ndarray
    centroid: np.ndarray
    face_area: float
    face_type: FaceType

    def __init__(self, occ_face, index):
        super().__init__()
        self.occ_face = occ_face
        self.index = index
        self.face_normal = get_face_normal(occ_face)

        self.face_type = get_face_type(occ_face)
        self.face_area, self.centroid = get_surface_area_and_centroid(occ_face)
        self.uv_face_attr = get_face_uv_grid(occ_face)

    def print_data(self):
        print(
            f'face_normal: {self.face_normal}, centroid: {self.centroid}, face_area: {self.face_area}, '
            f'face_type: {self.face_type} ')


def get_edge_type(occwl_edge: occwl.edge.Edge):
    curve_type = occwl_edge.curve_type_enum()
    if curve_type == GeomAbs_Line:
        return EdgeType.LINE
    elif curve_type == 1:
        return EdgeType.CIRCLE
    else:
        print('edge_type:', curve_type)
        return EdgeType.OTHER


def get_edge_length(occwl_edge: occwl.edge.Edge):
    return occwl_edge.length()
    pass


def get_circle_radius(occwl_edge: occwl.edge.Edge):
    if occwl_edge.curve_type_enum() == 1:
        circle = occwl_edge.specific_curve()
        return circle.Radius()
    else:
        return -1


pass


def get_start_end_points(occwl_edge: occwl.edge.Edge):
    start_point = occwl_edge.start_vertex().point()
    end_point = occwl_edge.end_vertex().point()
    return start_point, end_point


def get_edge_ugrid(occwl_edge: occwl.edge.Edge, num_u=10):
    points = ugrid(occwl_edge, method="point", num_u=num_u)
    tangents = ugrid(occwl_edge, method="tangent", num_u=num_u)
    edge_uv_attr = np.concatenate((points, tangents), axis=-1)
    return edge_uv_attr


class EdgeInfo:

    def __init__(self, edge, index):
        self.index = index
        self.edge = edge
        self.faces = []
        self.face_hashes = []
        self.face_tags = []
        self.edge_type = get_edge_type(edge)
        self.length = get_edge_length(edge)
        self.radius = get_circle_radius(edge)
        self.start_point, self.end_point = get_start_end_points(edge)
        self.edge_uv_attr = get_edge_ugrid(edge)

    def print_data(self):
        print(f'edge_type: {self.edge_type}, faces: {self.face_tags}')
