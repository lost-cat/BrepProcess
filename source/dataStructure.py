from enum import IntEnum
from typing import Tuple

import numpy as np
import occwl.face
from OCC.Core import BRepGProp
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.gp import gp_Vec, gp_Pnt
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


def get_face_type(face: TopoDS_Face) -> FaceType:
    surface = BRepAdaptor_Surface(face, True)
    face_type = surface.GetType()
    if face_type == 0:
        return FaceType.PLANE
    elif face_type == 1:
        return FaceType.CYLINDER

    else:
        print('face_type:', face_type)
        return FaceType.OTHER


def get_surface_area_and_centroid(face: TopoDS_Face):
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    face_area = props.Mass()
    centroid = props.CentreOfMass().Coord()
    return face_area, centroid


def get_face_normal(face: TopoDS_Face):
    analysis_face = BRepGProp.BRepGProp_Face(face)
    umin, umax, vmin, vmax = analysis_face.Bounds()
    midu = (umin + umax) / 2
    midv = (vmin + vmax) / 2
    norm = gp_Vec()
    mid_point = gp_Pnt()
    analysis_face.Normal(midu, midv, mid_point, norm)
    # if is_reverse:
    #     print('has reverse', norm.Coord())
    #     norm = norm.Reversed()
    norm = norm.Normalized()
    return norm.Coord()


def get_face_uv_grid(face, num_u=10, num_v=10):
    occwl_face = occwl.face.Face(face)
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
    face_normal: Tuple[float, float, float]
    centroid: Tuple[float, float, float]
    face_area: float
    face_type: FaceType

    def __init__(self, face, index):
        super().__init__()
        self.face = face
        self.index = index
        self.hash = hash(face)
        self.face_normal = get_face_normal(face)

        self.face_type = get_face_type(face)
        self.face_area, self.centroid = get_surface_area_and_centroid(face)
        self.uv_face_attr = get_face_uv_grid(face)

    def print_data(self):
        print(
            f'face_normal: {self.face_normal}, centroid: {self.centroid}, face_area: {self.face_area}, '
            f'face_type: {self.face_type} ')


def get_edge_type(edge):
    curve = BRepAdaptor_Curve(edge)
    if curve.GetType() == GeomAbs_Line:
        return EdgeType.LINE
    elif curve.GetType() == 1:
        return EdgeType.CIRCLE
    else:
        print('edge_type:', curve.GetType())
        return EdgeType.OTHER


def get_edge_length(edge):
    occwl_edge = occwl.edge.Edge(edge)
    return occwl_edge.length()
    pass


def get_circle_radius(edge):
    curve = BRepAdaptor_Curve(edge)
    if curve.GetType() == 1:
        circle = curve.Circle()
        return circle.Radius()
    else:
        return -1


pass


def get_start_end_points(edge):
    occwl_edge = occwl.edge.Edge(edge)
    start_point = occwl_edge.start_vertex().point()
    end_point = occwl_edge.end_vertex().point()
    return start_point, end_point


def get_edge_ugrid(edge, num_u=10):
    occwl_edge = occwl.edge.Edge(edge)
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
        # Convex = 0, Concave = 1, Other = 2
        self.convexity = None
        self.edge_type = get_edge_type(edge)
        self.hash = hash(edge)
        self.length = get_edge_length(edge)
        self.radius = get_circle_radius(edge)
        self.start_point, self.end_point = get_start_end_points(edge)
        self.edge_uv_attr = get_edge_ugrid(edge)

    def print_data(self):
        print(f'edge_type: {self.edge_type}, convexity: {self.convexity}, faces: {self.face_tags}')
