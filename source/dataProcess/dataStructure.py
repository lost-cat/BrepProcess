from enum import IntEnum

from OCC.Core import BRepGProp
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties

from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_SurfaceType, GeomAbs_Line
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.gp import gp_Vec, gp_Pnt
from typing import Tuple


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


class FaceInfo:
    face_normal: Tuple[float, float, float]
    centroid: Tuple[float, float, float]
    face_area: float
    face_type: GeomAbs_SurfaceType

    def __init__(self, face, index):
        super().__init__()
        self.face = face
        self.index = index
        self.hash = hash(face)
        self.face_normal = get_face_normal(face)

        self.face_type = get_face_type(face)
        self.face_area, self.centroid = get_surface_area_and_centroid(face)

        # BRepGProp.BRepGProp_Face(self.face).

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


def get_line_length(edge):
    curve = BRepAdaptor_Curve(edge)
    if curve.GetType() == 0:
        props = GProp_GProps()
        brepgprop_LinearProperties(edge, props)
        return props.Mass()
    pass


def get_circle_radius(edge):
    curve = BRepAdaptor_Curve(edge)
    if curve.GetType() == 1:
        circle = curve.Circle()
        return circle.Radius()


pass


def get_start_end_points(edge):
    curve = BRepAdaptor_Curve(edge)
    start_point = curve.Value(curve.FirstParameter())
    end_point = curve.Value(curve.LastParameter())
    return start_point.Coord(), end_point.Coord()


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
        self.length = get_line_length(edge)
        self.radius = get_circle_radius(edge)
        self.start_point, self.end_point = get_start_end_points(edge)

    def print_data(self):
        print(f'edge_type: {self.edge_type}, convexity: {self.convexity}, faces: {self.face_tags}')
