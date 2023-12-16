from OCC.Core import BRepGProp
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_SurfaceType
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.gp import gp_Vec, gp_Pnt
from typing import Tuple


def get_face_type(face: TopoDS_Face) -> GeomAbs_SurfaceType:
    surface = BRepAdaptor_Surface(face, True)
    face_type = surface.GetType()
    return face_type


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

def get_edge_type(edge):
    curve = BRepAdaptor_Curve(edge)
    return curve.GetType()


def get_edge_convexity(edge):
    pass


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
