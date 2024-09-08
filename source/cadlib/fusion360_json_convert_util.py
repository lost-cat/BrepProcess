import numpy as np

from source.cadlib.curves import Arc, Line, Circle
from source.cadlib.extrude import Extrude, CADSequence
from source.cadlib.macro import EXTRUDE_OPERATIONS, EXTENT_TYPE


def get_point(stat):
    assert stat['type'] == 'Point3D'
    return np.array([stat['x'], stat['y'], stat['z']])


def get_2d_point(point):
    assert point['type'] == 'Point3D'
    return np.array([point['x'], point['y']])


def get_vector(stat, normalize=False):
    assert stat['type'] == 'Vector3D'
    vec = np.array([stat['x'], stat['y'], stat['z']])
    if normalize:
        vec = vec / np.linalg.norm(vec)
    return vec


def get_2d_vector(stat):
    assert stat['type'] == 'Vector3D'
    return np.array([stat['x'], stat['y']])


def project_point_to_plane(point, plane):
    assert plane['type'] == 'Plane'
    origin = get_point(plane['origin'])
    u_direction = get_vector(plane['u_direction'])
    u_length = plane['u_direction']['length']
    v_direction = get_vector(plane['v_direction'])
    v_length = plane['v_direction']['length']
    op = point - origin
    x = np.dot(op, u_direction) / u_length
    y = np.dot(op, v_direction) / v_length

    return np.array([x, y])


def from_fusion360_dict_Arc(stat):
    assert stat['type'] == 'Arc3D'
    start_point = get_2d_point(stat['start_point'])
    end_point = get_2d_point(stat['end_point'])
    center_point = get_2d_point(stat['center_point'])
    radius = stat['radius']
    normal = get_vector(stat['normal'], normalize=True)
    start_angle = stat['start_angle']
    end_angle = stat['end_angle']
    ref_vec = get_2d_vector(stat['reference_vector'])
    return Arc(start_point, end_point, center_point, radius, normal,
               start_angle, end_angle, ref_vec)

    pass


def from_fusion360_dict_Line(stat):
    assert stat['type'] == 'Line3D'
    start_point = get_2d_point(stat['start_point'])
    end_point = get_2d_point(stat['end_point'])
    return Line(start_point, end_point)


def from_fusion360_dict_Circle(stat):
    assert stat['type'] == 'Circle3D'
    center_point = get_2d_point(stat['center_point'])
    normal = get_vector(stat['normal'], normalize=True)
    radius = stat['radius']
    return Circle(center_point, radius, normal)


def get_curve_from_dict(item):
    if item['type'] == 'Line3D':
        return from_fusion360_dict_Line(item)
    elif item['type'] == 'Arc3D':
        return from_fusion360_dict_Arc(item)
    elif item['type'] == 'Circle3D':
        return from_fusion360_dict_Circle(item)
    else:
        raise ValueError(f"Unknown curve type {item['type']}")


def from_fusion360_dict_Loop(stat):
    all_curves = [get_curve_from_dict(item) for item in stat['profile_curves']]
    from source.cadlib.sketch import Loop
    this_loop = Loop(all_curves)
    this_loop.is_outer = stat['is_outer']
    return this_loop


def from_fusion360_dict_Profile(stat):
    all_loops = [from_fusion360_dict_Loop(loop) for loop in stat['loops']]
    from source.cadlib.sketch import Profile
    return Profile(all_loops)


def from_fusion360_dict_CoordSystem(sketch, profile_id):
    assert sketch['type'] == 'Sketch'
    from source.cadlib.math_utils import polar_parameterization
    from source.cadlib.extrude import CoordSystem
    from source.cadlib.math_utils import cartesian2polar

    transform = sketch['transform']

    origin = get_point(transform['origin'])
    # origin = origin + np.array([0, 0, sketch['points'][1]['z']])
    z = get_z_distance(sketch, profile_id)

    normal_3d = get_vector(transform['z_axis'], normalize=True)
    x_axis_3d = get_vector(transform['x_axis'], normalize=True)
    y_axis_3d = get_vector(transform['y_axis'], normalize=True)

    origin += z * normal_3d

    theta, phi, gamma = polar_parameterization(normal_3d, x_axis_3d)
    return CoordSystem(origin, theta, phi, gamma, y_axis=cartesian2polar(y_axis_3d))


def get_z_distance(sket_entity, profile_id):
    curve = sket_entity['profiles'][profile_id]['loops'][0]['profile_curves'][0]
    if curve['type'] == 'Line3D':
        return curve['start_point']['z']
    elif curve['type'] == 'Arc3D' or curve['type'] == 'Circle3D':
        return curve['center_point']['z']


def from_fusion360_dict_Extrude(all_stat, extrude_id, sketch_dim=256):
    extrude_entity = all_stat["entities"][extrude_id]
    assert extrude_entity["start_extent"]["type"] == "ProfilePlaneStartDefinition"

    all_skets = []
    n = len(extrude_entity["profiles"])
    for i in range(len(extrude_entity["profiles"])):
        sket_id, profile_id = extrude_entity["profiles"][i]["sketch"], extrude_entity["profiles"][i]["profile"]
        sket_entity = all_stat["entities"][sket_id]
        sket_profile = from_fusion360_dict_Profile(sket_entity["profiles"][profile_id])

        sket_plane = from_fusion360_dict_CoordSystem(sket_entity, profile_id)
        # normalize profile to 256x256 bounding box
        point = sket_profile.start_point
        sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis + sket_plane.origin
        sket_size = sket_profile.bbox_size
        sket_profile.normalize(sketch_dim)
        all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))

    operation = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])

    extent_type = EXTENT_TYPE.index(extrude_entity["extent_type"])
    extent_one = extrude_entity["extent_one"]["distance"]["value"]
    extent_two = 0.0
    if extrude_entity["extent_type"] == "TwoSidesFeatureExtentType":
        extent_two = extrude_entity["extent_two"]["distance"]["value"]
    elif extrude_entity["extent_type"] == "SymmetricFeatureExtentType":
        extent_two = extent_one

    if operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation"):
        all_operations = [operation] + [EXTRUDE_OPERATIONS.index("JoinFeatureOperation")] * (n - 1)
    else:
        all_operations = [operation] * n

    return [Extrude(all_skets[i][0], all_skets[i][1], all_operations[i], extent_type, extent_one, extent_two,
                    all_skets[i][2], all_skets[i][3]) for i in range(n)]


def from_fusion360_dict_CADSequence(all_stat):
    seq = []
    for item in all_stat["timeline"]:
        entity_id = item["entity"]
        entity = all_stat["entities"][entity_id]
        if entity["type"] == "ExtrudeFeature":
            try:

                extrude_ops = from_fusion360_dict_Extrude(all_stat, item["entity"])
                seq.extend(extrude_ops)
            except Exception as e:
                print(f"Error in processing extrude {entity_id}")
                print(e)

    bbox_info = all_stat["properties"]["bounding_box"]
    max_point = np.array([bbox_info["max_point"]["x"], bbox_info["max_point"]["y"], bbox_info["max_point"]["z"]])
    min_point = np.array([bbox_info["min_point"]["x"], bbox_info["min_point"]["y"], bbox_info["min_point"]["z"]])
    bbox = np.stack([max_point, min_point], axis=0)
    return CADSequence(seq, bbox)
