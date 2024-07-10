import json
import os

from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_GTransform, BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer
from OCC.Core._STEPControl import STEPControl_AsIs
from OCC.Core.gp import gp_GTrsf, gp_Vec, gp_Trsf, gp_Pnt

step_path = '../data/test.step'
step_reader = STEPControl_Reader()
step_reader.ReadFile(step_path)
step_reader.TransferRoot()
shape = step_reader.OneShape()


# Calculate the current bounding box




normalize(shape)

# Optional: Translate the shape to center it in the [-1, 1] bounding box
# Calculate the translation needed after scaling

# # Save the final shape
# step_writer = STEPControl_Writer()
# step_writer.Transfer(final_shape, STEPControl_AsIs)
# status = step_writer.Write('../data/test_scaled.step')
