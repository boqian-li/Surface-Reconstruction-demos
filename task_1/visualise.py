import numpy as np
import mcubes
import open3d as o3d
import matplotlib.pyplot as plt


sdf = np.load('../target/sdf_1.npy')
sdf = -sdf

vertices, triangles = mcubes.marching_cubes(sdf, 0)
vertices = (vertices / 99 - 0.5) * 1.3
mcubes.export_obj(vertices, triangles, '../target/tmp_1.obj')

'''
#../data/target.obj
mesh= o3d.io.read_triangle_mesh('../target/tmp_1.obj')
mesh_ = o3d.io.read_triangle_mesh('../data/target.obj')

o3d.visualization.draw_geometries([mesh, mesh_])

'''