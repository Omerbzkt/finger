# face reconstruction with ball pivoting
import open3d as o3d
import numpy as np
import trimesh
input_path = 'C:/Users/PC/Desktop/000_all_face.obj'
mesh = o3d.io.read_triangle_mesh(input_path)


mesh2 = trimesh.load_mesh(input_path)
vertex_colors = np.asarray(mesh.vertex_colors)
faces = np.asarray(mesh2.faces)
face_colors = vertex_colors[faces]


mesh.compute_vertex_normals()
pcd = mesh.sample_points_poisson_disk(15000)

radii = [0.005, 0.01, 0.02, 0.03]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

face_colors = face_colors.reshape(-1, 3)

rec_mesh.vertex_colors = o3d.utility.Vector3dVector(face_colors)

o3d.visualization.draw_geometries([rec_mesh])

# take the color information of the file
'''
input_path = 'C:/Users/PC/Finger print/obj_file_separated_for_segmentation/002_mouth_colored.obj'
mesh = o3d.io.read_triangle_mesh(input_path)
mesh2 = trimesh.load_mesh(input_path)

vertex_colors = np.asarray(mesh.vertex_colors)

faces = np.asarray(mesh2.faces)

face_colors = vertex_colors[faces]

flat_face_colors = face_colors.reshape(-1, face_colors.shape[-1])
'''
