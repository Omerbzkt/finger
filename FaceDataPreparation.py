# create mesh using vertices and faces
import open3d as o3d
import os
import trimesh
from vedo import *
import numpy as np

# mesh = trimesh.load_mesh('obj_file_seperated/000_mouth_colored.obj')
# vertices = mesh.vertices
# faces = mesh.faces

vertices = np.load("updated_npy_files_vertices/000_eye_colored_vertices.npy")
faces = np.load("updated_npy_files_faces/000_eye_colored_faces.npy")

colored_mesh = o3d.geometry.TriangleMesh()
colored_mesh.vertices = o3d.utility.Vector3dVector(vertices)
colored_mesh.triangles = o3d.utility.Vector3iVector(faces)

output_file = "output.obj"

o3d.io.write_triangle_mesh(output_file, colored_mesh)

mesh = Mesh('output.obj')
mesh.show()


# ayrılmış obj dosyalarını faces ve verticeslarını alarak ayrı iki dosyaya kaydedilmesi
import trimesh
import os
import numpy as np

# obj files
obj_directory = 'obj_file_seperated/'

# npy files documents
output_directory = 'output_npy_files_faces/'
output_directory_2 = 'output_npy_files_vertices/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(output_directory_2):
    os.makedirs(output_directory_2)
    
for root, dirs, files in os.walk(obj_directory):
    for file in files:
        if file.endswith('.obj'):
            obj_file = os.path.join(root, file)
            mesh = trimesh.load_mesh(obj_file)

            vertices = mesh.vertices
            faces = mesh.faces

            vertices_output_filename = os.path.join(output_directory_2, os.path.splitext(file)[0] + '_vertices.npy')
            faces_output_filename = os.path.join(output_directory, os.path.splitext(file)[0] + '_faces.npy')

            np.save(vertices_output_filename, vertices)
            np.save(faces_output_filename, faces)


print("Process finished!")


# npy dosyalarındaki verileri en yüksek shape a sahip veriye eşitleme
import os
import numpy as np

npy_dir = "output_npy_files_faces"
output_dir = "updated_npy_files_faces"

max_shape = 0

for npy_file in os.listdir(npy_dir):
    if npy_file.endswith(".npy"):
        file_path = os.path.join(npy_dir, npy_file)
        data = np.load(file_path)
        if data.shape[0] > max_shape:
            max_shape = data.shape[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for npy_file in os.listdir(npy_dir):
    if npy_file.endswith(".npy"):
        file_path = os.path.join(npy_dir, npy_file)
        data = np.load(file_path)
        if data.shape[0] < max_shape:
            pad_width = ((0, max_shape - data.shape[0]), (0, 0))
            data = np.pad(data, pad_width, mode='constant')
            updated_file_path = os.path.join(output_dir, npy_file)
            np.save(updated_file_path, data)

print("Tüm işlemler tamamlandı.")
