# ben üsttekinin yapamadığını yapıyorum
import os
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from scipy.spatial import KDTree
import os 

def read_picked_points(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    points = []
    for point_element in root.findall('point'):
        x = float(point_element.get('x'))
        y = float(point_element.get('y'))
        z = float(point_element.get('z'))
        points.append([x, y, z])

    return np.array(points)

def find_nearest_point(source_points, target_points):
    source_tree = KDTree(target_points, leafsize=50)
    _, indices = source_tree.query(source_points)
    return target_points[indices]

def color_mesh_interior(mesh, lines, color):
    mesh_colors = np.zeros((len(mesh.vertices), 3))

    for face in mesh.triangles:
        a = mesh.vertices[face[0]]
        b = mesh.vertices[face[1]]
        c = mesh.vertices[face[2]]
        for vertex_id in face:
            p = mesh.vertices[vertex_id]
            if is_inside_lines(p, lines):
                mesh_colors[vertex_id] = color

    return mesh_colors

   
def is_inside_lines(p, lines):
    inside = False
    for line in lines:
        start_point = line[0]
        end_point = line[1]
        if ((start_point[1] > p[1]) != (end_point[1] > p[1])) and (
            p[0]
            < (end_point[0] - start_point[0]) * (p[1] - start_point[1])
            / (end_point[1] - start_point[1])
            + start_point[0]
        ):
            inside = not inside
    return inside

def extract_colored_triangles(mesh, lines, color):
    colored_triangles = []
    
    for face in mesh.triangles:
        a = mesh.vertices[face[0]]
        b = mesh.vertices[face[1]]
        c = mesh.vertices[face[2]]
        for vertex_id in face:
            p = mesh.vertices[vertex_id]
            if is_inside_lines(p, lines):
                colored_triangles.append(face)
                break  # Once a single vertex in the face is inside, add the whole face
                
    return colored_triangles

color_list = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
]

picked_points_filenames = [
    'forehead.pp',
    'eye.pp',
    'nose.pp',
    'mouth.pp',
    'chin.pp',
]

output_folder = 'obj_file_seperated_for_segmentation'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
obj_files_directory = 'obj_file'

obj_file_paths = [os.path.join(obj_files_directory, filename) for filename in os.listdir(obj_files_directory) if filename.endswith('.obj')]

for obj_file_path in obj_file_paths:
    target_mesh = o3d.io.read_triangle_mesh(obj_file_path)
    target_points = np.asarray(target_mesh.vertices)

    for idx, picked_points_filename in enumerate(picked_points_filenames):
        picked_points = read_picked_points(picked_points_filename)
        source_points = np.array(picked_points)

        matched_points = find_nearest_point(source_points, target_points)

        lines = np.array(
            [
                [matched_points[i], matched_points[(i + 1) % len(matched_points)]]
                for i in range(len(matched_points))
            ]
        )

        in_between_color = color_list[idx]
        colored_triangles = extract_colored_triangles(target_mesh, lines, in_between_color)

        colored_mesh = o3d.geometry.TriangleMesh()
        colored_mesh.vertices = target_mesh.vertices
        colored_mesh.triangles = o3d.utility.Vector3iVector(colored_triangles)

        base_obj_filename = os.path.splitext(os.path.basename(obj_file_path))[0]
        base_pp_filename = os.path.splitext(picked_points_filename)[0]
        output_filename = f"{base_obj_filename}_{base_pp_filename}_colored.obj"
        output_path = os.path.join(output_folder, output_filename)
        o3d.io.write_triangle_mesh(output_path, colored_mesh, write_vertex_colors = True)
