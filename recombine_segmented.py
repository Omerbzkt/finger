# segmente edilmiş obj dosyalarının yeniden birleştirilmesi
import os
import open3d as o3d

def load_obj(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    return mesh

def save_obj(output_directory, file_name, mesh):
    output_path = os.path.join(output_directory, file_name)
    o3d.io.write_triangle_mesh(output_path, mesh)


# birleştir ve kaydet
def merge_and_save(base_directory, output_directory):
    file_list = os.listdir(base_directory)
    # dosya yolundaki bütün dosyaları al 5 dosyaya ulaşınca birleştir
    merged_mesh = None
    file_count = 0

    for file_name in sorted(file_list):
        if file_name.endswith("_colored.obj"):
            file_path = os.path.join(base_directory, file_name)
            mesh_section = load_obj(file_path)

            if merged_mesh is None:
                merged_mesh = mesh_section
            else:
                merged_mesh += mesh_section

            file_count += 1

            if file_count == 5:
                save_obj(output_directory, f"Merged_Output_{file_name[:3]}.obj", merged_mesh)
                merged_mesh = None
                file_count = 0

    if merged_mesh is not None:
        save_obj(output_directory, f"Merged_Output_{file_name[:3]}.obj", merged_mesh)

        
# dosya yolları
base_directory = "C:/Users/PC/Finger print/obj_file_separated_for_segmentation/"
output_directory = "C:/Users/PC/Finger print/obj_file_separated_for_segmentation/Merged/"

merge_and_save(base_directory, output_directory)
