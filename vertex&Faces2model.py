# Eğer modelin üçgen indekslerine ve nokta koordinatlarına sahipsek bu dönüşüm ile 3D yüz modeli elde edilmektedir.
def write_ply_file(vertices, triangles, filename):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(vertices)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face {}\n'.format(len(triangles)))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        
        for vertex in vertices:
            f.write('{} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))
        
        for triangle in triangles:
            f.write('3 {} {} {}\n'.format(triangle[0], triangle[1], triangle[2]))

output_filename = 'output.ply'
write_ply_file(vertices, triangles, output_filename)
