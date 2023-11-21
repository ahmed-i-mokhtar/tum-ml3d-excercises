"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    f = open(path, "w")
    for vertex in vertices:
        f.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))

    for face in faces:     
        f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))

    f.close()

    print("Exported mesh object to {}.".format(path))
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    f = open(path, "w")
    for x, y, z in pointcloud:
        f.write("v {} {} {}\n".format(x, y, z))

    f.close()

    print("Exported point cloud object to {}.".format(path))
    # ###############
