from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import trimesh
from urdfpytorch.utils import unparse_origin, parse_origin
from .meshlab_converter import ply_to_obj


def resolve_package_path(urdf_path, mesh_path):
    urdf_path = Path(urdf_path)
    search_dir = urdf_path.parent
    relative_path = Path(str(mesh_path).replace('package://', ''))
    while True:
        absolute_path = (search_dir / relative_path)
        if absolute_path.exists():
            return absolute_path
        search_dir = search_dir.parent


def extract_mesh_visuals(mesh):
    visuals = []
    graph = mesh.graph
    geometries = mesh.geometry
    for node_id, node_infos in graph.to_flattened().items():
        geometry = node_infos.get('geometry')
        if geometry is not None:
            visuals.append((geometries[geometry], node_infos['transform']))
    return visuals


def obj_to_urdf(obj_path, urdf_path):
    obj_path = Path(obj_path)
    urdf_path = Path(urdf_path)
    assert urdf_path.parent == obj_path.parent

    geometry = ET.Element('geometry')
    mesh = ET.SubElement(geometry, 'mesh')
    mesh.set('filename', obj_path.name)
    mesh.set('scale', '1.0 1.0 1.0')

    material = ET.Element('material')
    material.set('name', 'mat_part0')
    color = ET.SubElement(material, 'color')
    color.set('rgba', '1.0 1.0 1.0 1.0')

    inertial = ET.Element('inertial')
    origin = ET.SubElement(inertial, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '0.0 0.0 0.0')

    mass = ET.SubElement(inertial, 'mass')
    mass.set('value', '0.1')

    inertia = ET.SubElement(inertial, 'inertia')
    inertia.set('ixx', '1')
    inertia.set('ixy', '0')
    inertia.set('ixz', '0')
    inertia.set('iyy', '1')
    inertia.set('iyz', '0')
    inertia.set('izz', '1')

    robot = ET.Element('robot')
    robot.set('name', obj_path.with_suffix('').name)

    link = ET.SubElement(robot, 'link')
    link.set('name', 'base_link')

    visual = ET.SubElement(link, 'visual')
    visual.append(geometry)
    visual.append(material)

    collision = ET.SubElement(link, 'collision')
    collision.append(geometry)

    link.append(inertial)

    xmlstr = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="   ")
    Path(urdf_path).write_text(xmlstr)  # Write xml file
    return


def make_multivisual_urdf(urdf_path, out_base_dir, add_uv=True, texture_size=(100, 100)):
    urdf_path = Path(urdf_path)
    out_urdf_dir = Path(out_base_dir)
    meshes_dir = out_urdf_dir / 'patched_visual_meshes'
    meshes_dir.mkdir(exist_ok=True, parents=True)
    out_urdf_path = out_urdf_dir / 'patched_urdf' / urdf_path.name
    out_urdf_path.parent.mkdir(exist_ok=True, parents=True)
    root = ET.fromstring(Path(urdf_path).read_text())
    for link in root.iter('link'):
        visuals = list(link.iter('visual'))
        xml_visuals = []
        link_name = link.attrib['name']
        for visual in visuals:
            TLV = parse_origin(visual)
            geometry = visual.find('geometry')
            mesh = geometry.find('mesh')
            if mesh is None:
                continue
            filename = mesh.attrib['filename']
            filename = resolve_package_path(urdf_path, filename)
            visual_mesh = trimesh.load(filename)
            visuals = extract_mesh_visuals(visual_mesh)
            for mesh_id, (mesh, TVM) in enumerate(visuals):
                ply_path = meshes_dir / link_name / (filename.with_suffix('').name + f'_part={mesh_id}.ply')
                obj_path = ply_path.with_suffix('.obj')
                ply_path.parent.mkdir(exist_ok=True)
                keep_faces = []
                for face_id, face in enumerate(mesh.faces):
                    if len(np.unique(face)) == 3:
                        keep_faces.append(face_id)
                mesh.faces = mesh.faces[keep_faces]
                if mesh.visual.uv is not None:
                    mesh.export(obj_path)
                else:
                    mesh.visual = mesh.visual.to_color()
                    mesh.export(ply_path)
                    ply_to_obj(ply_path, obj_path)
                    ply_path.unlink()
                relative_obj_path = Path('../') / meshes_dir.name / link_name / obj_path.name
                xml_visual = ET.Element('visual')
                xml_geom = ET.SubElement(xml_visual, 'geometry')
                xml_mesh = ET.SubElement(xml_geom, 'mesh', attrib=dict(filename=str(relative_obj_path)))
                TLM = TLV @ np.array(TVM)
                attrib = dict(unparse_origin(TLM).attrib)
                xml_origin = ET.SubElement(xml_visual, 'origin', attrib=attrib)
                xml_material = visual.find('material')
                if xml_material is None:
                    xml_material = ET.SubElement(xml_visual, 'material', {'name': f'mat_part{np.random.randint(1e4)}'})
                    xml_color = ET.SubElement(xml_material, 'color', {'rgba': '1.0 1.0 1.0 1.0'})
                xml_visuals.append(xml_visual)
            link.remove(visual)

            for new_visual in xml_visuals:
                link.append(new_visual)
    xmlstr = minidom.parseString(ET.tostring(root).decode()).toprettyxml(indent="   ")
    Path(out_urdf_path).write_text(xmlstr)
    return out_urdf_path


if __name__ == '__main__':
    # base_dir = Path('/home/ylabbe/projects/robopose/deps/baxter-description/baxter_description/')
    # urdf_path = base_dir / 'urdf/baxter.urdf'

    # base_dir = Path('/home/ylabbe/projects/robopose/deps/kuka-description/iiwa_description')
    # urdf_path = base_dir / 'urdf/iiwa7.urdf'

    # base_dir = Path('/home/ylabbe/projects/robopose/deps/kuka-description/lbr_iiwa7_r800')
    # urdf_path = base_dir / 'urdf/lbr_iiwa7_r800.urdf'

    base_dir = Path('/home/ylabbe/projects/robopose/deps/panda-description/')
    urdf_path = base_dir / 'panda.urdf'

    make_multivisual_urdf(urdf_path, base_dir)
