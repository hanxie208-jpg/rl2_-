import xml.etree.ElementTree as ET
from pathlib import Path

urdf_path = Path(__file__).resolve().parent / 'dog' / 'urdf' / 'dog.urdf'
root = ET.parse(urdf_path).getroot()

links = [link.attrib['name'] for link in root.findall('link')]
joints = [(joint.attrib['name'], joint.attrib['type'], joint.find('parent').attrib['link'], joint.find('child').attrib['link']) for joint in root.findall('joint')]

print('Links:')
for name in links:
    print('  -', name)

print('\nJoints:')
for name, jtype, parent, child in joints:
    print(f'  - {name} ({jtype}): {parent} -> {child}')

# Find possible mesh mismatches in foot links
print('\nFoot link visual meshes:')
for link in root.findall('link'):
    name = link.attrib['name']
    if name.endswith('_foot'):
        mesh = link.find('visual/geometry/mesh')
        collision = link.find('collision/geometry')
        mesh_file = mesh.attrib['filename'] if mesh is not None else 'none'
        coll_type = collision[0].tag if collision is not None and len(collision) else 'none'
        print(f'  - {name}: visual={mesh_file}, collision={coll_type}')
