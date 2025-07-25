import xml.etree.ElementTree as ET
import numpy as np

def extract_link_data(filename:str="./resources/robots/g1_description/g1_29dof_rev_1_0.xml"):
    tree = ET.parse(filename)
    root = tree.getroot()

    link_data = {}

    # Iterate over all 'body' elements in 'worldbody'
    for body in root.find("worldbody").iter("body"):
        name = body.get("name", "")
        if "_link" in name or (name == 'pelvis'):  # Filter only the link bodies
            inertial = body.find("inertial")
            if inertial is not None:
                mass = float(inertial.get("mass", "0"))
                pos = np.array(list(map(float, inertial.get("pos", "0 0 0").split())))
                quat = np.array(list(map(float, inertial.get("quat", "0 0 0 0").split())))
            else:
                mass = 0.0
                pos = np.array([0.0, 0.0, 0.0])
                quat = np.array([0.0, 0.0, 0.0, 0.0])

            link_data[name] = {
                "mass": mass,
                "pos": pos,
                "quat": quat
            }

    return link_data

def main():
    print(extract_link_data())

if __name__ == '__main__':
    main()
