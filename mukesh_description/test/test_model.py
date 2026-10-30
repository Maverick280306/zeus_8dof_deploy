"""
mukesh.urdf is generated and the deployed policy was trained on it. If it drifts
from the export + YAML, or stops describing the joints the protocol carries,
everything downstream is quietly about a different robot.
"""

import filecmp
import importlib.util
import os
import shutil
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, "mukesh_description")

spec = importlib.util.spec_from_file_location("nexus_proto",
                                              os.path.join(ROOT, "zeus_link", "zeus_link", "nexus_proto.py"))
proto = importlib.util.module_from_spec(spec)
import sys                                                  # noqa: E402
sys.modules["nexus_proto"] = proto
spec.loader.exec_module(proto)


def joints():
    root = ET.parse(os.path.join(PKG, "urdf", "mukesh.urdf")).getroot()
    return {j.get("name"): j.get("type") for j in root.findall("joint")}


def test_every_protocol_joint_is_in_the_model():
    # A joint the URDF lacks is rejected by robot_state_publisher on every message.
    j = joints()
    assert all(j.get(n) == "revolute" for n in proto.JOINT_NAMES)


def test_the_four_springs_are_in_the_model():
    j = joints()
    assert all(j.get(n + "_spring") == "revolute" for n in proto.SPRING_NAMES)


def test_urdf_is_regenerated(tmp_path):
    shutil.copytree(os.path.join(PKG, "config"), tmp_path / "config")
    (tmp_path / "urdf").mkdir()
    shutil.copy(os.path.join(PKG, "urdf", "mukesh_raw.urdf"), tmp_path / "urdf")
    os.symlink(os.path.join(PKG, "meshes"), tmp_path / "meshes")
    tool = os.path.join(ROOT, "zeus_description", "scripts", "clean_urdf.py")
    code = os.system(f'"{sys.executable}" "{tool}" --config "{tmp_path}/config/mukesh_model.yaml" > /dev/null')
    assert code == 0
    assert filecmp.cmp(tmp_path / "urdf" / "mukesh.urdf", os.path.join(PKG, "urdf", "mukesh.urdf"),
                       shallow=False)
