"""Pure XML contract tests; no simulator, checkpoint, or API key is loaded."""

import unittest
import xml.etree.ElementTree as ET

from wbc_link_server.benchmarks.mjcf import camera_xml, inline_defaults


class MJCFTests(unittest.TestCase):
    def test_nested_defaults_retain_physics_and_explicit_overrides(self):
        root = ET.fromstring('''<mujoco><default><joint damping=".1"/>
          <default class="g1"><site group="5"/>
            <default class="collision"><geom contype="1" conaffinity="1" group="3" type="capsule"/>
              <default class="foot"><geom size=".01"/></default>
            </default>
            <default class="visual"><geom contype="0" conaffinity="0" group="2"/></default>
          </default></default>
          <worldbody><body name="pelvis" childclass="g1">
            <joint name="hip" damping="0"/><site name="palm"/>
            <geom name="sole" class="foot" size=".02"/>
            <geom name="mesh" class="visual"/>
          </body></worldbody></mujoco>''')
        inline_defaults(root)
        self.assertIsNone(root.find("default"))
        self.assertEqual(root.find(".//joint").get("damping"), "0")
        self.assertEqual(root.find(".//site").get("group"), "5")
        self.assertEqual(root.find(".//geom[@name='sole']").attrib,
                         {"name": "sole", "size": ".02", "contype": "1", "conaffinity": "1",
                          "group": "3", "type": "capsule"})
        self.assertEqual(root.find(".//geom[@name='mesh']").get("contype"), "0")
        self.assertTrue(all("class" not in e.attrib and "childclass" not in e.attrib for e in root.iter()))

    def test_camera_names_and_references_change_without_order_or_path_changes(self):
        xml = '''<mujoco><asset><mesh name="robot0_hand" file="/assets/robot0_hand.stl"/></asset>
          <worldbody><body name="robot0_pelvis"><freejoint name="robot0_root"/>
            <body name="robot0_arm"><joint name="robot0_shoulder"/>
              <geom name="robot0_hand_geom" mesh="robot0_hand"/><site name="robot0_palm"/>
            </body></body><body name="can"><freejoint name="can_free"/></body></worldbody>
          <actuator><motor name="robot0_shoulder_motor" joint="robot0_shoulder"/></actuator>
          <tendon><spatial name="robot0_cable"><site site="robot0_palm"/></spatial></tendon>
          <contact><exclude body1="robot0_pelvis" body2="robot0_arm"/></contact>
          <sensor><framepos name="robot0_pose" objtype="site" objname="robot0_palm"/></sensor>
        </mujoco>'''
        root = ET.fromstring(camera_xml(xml))
        self.assertEqual([e.get("name") for e in root.iter() if e.tag in ("joint", "freejoint")],
                         ["root", "shoulder", "can_free"])
        self.assertEqual(root.find(".//asset/mesh").get("file"), "/assets/robot0_hand.stl")
        self.assertEqual(root.find(".//asset/mesh").get("name"), "robot0_hand")
        self.assertEqual(root.find(".//geom").get("mesh"), "robot0_hand")
        self.assertEqual(root.find(".//motor").get("joint"), "shoulder")
        self.assertEqual(root.find(".//spatial/site").get("site"), "palm")
        self.assertEqual(root.find(".//exclude").get("body1"), "pelvis")
        self.assertEqual(root.find(".//framepos").get("objname"), "palm")

    def test_camera_translation_keeps_tiny_scene_mass_precision(self):
        xml = '<mujoco><worldbody><body name="switch"><geom name="visual" mass="1e-08"/></body></worldbody></mujoco>'
        self.assertEqual(ET.fromstring(camera_xml(xml)).find(".//geom").get("mass"), "1e-08")


if __name__ == "__main__":
    unittest.main()
