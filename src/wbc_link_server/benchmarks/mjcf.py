"""XML-only adaptation needed by robosuite's shallow MJCF default loader."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET


def inline_defaults(root: ET.Element) -> None:
    """Expand HITCH's nested tag defaults and childclass inheritance in place.

    robosuite removes the <default> tree after expanding only top-level classes.
    Materializing HITCH's attributes first prevents lost geom and joint settings.
    """
    classes = {"main": {}}
    defaults = root.find("default")

    def collect(element, inherited):
        values = copy.deepcopy(inherited)
        for child in element:
            if child.tag != "default":
                values.setdefault(child.tag, {}).update(child.attrib)
        classes[element.get("class", "main")] = values
        for child in element:
            if child.tag == "default":
                collect(child, values)

    if defaults is not None:
        collect(defaults, {})
        root.remove(defaults)

    def expand(element, inherited):
        selected = element.attrib.pop("class", inherited)
        for name, value in classes[selected].get(element.tag, {}).items():
            element.attrib.setdefault(name, value)
        childclass = element.attrib.pop("childclass", inherited)
        for child in element:
            expand(child, childclass)

    expand(root, "main")


def camera_xml(xml: str, prefix="robot0_") -> str:
    """Strip the robot namespace in names and all references, never asset paths.

    This creates the HITCH camera's model only. The native evaluator retains its
    original names; neither element ordering nor numerical attributes change.
    """
    root = ET.fromstring(xml)
    asset_tags = {"mesh", "texture", "material", "hfield", "skin"}
    preserved = {"file", "class", "childclass", *asset_tags}
    for element in root.iter():
        for name, value in list(element.attrib.items()):
            if name in preserved or (name == "name" and element.tag in asset_tags):
                continue
            element.set(name, " ".join(token.removeprefix(prefix) for token in value.split()))
    return ET.tostring(root, encoding="unicode")
