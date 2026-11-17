"""The two Responses providers supported by the Codex entry."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Provider:
    endpoint: str
    model: str


PROVIDERS = {
    "deepseek": Provider("https://api.deepseek.com/responses", "deepseek-flash"),
    "dpsk": Provider("https://ai.cnjsbc.com/v1/responses", "deepseek-v4.1-flash"),
}


def key_from_file(path: Path) -> str:
    """Accept a plain key or a document containing exactly one sk- credential."""
    keys = re.findall(r"(?<![\w-])sk-[A-Za-z0-9_-]+", path.read_text())
    if len(keys) != 1:
        raise ValueError("API key file must contain exactly one sk- credential")
    return keys[0]


def write_catalog(template: Path, destination: Path, provider: Provider) -> None:
    """Use the verified Codex capabilities with the selected model identity."""
    catalog = json.loads(template.read_text())
    model = catalog["models"][0]
    model["slug"] = provider.model
    model["display_name"] = provider.model
    destination.write_text(json.dumps(catalog, indent=2) + "\n")
