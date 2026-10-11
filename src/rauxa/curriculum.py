"""Difficulty curricula, a simplified version of reasoning-gym's ``BaseCurriculum``.

A curriculum owns a set of attributes, each with an ordered list of levels. Each level is
the value assigned to one config field; ``generate_configuration`` builds a config from the
current level of every attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScalarAttributeDefinition:
    name: str
    field_name: str
    levels: tuple[Any, ...]
    description: str = ""


class BaseCurriculum:
    def __init__(self, name: str, config_cls: type, attributes: list[ScalarAttributeDefinition]):
        self.name = name
        self.config_cls = config_cls
        self.attributes = {attr.name: attr for attr in attributes}
        self._levels = {attr.name: 0 for attr in attributes}

    def get_attr_level(self, attr_name: str) -> int:
        return self._levels[attr_name]

    def set_attr_level(self, attr_name: str, level: int) -> None:
        attr = self.attributes[attr_name]
        if not 0 <= level < len(attr.levels):
            raise ValueError(f"level {level} out of range for '{attr_name}' (0..{len(attr.levels) - 1})")
        self._levels[attr_name] = level

    def increment_attr_level(self, attr_name: str) -> bool:
        """Raise ``attr_name`` one level; return False if it is already at the top."""
        if self._levels[attr_name] + 1 >= len(self.attributes[attr_name].levels):
            return False
        self._levels[attr_name] += 1
        return True

    def decrement_attr_level(self, attr_name: str) -> bool:
        if self._levels[attr_name] == 0:
            return False
        self._levels[attr_name] -= 1
        return True

    def generate_configuration(self, base_value: dict[str, Any] | None = None) -> Any:
        values = dict(base_value or {})
        for name, attr in self.attributes.items():
            values[attr.field_name] = attr.levels[self._levels[name]]
        config = self.config_cls(**values)
        config.validate()
        return config
