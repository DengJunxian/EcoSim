"""Strict, immutable record foundations shared by research schemas.

Extracted without behavior changes in Phase 2 to break schema import cycles.
The legacy v1 configuration loader remains independent. Research input is a
finite JSON value tree; YAML is restricted to that same scalar vocabulary.
Numbers are never repaired or converted from strings. A declared float needs
a binary64 Python float, and a declared integer needs an actual int (not bool).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal, TypeVar
from collections.abc import Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResearchInputError(ValueError):
    """The input cannot represent a strict research record."""


def _immutable(*args: Any, **kwargs: Any) -> None:
    raise TypeError("Research record containers are immutable")


class _FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = _immutable
    pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = _immutable
    update = __ior__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _check_tree(value: Any, path: str = "$", active: set[int] | None = None) -> None:
    """Reject nonfinite values, foreign objects, non-string keys and cycles."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ResearchInputError(f"Nonfinite number at {path}")
        return
    if isinstance(value, StrictResearchModel):
        # Instances are already validated and recursively frozen; Pydantic
        # also revalidates them when they occur in a declared model field.
        return
    if not isinstance(value, (dict, list, tuple)):
        raise ResearchInputError(f"Non-JSON value at {path}: {type(value).__name__}")
    active = set() if active is None else active
    if id(value) in active:
        raise ResearchInputError(f"Cyclic input at {path}")
    active.add(id(value))
    try:
        is_object = isinstance(value, dict)
        children = value.items() if is_object else enumerate(value)
        for key, item in children:
            if is_object and type(key) is not str:
                raise ResearchInputError(f"Object keys must be strings at {path}")
            item_type = type(item)
            if item is None or item_type in (str, bool, int) or (item_type is float and math.isfinite(item)):
                continue
            item_path = f"{path}.{key}" if is_object else f"{path}[{key}]"
            _check_tree(item, item_path, active)
    finally:
        active.remove(id(value))


def _check_scalar_preservation(raw: Any, validated: Any) -> None:
    """Reject scalar coercion after Pydantic resolves unions and containers.

Pydantic strict floats accept ints, and numeric Literals compare equal to bool
or 1.0. Comparing the actual result closes those routes for list, Sequence,
Mapping, tuple, TypedDict and union fields without guessing the selected type.
Nested StrictResearchModel objects validate their own fields in the same way.
"""
    if raw is None or type(raw) in (str, bool, int, float):
        if type(raw) is not type(validated):
            raise ValueError(
                f"Scalar coercion is forbidden: {type(raw).__name__} to {type(validated).__name__}"
            )
    elif isinstance(raw, Mapping) and isinstance(validated, Mapping):
        for key, value in raw.items():
            if key in validated:
                _check_scalar_preservation(value, validated[key])
    elif isinstance(raw, (list, tuple)) and isinstance(validated, Sequence):
        for value, result in zip(raw, validated):
            _check_scalar_preservation(value, result)


class StrictResearchModel(BaseModel):
    """Finite records with exact scalar types and recursively frozen containers.

    Validated list/dict containers retain their JSON shapes and normal read APIs.
    Nested model fields must also derive from this class; ordinary/legacy
    BaseModel instances are rejected, not trusted as prevalidated input.
Normal mutation methods and model_copy(update=...) cannot bypass validation.
This is an application invariant, not a sandbox against object.__setattr__.
"""

    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, allow_inf_nan=False,
        validate_default=True, revalidate_instances="always",
    )

    @model_validator(mode="before")
    @classmethod
    def _finite_input(cls, value: Any) -> Any:
        _check_tree(value)
        return value

    @field_validator("*", mode="wrap")
    @classmethod
    def _exact_scalars(cls, value: Any, handler: Any) -> Any:
        result = handler(value)
        _check_scalar_preservation(value, result)
        return result

    @model_validator(mode="after")
    def _freeze_containers(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            _check_tree(value, f"$.{name}")
            object.__setattr__(self, name, _freeze(value))
        return self

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any):
        if kwargs.get("strict") is False or kwargs.get("extra") not in (None, "forbid"):
            raise ResearchInputError("Research validation cannot relax strictness or extra fields")
        return super().model_validate(obj, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any):
        return cls.model_validate(strict_json_loads(json_data), **kwargs)

    @classmethod
    def model_validate_strings(cls, obj: Any, **kwargs: Any):
        raise ResearchInputError("String-to-scalar coercion is forbidden for research records")

    @classmethod
    def model_construct(cls, _fields_set=None, **values: Any):
        raise ResearchInputError("Research records require validation; model_construct is unavailable")

    def model_copy(self, *, update=None, deep=False):
        payload = self.model_dump(round_trip=True)
        if update:
            payload.update(update)
        return type(self).model_validate(payload)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResearchInputError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ResearchInputError(f"Nonfinite numeric constant: {value}")


def _finite_float(text: str) -> float:
    result = float(text)
    if not math.isfinite(result):
        raise ResearchInputError("Numeric literal exceeds finite binary64 range")
    return result


def strict_json_loads(text: str | bytes | bytearray) -> Any:
    """Decode one JSON document, preserving numbers and null without repair."""
    try:
        value = json.loads(text, object_pairs_hook=_object_without_duplicates,
                           parse_constant=_reject_constant, parse_float=_finite_float)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResearchInputError(f"Invalid JSON: {exc}") from exc
    _check_tree(value)
    return value


class _StrictYamlLoader(yaml.SafeLoader):
    # Do not mutate SafeLoader's shared resolver table or the legacy loader.
    yaml_implicit_resolvers = {}

    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent) or self.peek_event().anchor is not None:
            raise ResearchInputError("YAML anchors/aliases are not research values; expand explicitly")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ResearchInputError("Expected a YAML mapping")
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if type(key) is not str:
                raise ResearchInputError("YAML mapping keys must be strings")
            if key == "<<":
                raise ResearchInputError("YAML merge keys are not supported in research input")
            if key in result:
                raise ResearchInputError(f"Duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_StrictYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:null", re.compile(r"^(?:null|~|)$"), ["n", "~", ""])
_StrictYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), ["t", "f"])
_StrictYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int", re.compile(r"^-?(?:0|[1-9][0-9]*)$"), list("-0123456789"))
_StrictYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)$"),
    list("-0123456789"))
_StrictYamlLoader.add_implicit_resolver(
    "!nonfinite", re.compile(r"^[+-]?\.?(?:nan|inf(?:inity)?)$", re.IGNORECASE),
    list("+-.nNiI"))
_StrictYamlLoader.add_constructor("!nonfinite", lambda loader, node: _reject_constant(node.value))
_StrictYamlLoader.add_constructor(
    "tag:yaml.org,2002:float", lambda loader, node: _finite_float(loader.construct_scalar(node)))


def strict_yaml_loads(text: str | bytes) -> Any:
    """Decode YAML with JSON-style scalars, no aliases, merges or custom tags.

Unquoted yes/on/dates remain strings, not PyYAML's implicit bool/datetime.
Explicit tags are rejected so !!int cannot repair a quoted integer string.
"""
    try:
        for token in yaml.scan(text):
            if isinstance(token, yaml.tokens.TagToken):
                raise ResearchInputError("Explicit YAML tags are forbidden")
        value = yaml.load(text, Loader=_StrictYamlLoader)
    except yaml.YAMLError as exc:
        raise ResearchInputError(f"Invalid YAML: {exc}") from exc
    _check_tree(value)
    return value


ResearchModelT = TypeVar("ResearchModelT", bound=StrictResearchModel)


def load_research_model(path: str | Path, model_type: type[ResearchModelT]) -> ResearchModelT:
    """Read an explicitly typed research model; never guess or repair formats."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = strict_json_loads(content)
    elif path.suffix.lower() in (".yaml", ".yml"):
        payload = strict_yaml_loads(content)
    else:
        raise ResearchInputError("Research inputs must use .json, .yaml or .yml")
    if not isinstance(model_type, type) or not issubclass(model_type, StrictResearchModel):
        raise TypeError("model_type must be a StrictResearchModel subclass")
    return model_type.model_validate(payload)

