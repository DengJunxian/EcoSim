"""Typed scientific identities, separate from legacy log redaction hashes.

research-canonical-v1 uses a version wrapper around a tagged JSON tree. Paths
and token/auth-like field names are ordinary semantic values here. A caller
must apply its declared schema/projection before hashing, never a secret filter.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from core.experiments.strict_types import ResearchInputError, StrictResearchModel

IDENTITY_VERSION = "research-canonical-v1"
HASH_PATTERN = r"^[0-9a-f]{64}$"


class ResearchIntegrityError(ValueError):
    """A referenced file or content graph disagrees with its declared identity."""


def _tagged(value: Any, active: set[int]) -> list:
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is str:
        return ["string", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise ResearchInputError("Scientific identity requires finite binary64 values")
        return ["f64", (0.0 if value == 0.0 else value).hex()]
    if isinstance(value, StrictResearchModel):
        # The validated schema owns JSON array representation (including tuple
        # fields inherited from v1). Raw Python tuples/objects remain illegal.
        return _tagged(value.model_dump(mode="json"), active)
    if not isinstance(value, (dict, list)):
        raise ResearchInputError(f"Scientific identity rejects {type(value).__name__}")
    if id(value) in active:
        raise ResearchInputError("Scientific identity rejects cyclic content")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ResearchInputError("Scientific identity object keys must be strings")
            return ["dict", [[key, _tagged(value[key], active)] for key in sorted(value)]]
        return ["list", [_tagged(item, active) for item in value]]
    finally:
        active.remove(id(value))


def canonical_research_bytes(value: Any) -> bytes:
    """Encode a validated JSON semantic tree without coercion or field removal.

    The bytes are UTF-8 JSON ``[version, tagged-tree]``, with ensure_ascii=False
    and separators=(',', ':'). Object key order is Unicode code-point order.
    JSON text is a string value, not implicitly parsed; use strict_json_loads
    at the input boundary to reject duplicate keys before invoking this API.
    """
    tree = [IDENTITY_VERSION, _tagged(value, set())]
    try:
        return json.dumps(tree, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except UnicodeError as exc:
        raise ResearchInputError("Scientific strings must have a valid UTF-8 encoding") from exc


def research_hash(value: Any) -> str:
    return hashlib.sha256(canonical_research_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash actual bytes, not a path name or a future receipt."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute() or ".." in path.parts
            or path.as_posix() != value or value in (".", "")):
        raise ResearchInputError("Implementation paths must be canonical repository-relative POSIX paths")
    return value


class ImplementationFile(StrictResearchModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=HASH_PATTERN)
    bytes: int = Field(ge=0)
    role: Literal["code", "resource", "prompt", "configuration"]

    @model_validator(mode="after")
    def _path(self):
        _relative_path(self.path)
        return self


class ImplementationManifest(StrictResearchModel):
    schema_version: Literal[1] = 1
    identity_version: Literal["research-canonical-v1"] = IDENTITY_VERSION
    files: list[ImplementationFile] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_sorted(self):
        paths = [entry.path for entry in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("Implementation entries must have unique sorted paths")
        return self

    @property
    def source_hash(self) -> str:
        return research_hash(self)


def build_implementation_manifest(repo_root: str | Path, paths: list[str], *,
                                  roles: dict[str, str] | None = None,
                                  manifest_path: str | None = None) -> ImplementationManifest:
    """Read an explicit implementation/resource closure without output globbing.

    Code that executes or resources that affect it must be listed by the
    composition root. This function does not infer that all docs/prompts are
    irrelevant. A manifest cannot include itself or runtime output directories.
    """
    if not isinstance(paths, list) or any(type(path) is not str for path in paths):
        raise ResearchInputError("Implementation paths require an explicit string list")
    if len(paths) != len(set(paths)):
        raise ResearchInputError("Duplicate implementation path")
    roles = {} if roles is None else roles
    if set(roles) - set(paths):
        raise ResearchInputError("A resource role refers to an unlisted implementation path")
    root = Path(repo_root).resolve(strict=True)
    self_path = _relative_path(manifest_path) if manifest_path is not None else None
    entries = []
    for relative in sorted(paths):
        _relative_path(relative)
        if relative == self_path or PurePosixPath(relative).parts[0] in ("artifacts", "outputs", "runs"):
            raise ResearchInputError("Implementation identity cannot include itself or runtime outputs")
        actual = (root / relative).resolve(strict=True)
        if not actual.is_relative_to(root) or not actual.is_file():
            raise ResearchInputError("Implementation references must resolve to files inside the repository")
        role = roles.get(relative, "code" if actual.suffix in (".py", ".cpp", ".h", ".hpp") else "resource")
        payload = actual.read_bytes()
        entries.append(ImplementationFile(path=relative, sha256=hashlib.sha256(payload).hexdigest(),
                                           bytes=len(payload), role=role))
    return ImplementationManifest(files=entries)


def verify_implementation_manifest(repo_root: str | Path, manifest: ImplementationManifest) -> None:
    if not isinstance(manifest, ImplementationManifest):
        raise TypeError("Expected an ImplementationManifest")
    actual = build_implementation_manifest(repo_root, [entry.path for entry in manifest.files],
                                            roles={entry.path: entry.role for entry in manifest.files})
    if actual != manifest:
        raise ResearchIntegrityError("Implementation contents no longer match the source manifest")


# Exact top-level archive metadata only. Nested keys with the same spelling and
# every other field (including resource paths, economics, clocks and validity)
# remain scientific content. Per-record adapters call this before aggregation.
SCIENTIFIC_METADATA_FIELDS = frozenset({
    "run_id", "attempt_id", "output_dir", "started_at", "ended_at", "generated_at_utc",
})


def scientific_projection(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ResearchInputError("Scientific projection expects one explicit record")
    canonical_research_bytes(record)  # Validate before removing even metadata.
    return {key: value for key, value in record.items() if key not in SCIENTIFIC_METADATA_FIELDS}


class IdentityNode(StrictResearchModel):
    """A semantic body with already-resolved content dependencies, never itself."""
    kind: str = Field(min_length=1)
    body: dict[str, Any]
    dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _dependencies(self):
        import re
        if (len(self.dependencies) != len(set(self.dependencies))
                or any(re.fullmatch(HASH_PATTERN, digest) is None for digest in self.dependencies)):
            raise ValueError("Identity dependencies require distinct SHA-256 references")
        canonical_research_bytes(self.body)
        return self

    @property
    def content_hash(self) -> str:
        return research_hash(self)


def validate_identity_graph(nodes: dict[str, IdentityNode]) -> None:
    """Resolve every reference and reject missing, mismatched or cyclic nodes."""
    if not isinstance(nodes, dict) or not nodes:
        raise ResearchIntegrityError("Identity graph must contain explicit content")
    for digest, node in nodes.items():
        if not isinstance(node, IdentityNode) or digest != node.content_hash:
            raise ResearchIntegrityError("Identity node key does not match its content")
        if any(dependency not in nodes for dependency in node.dependencies):
            raise ResearchIntegrityError("Unresolved identity dependency")
    visited, active = set(), set()
    def visit(digest):
        if digest in active:
            raise ResearchIntegrityError("Cyclic identity references")
        if digest in visited:
            return
        active.add(digest)
        for dependency in nodes[digest].dependencies:
            visit(dependency)
        active.remove(digest)
        visited.add(digest)
    for digest in nodes:
        visit(digest)
