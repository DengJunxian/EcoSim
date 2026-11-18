"""Minimal local export. Durable crash recovery/commit markers belong to WP08."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import civitas
from core.experiments import identity, strict_types

from civitas.contracts.base import content_hash
from civitas.contracts.config import RunConfig
from civitas.contracts.run import FinancialRunManifest, PortfolioRunManifest, RunManifest
from civitas.core.rng import ALGORITHM
from civitas.core.world import WorldKernel


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def implementation_identity(*, financial=False) -> dict:
    package = Path(civitas.__file__).resolve().parent
    sources = {"civitas/" + path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(package.rglob("*.py"))}
    for module in (identity, strict_types):
        sources[module.__name__.replace(".", "/") + ".py"] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if financial:
        import config
        import agents
        root = Path(config.__file__).resolve().parent
        paths = [Path(config.__file__)]
        for directory in (Path(agents.__file__).parent, root / "core", root / "core/experiments",
                          root / "core/exchange", root / "core/society", root / "core/interventions"):
            paths.extend(directory.glob("*.py"))
        paths.extend((root / "data").glob("market_composition.*"))
        for path in sorted(set(paths)):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = None
    dirty = None
    try:
        result = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"], text=True,
                                capture_output=True, check=True, timeout=5)
        commit = result.stdout.strip()
        status = subprocess.run(["git", "-C", str(package), "status", "--porcelain"], text=True,
                                capture_output=True, check=True, timeout=5)
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass  # Installed wheels retain the exact source hash without assuming a Git checkout.
    dependencies = {name: importlib.metadata.version(name) for name in ("pydantic", "pydantic_core", "PyYAML")}
    if financial:
        dependencies.update({name: importlib.metadata.version(name) for name in ("numpy", "pandas", "networkx")})
    dependencies.update(python=platform.python_version(), platform=platform.platform(), machine=platform.machine())
    return {"code_commit": commit, "code_dirty": dirty, "source_hash": content_hash(sources), "source_files": sources,
            "dependencies": dependencies, "dependency_hash": content_hash(dependencies)}


def mechanism_versions(world: WorldKernel) -> dict:
    policy_versions = getattr(world.policy, 'mechanism_versions', {})
    mechanism_versions = {m.spec.mechanism_id: m.spec.version for m in world.registry.mechanisms}
    if world.observation_effect is not None:
        mechanism_versions[world.observation_effect.mechanism_id] = world.observation_effect.version
    if any(k in mechanism_versions and mechanism_versions[k] != v for k,v in policy_versions.items()):
        raise ValueError('Policy and registry mechanism versions conflict')
    mechanism_versions.update(policy_versions)
    return mechanism_versions


async def run_to_directory(world: WorldKernel, config: RunConfig, destination: Path) -> RunManifest:
    destination.mkdir(parents=True, exist_ok=False)
    financial = world.snapshot.schema_version == 2
    portfolio = world.snapshot.schema_version == 3
    versions = mechanism_versions(world)
    implementation = implementation_identity(financial=financial or 'reference_value_v1' in versions)
    initial_hash = content_hash(world.snapshot)
    manifest_type = {1: RunManifest, 2: FinancialRunManifest, 3: PortfolioRunManifest}[world.snapshot.schema_version]
    manifest = manifest_type(status="running", code_commit=implementation["code_commit"], code_dirty=implementation["code_dirty"],
        source_hash=implementation["source_hash"], dependency_hash=implementation["dependency_hash"],
        config_hash=content_hash(config), initial_state_hash=initial_hash, final_state_hash=initial_hash,
        scenario_id=world.scenario.scenario_id, scenario_version=world.scenario.version,
        policy_id=world.policy.policy_id, policy_version=world.policy.version,
        mechanism_versions=versions,
        rng_algorithm="legacy-python-random+numpy-RandomState" if financial else ALGORITHM)
    write_json(destination / "manifest.json", manifest.model_dump())
    write_json(destination / "config.json", config.model_dump())
    policy_config = getattr(world.policy, 'configuration', None)
    if policy_config is not None:
        write_json(destination / 'policy-config.json', policy_config.model_dump() if hasattr(policy_config, 'model_dump') else policy_config)
    write_json(destination / "implementation.json", implementation)
    if financial or portfolio:
        lock = "".join(f"{d.metadata['Name']}=={d.version}\n" for d in sorted(importlib.metadata.distributions(),
                       key=lambda d: d.metadata.get("Name", "")) if d.metadata.get("Name"))
        (destination / "requirements-lock.txt").write_text(lock, encoding="utf-8")
        write_json(destination / "dependency-lock.json", {"sha256": hashlib.sha256(lock.encode()).hexdigest()})
    write_json(destination / "initial_state.json", world.snapshot.model_dump())
    try:
        await world.run(config.ticks)
    except BaseException as exc:
        manifest = manifest.model_copy(update={"status": "failed", "final_state_hash": content_hash(world.snapshot),
                                               "error": f"{type(exc).__name__}: {exc}"})
        write_json(destination / "state.json", world.snapshot.model_dump())
        write_json(destination / "manifest.json", manifest.model_dump())
        if financial or portfolio:
            write_json(destination / "failures.json", list(world.failures))
            write_json(destination / "settlement-diagnostics.json", (world.scenario.adapter if financial else world.scenario.market).diagnostics)
            write_json(destination / "invocations.json", [r.model_dump() for r in world.invocations])
        raise
    snapshot = world.snapshot
    write_json(destination / "state.json", snapshot.model_dump())
    write_json(destination / "metrics.json", world.scenario.metrics(world._context(snapshot)))
    for name in ("events", "decisions", "receipts", "messages"):
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for record in getattr(snapshot, name):
                stream.write(record.model_dump_json() + "\n")
    if portfolio:
        write_json(destination / "population.json", world.scenario.population.model_dump())
        with (destination / "ledger-journal.jsonl").open("w", encoding="utf-8") as stream:
            for record in world.scenario.market.ledger.journal:
                stream.write(record.model_dump_json() + "\n")
    manifest = manifest.model_copy(update={"status": "completed", "final_state_hash": content_hash(snapshot)})
    write_json(destination / "manifest.json", manifest.model_dump())
    return manifest
