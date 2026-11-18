"""Scoped certificates require registered semantic evidence, never pytest alone."""
from pathlib import Path
from typing import Literal
from pydantic import Field,model_validator
from core.experiments.strict_types import StrictResearchModel,strict_json_loads
from core.experiments.identity import research_hash,HASH_PATTERN,file_sha256
from core.experiments.archive import verify_archive_v3
from core.validation.health import verify_frozen_profile,profile_semantic,assess_criteria,diagnostic_metrics


class GateCertificate(StrictResearchModel):
    schema_version:Literal[1]=1
    gate_id:Literal['S0','S1','S2','S3','S4','S5','S6']
    status:Literal['PASS','FAIL','INCONCLUSIVE','NOT_APPLICABLE']
    scope_hash:str=Field(pattern=HASH_PATTERN)
    profile_hash:str=Field(pattern=HASH_PATTERN)
    source_hash:str=Field(pattern=HASH_PATTERN)
    evidence_hashes:list[str]
    claim_level:Literal['NONE','VALID_FOR_SYNTHETIC_TASK','CALIBRATED_FOR_SPECIFIED_STATS']
    limitations:list[str]

    @model_validator(mode='after')
    def _claims(self):
        import re
        if len(set(self.evidence_hashes))!=len(self.evidence_hashes) or any(not re.fullmatch(HASH_PATTERN,h) for h in self.evidence_hashes):raise ValueError('Invalid evidence identities')
        if self.status=='PASS' and (not self.evidence_hashes or self.claim_level=='NONE'):raise ValueError('PASS requires evidence and a scoped claim')
        if self.status!='PASS' and self.claim_level!='NONE':raise ValueError('Unpassed gate cannot confer eligibility')
        return self


def validate_health_run_scope(spec,run,planned_ids):
    if (spec.data_phase!='validation' or spec.random.replication_id!=run['replication_id']
        or spec.random.replication_id not in planned_ids):
        raise ValueError('GATE_ARCHIVE_VALIDATION_NAMESPACE_MISMATCH')


def evaluate_gate(gate_id,profile_path,evidence_root,*,witness_out=None):
    profile,freeze=verify_frozen_profile(Path(profile_path));root=Path(evidence_root)
    profile_hash=research_hash(profile_semantic(profile));evidence=[];limitations=[];status='INCONCLUSIVE'
    health_path=root/'health/health.json'
    if not health_path.exists():limitations.append('Independent frozen-profile health evidence is absent')
    else:
        report=strict_json_loads(health_path.read_text())
        if (report['namespace']!='validation' or report['source_hash']!=profile.source_hash
            or report['profile_hash']!=profile_hash or report['model_scope_hash']!=profile.model_scope_hash
            or set(report['replication_ids'])!=set(profile.validation_ids)):
            raise ValueError('GATE_HEALTH_SCOPE_MISMATCH')
        for run in report['runs']:
            verify_archive_v3(run['archive'])
            from core.experiments.research_cli import runtime_implementation_manifest
            from core.experiments.design_config import ResearchRunSpec
            effective=ResearchRunSpec.model_validate_json((Path(run['archive'])/'research_spec.json').read_text())
            validate_health_run_scope(effective,run,profile.validation_ids)
            if runtime_implementation_manifest(effective).source_hash!=profile.source_hash:
                raise ValueError('GATE_EXECUTION_IMPLEMENTATION_CHANGED')
            if file_sha256(Path(run['archive'])/'archive_manifest.json')!=run['manifest_sha256'] or diagnostic_metrics(Path(run['archive']))!=run['metrics']:
                raise ValueError('GATE_EVIDENCE_CONTENT_MISMATCH')
        if sorted(r['replication_id'] for r in report['runs'])!=sorted(profile.validation_ids):
            raise ValueError('GATE_INCOMPLETE_OR_DUPLICATED_VALIDATION_RUNS')
        checked=assess_criteria(profile,report['runs'])
        if checked!=report['criteria']:raise ValueError('GATE_CRITERIA_CHANGED')
        evidence.append(file_sha256(health_path))
        if gate_id=='S4':
            status='PASS' if checked and all(c['status']=='PASS' for c in checked) else 'FAIL' if any(c['status']=='FAIL' for c in checked) else 'INCONCLUSIVE'
            if status!='PASS':limitations.append('Policy response, trading health or precision is unresolved under frozen criteria')
        elif gate_id=='S5':
            # Synthetic qualification requires explicit provenance in every run,
            # a passing independent health scope, and no empirical substitution.
            from core.experiments.design_config import ResearchRunSpec
            synthetic=all(ResearchRunSpec.model_validate_json((Path(r['archive'])/'research_spec.json').read_text()).base.historical_input is None for r in report['runs'])
            status='PASS' if synthetic and checked and all(c['status']=='PASS' for c in checked) else 'INCONCLUSIVE'
            limitations.append('empirical_calibration=NOT_PERFORMED; scope is the specified synthetic task')
        else:
            from core.validation.semantic_checks import semantic_gate_witness
            from core.experiments.archive import durable_json
            witness=semantic_gate_witness(gate_id,report,root)
            witness.update(source_hash=profile.source_hash,profile_hash=profile_hash,gate_id=gate_id,
                required_scope=list(profile.required_gate_scopes[gate_id]))
            if witness_out is not None:
                destination=Path(witness_out)
                if destination.exists():raise ValueError('Semantic witness output is immutable')
                durable_json(destination,witness);evidence.append(file_sha256(destination))
                status=witness['status']
            else:limitations.append('Semantic witness was computed but has no durable artifact destination')
            if witness['status']!='PASS':limitations.append(witness.get('reason','Semantic witness is incomplete'))
    return GateCertificate(gate_id=gate_id,status=status,scope_hash=research_hash({'model_scope':profile.model_scope_hash,
        'gate':gate_id,'requirements':profile.required_gate_scopes[gate_id]}),profile_hash=profile_hash,
        source_hash=profile.source_hash,evidence_hashes=evidence,
        claim_level='VALID_FOR_SYNTHETIC_TASK' if status=='PASS' else 'NONE',limitations=limitations)


def require_certificate(certificates,*,source_hash,profile_hash,scope_hashes,profile_path,evidence_root):
    """Recompute applicability and evidence; cached PASS is never sufficient."""
    import tempfile
    checked=[GateCertificate.model_validate(c) for c in certificates]
    if len({c.gate_id for c in checked})!=len(checked):raise ValueError('DUPLICATE_GATE_CERTIFICATE')
    indexed={c.gate_id:c for c in checked}
    if not scope_hashes or not set(scope_hashes)<=set(indexed):raise ValueError('REQUIRED_GATE_CERTIFICATE_MISSING')
    with tempfile.TemporaryDirectory(prefix='cscwd-gate-recheck-') as folder:
        for gate,scope in scope_hashes.items():
            certificate=indexed[gate]
            if (certificate.status!='PASS' or certificate.source_hash!=source_hash or certificate.profile_hash!=profile_hash
                or certificate.scope_hash!=scope):raise ValueError('GATE_NOT_ELIGIBLE_FOR_REQUESTED_SCOPE')
            actual=evaluate_gate(gate,profile_path,evidence_root,witness_out=Path(folder)/(gate+'.json'))
            if actual!=certificate:raise ValueError('GATE_EVIDENCE_OR_SCOPE_CHANGED')
    return {'status':'ELIGIBLE_FOR_DECLARED_SCOPE','source_hash':source_hash,'profile_hash':profile_hash,
            'gate_scopes':dict(scope_hashes)}
