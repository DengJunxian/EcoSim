"""Data validation and explicitly bounded calibration entry points."""
import argparse
import json
from pathlib import Path

from core.validation.calibration import CalibrationPlan, data_check, fit_calibration
from core.experiments.strict_types import load_research_model


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    data=commands.add_parser('data-check');data.add_argument('--manifest',required=True);data.add_argument('--out',required=True)
    fit=commands.add_parser('calibrate');fit.add_argument('--plan',required=True);fit.add_argument('--out',required=True)
    health=commands.add_parser('health');health.add_argument('--suite',required=True);health.add_argument('--profile',required=True)
    health.add_argument('--out',required=True);health.add_argument('--workers',type=int,default=1);health.add_argument('--development',action='store_true')
    freeze=commands.add_parser('freeze-profile');freeze.add_argument('--draft',required=True)
    freeze.add_argument('--development-evidence',required=True);freeze.add_argument('--out',required=True)
    for name in ('gate','check-all'):
        gate=commands.add_parser(name);gate.add_argument('--profile',required=True);gate.add_argument('--evidence-root',required=True);gate.add_argument('--out',required=True)
        if name=='gate':gate.add_argument('--gate',choices=[f'S{i}' for i in range(7)],required=True)
    args=parser.parse_args(argv);output=Path(args.out)
    if args.command in {'health','freeze-profile','gate','check-all'}:
        return health_main(args)
    # Existing evidence is never silently replaced by a second attempt.
    output.mkdir(parents=True,exist_ok=True)
    name='data_check.json' if args.command=='data-check' else 'calibration_receipt.json'
    if (output/name).exists():
        print(json.dumps({'status':'INVALID_INPUT','reason':'OUTPUT_ALREADY_EXISTS; choose a fresh attempt directory'}))
        return 2
    try:
        if args.command=='data-check':
            result=data_check(args.manifest);code=0
        else:
            plan_path=Path(args.plan).resolve()
            plan=load_research_model(plan_path,CalibrationPlan)
            result=fit_calibration(plan,base_dir=plan_path.parent)
            code=3 if result['status']=='NOT_PERFORMED' else 0
        (output/name).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(result,ensure_ascii=False,allow_nan=False));return code
    except (OSError,ValueError,TypeError,KeyError) as exc:
        result={'status':'INVALID_INPUT','error_type':type(exc).__name__,'reason':str(exc)}
        (output/name).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(result,ensure_ascii=False));return 2


def health_main(args):
    from core.validation.health import GateProfile,HealthSuite,run_health,freeze_profile,verify_frozen_profile
    from core.validation.gates import evaluate_gate
    from core.experiments.research_cli import engineering_budget
    from core.experiments.archive import durable_json,ArtifactIntegrityError
    try:
        if args.command=='freeze-profile':
            result=freeze_profile(args.draft,args.development_evidence,args.out)
            code=0 if result['status']=='FROZEN' else 3
        elif args.command=='health':
            suite=load_research_model(args.suite,HealthSuite)
            profile,freeze_receipt=(load_research_model(args.profile,GateProfile),None) if args.development else verify_frozen_profile(Path(args.profile))
            output=Path(args.out).resolve()
            if output.exists():raise ValueError('Health attempt output already exists')
            budget=engineering_budget(output.parent/(output.name+'-budget.json'),run_count=sum(len(c.replication_ids) for c in suite.cases))
            result=run_health(suite,profile,base_dir=Path(args.suite).resolve().parent,output=output,
                reservation_port=budget,workers=args.workers,development=args.development,freeze_receipt=freeze_receipt)
            code=0 if result['status'] in {'DEVELOPMENT_ONLY','PASS'} else 3
        else:
            output=Path(args.out);output.mkdir(parents=True,exist_ok=False)
            ids=[args.gate] if args.command=='gate' else [f'S{i}' for i in range(7)]
            certificates=[evaluate_gate(g,args.profile,args.evidence_root,witness_out=output/(g+'-witness.json')).model_dump(mode='json') for g in ids]
            result={'certificates':certificates,'status':'PASS' if all(c['status']=='PASS' for c in certificates) else 'INCONCLUSIVE'}
            durable_json(output/'certificates.json',result);code=0 if result['status']=='PASS' else 3
        print(json.dumps(result,ensure_ascii=False,allow_nan=False));return code
    except ArtifactIntegrityError as exc:
        print(json.dumps({'status':'INVALID_ARTIFACT','reason':str(exc)},ensure_ascii=False));return 5
    except (ValueError,OSError,TypeError,KeyError,RuntimeError) as exc:
        print(json.dumps({'status':'INVALID_OR_MISSING_INPUT','error_type':type(exc).__name__,'reason':str(exc)},ensure_ascii=False))
        if isinstance(exc,RuntimeError):return 4
        return 3 if isinstance(exc,FileNotFoundError) or 'PROFILE_NOT_FROZEN' in str(exc) or type(exc).__name__=='BudgetExceeded' else 2


if __name__=='__main__':
    raise SystemExit(main())
